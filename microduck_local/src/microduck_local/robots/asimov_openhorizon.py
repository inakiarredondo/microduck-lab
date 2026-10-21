"""openhorizon-labs' Asimov get-up, driven from OUR 78-float observation.

Their policy takes 375 inputs and ours take 78, which reads like a contract
mismatch and is not one. Measured against both publications, every quantity
their observation needs is ALREADY in Menlo's 78-float layout, on the same
scale, in the same joint order:

| their term          | dim | our slot            | scale (theirs / ours) |
|---------------------|----:|---------------------|-----------------------|
| `base_ang_vel`      |   3 | `OBS_ANG_VEL` 0:3   | 0.25 / 0.25           |
| `projected_gravity` |   3 | `OBS_GRAVITY` 3:6   | 1.0  / 1.0            |
| `joint_pos`         |  23 | `OBS_JOINT_POS` 9:32| 1.0  / 1.0            |
| `joint_vel`         |  23 | `OBS_JOINT_VEL` 32:55| 0.1 / 0.1            |
| `actions`           |  23 | `OBS_LAST_ACTION` 55:78| 1.0 / 1.0          |

and `obs_joint_order` is `asimov_walk.slot_order()` exactly, `action.joint_order`
is `JOINT_NAMES` exactly, `default_joint_pos` matches `DEFAULT_POSE` to 0.0000
rad over all 23 joints, `action.scale` is 0.25 either way, and their
`physics_dt`/`rate_hz` are our `PHYSICS_DT`/50 Hz. Our 78 carries THREE MORE
floats than their 75 — the velocity command, which a get-up has no use for.

**So the whole of the gap is HISTORY**: 5 frames instead of 1 (375 = 75 x 5,
term-major, oldest->newest within each term). That is the single finding worth
taking from this file, because it says what a contract-preserving get-up here
actually needs, and it is not a different sensor suite.

This adapter therefore needs no MuJoCo state and no env: it is a stateful
callable with the lab's own `(obs78) -> action23` signature that keeps the ring
buffer their `Policy.step` keeps. Driven off OUR published observation rather
than off `data.qpos` on purpose — a policy on the lab stage then sees the same
noise and the same command delay ours does, which is what makes the two
comparable side by side. `tests/test_asimov_openhorizon.py` checks it
reproduces their own runner's actions to 1e-6 over a full get-up.

What is NOT the same, and must be said when quoting a number from here: their
training MJCF carries a COLLISION OVERLAY this repo's does not (their
`build_model` removes 4 geoms and adds 10 -- torso, pelvis, head, upper arms,
hands, shoulders -- because "the upstream MJCF collision shapes were tuned for
walking, not for lying on the floor"). Running their policy on our bare model
is their controller on a slightly different body. `probe_openhorizon_lab.py`
measures both.

Licence: Apache-2.0, `openhorizon-labs/asimov1-getup-safefall` — fetched, never
vendored, at the sha in `asimov_motor.SHA`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import asimov_motor as _motor

#: Which of their policies this module can drive. Both speak the same 375/23
#: interface and the same spec schema; the get-up is the one with a job in
#: this repo, and safe-fall is here because the pair is the whole loop.
POLICIES = ("getup", "safefall")


def onnx_path(name: str = "getup") -> Path:
    return _motor.cache_dir() / "policies" / f"{name}.onnx"


@lru_cache(maxsize=len(POLICIES))
def spec(name: str = "getup") -> dict:
    return json.loads(_motor.spec_path(name).read_text())


@lru_cache(maxsize=len(POLICIES))
def term_slices(name: str = "getup") -> tuple[tuple[str, int, int], ...]:
    """`(term, start, stop)` into OUR 78-obs, in THEIR term order.

    Derived from both publications and cross-checked, never typed: a term
    whose scale or joint order disagreed would be a silently wrong policy —
    "a wrong permutation still produces actions and still moves the robot; it
    just falls over".
    """
    from .. import contract as C
    from . import asimov_walk as W

    s = spec(name)
    obs = s["observation"]
    ours = {
        "base_ang_vel": (W.OBS_ANG_VEL, W.ANG_VEL_SCALE),
        "projected_gravity": (W.OBS_GRAVITY, 1.0),
        "joint_pos": (W.OBS_JOINT_POS, 1.0),
        "joint_vel": (W.OBS_JOINT_VEL, W.JOINT_VEL_SCALE),
        "actions": (W.OBS_LAST_ACTION, 1.0),
    }
    # The cross-checks, all of them, because each one is a way the adapter
    # could be wrong while still emitting plausible actions.
    if obs["obs_joint_order"] != [W.JOINT_NAMES[i] for i in W.slot_order()]:
        raise RuntimeError(
            "their obs_joint_order is no longer Menlo's slot order — the "
            "joint blocks would need a permutation this adapter does not do")
    if s["action"]["joint_order"] != list(W.JOINT_NAMES):
        raise RuntimeError(
            "their action joint_order is no longer JOINT_NAMES — the returned "
            "action would be applied to the wrong joints")
    if abs(float(s["action"]["scale"]) - float(W.ACTION_SCALE)) > 1e-9:
        raise RuntimeError(
            f"action scale differs: theirs {s['action']['scale']} vs ours "
            f"{W.ACTION_SCALE} — the same numbers would mean different angles")
    dflt = s["default_joint_pos"]
    off = max(abs(float(dflt[n]) - float(p))
              for n, p in zip(W.JOINT_NAMES, W.DEFAULT_POSE, strict=True))
    if off > 1e-6:
        raise RuntimeError(
            f"default pose differs by up to {off:.4f} rad — `joint_pos` is "
            "measured RELATIVE to it, so the observation would be offset")
    if abs(float(s["physics_dt"]) - float(C.PHYSICS_DT)) > 1e-12:
        raise RuntimeError(
            f"physics dt differs: theirs {s['physics_dt']} vs ours "
            f"{C.PHYSICS_DT}")

    out = []
    for t in obs["terms"]:
        name_, dim = t["name"], int(t["dim"])
        if name_ not in ours:
            raise RuntimeError(
                f"their observation term {name_!r} has no slot in Menlo's "
                "78-float layout — this policy no longer fits our contract")
        (a, b), our_scale = ours[name_]
        if b - a != dim:
            raise RuntimeError(
                f"{name_}: their dim {dim} vs our slot width {b - a}")
        if abs(float(t["scale"]) - float(our_scale)) > 1e-6:
            raise RuntimeError(
                f"{name_}: their scale {t['scale']} vs the one already baked "
                f"into our obs ({our_scale}) — the adapter would have to "
                "rescale and currently does not")
        out.append((name_, int(a), int(b)))
    width = sum(b - a for _n, a, b in out) * int(obs["history_length"])
    if width != int(obs["dim"]):
        raise RuntimeError(
            f"the terms sum to {width}, their graph takes {obs['dim']}")
    return tuple(out)


@lru_cache(maxsize=len(POLICIES))
def _session(name: str):
    """One ORT session per policy, shared. The HISTORY is per-caller and lives
    on the adapter, so the session itself carries no episode state."""
    import onnxruntime as ort

    p = onnx_path(name)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} is not fetched — uv run python -m "
            "microduck_local.robots.asimov_motor --fetch")
    o = ort.SessionOptions()
    # One thread, for the reason `viz_server._onnx_infer` gives: the lab steps
    # its whole roster serially in one frame loop.
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(p), sess_options=o,
                               providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


class OpenHorizonPolicy:
    """Their 375-input policy as a `(obs78) -> action23` callable.

    One instance PER duck: the 5-frame ring is episode state, and sharing it
    between two robots on the stage would feed each the other's frames.
    `reset()` drops it, which `Duck.reset` calls — their own runner calls
    `pol.reset()` once per episode for the same reason, and the spec is
    explicit that after a (re)start every history slot holds the first value
    rather than a zero.
    """

    #: What this callable CONSUMES, for the lab's contract check. 78, not 375:
    #: the history is built inside.
    obs_dim = 78

    def __init__(self, name: str = "getup"):
        if name not in POLICIES:
            raise ValueError(f"name must be one of {POLICIES}, got {name!r}")
        self.name = name
        self._terms = term_slices(name)
        self._hist = int(spec(name)["observation"]["history_length"])
        self._sess, self._in = _session(name)
        self._buf: dict[str, list[np.ndarray]] | None = None

    def reset(self) -> None:
        self._buf = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, np.float32)
        if o.shape[-1] != self.obs_dim:
            raise ValueError(
                f"{self.name} adapter wants Menlo's {self.obs_dim}-float "
                f"observation, got {o.shape[-1]}")
        first = self._buf is None
        if first:
            self._buf = {n: [] for n, _a, _b in self._terms}
        parts = []
        for n, a, b in self._terms:
            v = o[a:b]
            buf = self._buf[n]
            if not buf:
                # "After a (re)start, every history slot is filled with the
                # first value" — their spec. Zeros would hand the policy a
                # 5-frame velocity transient that never happened.
                buf.extend(v.copy() for _ in range(self._hist))
            else:
                buf.pop(0)
                buf.append(v.copy())
            parts.append(np.concatenate(buf))
        x = np.concatenate(parts).astype(np.float32)[None]
        return self._sess.run(None, {self._in: x})[0][0].astype(np.float32)


def lab_entries() -> tuple[dict, ...]:
    """Palette entries for the lab — `()` when the release is not fetched.

    `envKwargs` rides on the entry because the lab derives a policy's env from
    its run's `behavior.json`/`run.json`, and a FETCHED policy has neither:
    `env_kwargs_for_policy_path` returns `{}`, which `slot_env` honours by
    falling back to the body's `default_task` — so Menlo's shipped walker and
    this get-up both landed in the WALK env, standing, with nothing to get up
    from. That is the "duck-shaped question asked of a non-duck body" shape
    from `AGENTS.md`, and the fix belongs on the entry so the next fetched
    policy gets it for free.
    """
    if not _motor.spec_path().is_file() or not onnx_path().is_file():
        return ()
    return ({
        "id": "asimov1-openhorizon-getup",
        "label": "Asimov 1 get-up (OpenHorizon)",
        # `title` as well as `label`: the palette shows `label`, and the
        # ROSTER ROW shows `title` (that is what `do_assign` and `do_spawn_duck`
        # read). Without it a stage of three Asimovs labelled this one by its
        # raw policy id while the others carried their `describe-run` titles.
        "title": "Asimov get-up — OpenHorizon (the reference)",
        "group": "shipped",
        "groupTitle": "Asimov 1 (shipped)",
        "path": str(onnx_path()),
        "robot": "asimov1",
        "adapter": "openhorizon:getup",
        # THE GET-UP ENV, and the actuator it was trained and measured in.
        # `action_clip` 13.0 because their spec says `"clip": null` and their
        # own rollouts reach |a| = 11.51: at the body's inherited 4.0, 15.8 %
        # of their steps are clipped and the policy on stage would not be
        # theirs. `fell_cut` off so a robot that stands and then goes down is
        # watchable instead of respawning.
        "envKwargs": {
            "task": "getup",
            "getup_label": "prone",
            "actuator_force": "dc",
            "action_clip": 13.0,
            "fell_cut": False,
            "max_episode_s": 20.0,
            "limit_cost": 0.0,
        },
        "note": "openhorizon-labs' PPO get-up (Apache-2.0), 375 inputs = our "
                "78 minus the velocity command, 5 frames deep. Measured HERE "
                "inside Menlo's published speed-torque envelope: 24/24 rises, "
                "median 1.00 s. The thing that gets up — Menlo ship only the "
                "walker.",
    },)
