"""Asimov 1's REAL actuator envelope: a speed-torque curve, not a flat clamp.

A geared servo's available torque is not a constant. It peaks near stall and
falls to zero at the motor's no-load speed, so a joint that is already moving
fast has almost nothing left to give. Our envs model the actuators as MuJoCo
position servos clamped at a fixed per-group `forcerange`
(`asimov_walk.GAIN_GROUPS`, which is Menlo's URDF `<limit effort=>`) with NO
speed dependence at all -- a robot that is weak when still and unlimited when
moving. Every Asimov get-up this repo has trained was trained against that
robot, which is why its rise exceeds Menlo's published joint ratings on 82 % of
rise frames (max 5.29x) and cannot be a hardware claim.

This module is the honest version, and it is a COPY of a published model rather
than a model of our own: mjlab's `dc_motor_clip`, as `openhorizon-labs`
reproduce it in `run_policy_mujoco.py` and document in `getup_policy_spec.json`
under `motor_model`. Their get-up rises 24/24 (median 1.00 s) inside it on our
own fetched MJCF, so it is an envelope a get-up demonstrably fits through.

    tau = clip(kp*(q_target - q) - kd*qd, lo, hi)
    v   = clip(qd, -vlim*(1 + effort/peak), +vlim*(1 + effort/peak))
    hi  = min(+effort, +peak * (1 - v/vlim))
    lo  = max(-effort, -peak * (1 + v/vlim))

**Where each number comes from, because three of the four are published and one
is published by somebody else.** `kp`, `kd`, `armature` and `friction` are
already in `GAIN_GROUPS`, read off Menlo's `env.yaml` -- and they agree with
openhorizon's `pd_gains` on all 12 groups, exactly, which is two independent
publications of the same table and the cross-check `AGENTS.md` asks for.
`effort` and the rated speed `vlim` are Menlo's own URDF `<limit effort= />`
and `<limit velocity= />`, read from the file here rather than typed (and
openhorizon's `velocity_limit` matches the URDF on all 23 joints). `peak_torque`
is the one quantity NO Menlo file states: it is openhorizon's, and it is 3.000x
the URDF effort on nine of eleven groups (ankle_pitch 3.635, ankle_roll 3.388).
That is why `limits()` takes a `source` and records it, instead of presenting a
blend of two vendors as one robot.

**Why this is exactly faithful and not an approximation.** A MuJoCo position
servo with `gaintype=FIXED, gainprm=(kp,)`, `biastype=AFFINE,
biasprm=(0,-kp,-kd)` computes `kp*(ctrl - q) - kd*qd` and then clamps it to
`actuator_forcerange` -- which is `np.clip(tau, lo, hi)`, the same two lines.
So the envelope needs no torque actuators, no bypass of `mj_step` and no
`qfrc_applied`: it is `actuator_forcerange` rewritten from the live `qvel`
before each physics substep, which is what `DcMotorClip` does. Verified against
openhorizon's own `Policy.torque` to 0 N*m on random states by
`tests/test_asimov_motor.py`.

It replaces `AsimovGetupEnv._apply_vel_limit`, which was NOT this model and was
wrong in both directions: an opposing force proportional to overspeed, which
took away torque above the rating without ever granting the 3x peak below it.
Measurements made under it ("our get-up goes 24/24 -> 0/24 under limits")
over-penalised the policy and should be re-read under this module.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

#: openhorizon-labs' get-up / safe-fall release: the source of `peak_torque`,
#: of the `dc_motor_clip` reference implementation, and of the only policy
#: known to get this robot up inside its own ratings.
REPO = "openhorizon-labs/asimov1-getup-safefall"
SHA = "f657abede4c0e8c74b96d7a2546568b63fed619f"
LICENCE = "Apache-2.0"

#: Files `fetch()` brings down. The spec carries `peak_torque` and the
#: `motor_model` string; the runner is the reference implementation this module
#: is tested against; the README carries the licence, because "a licence that
#: stayed on GitHub is a licence nobody read".
FETCH_FILES = (
    "README.md",
    "run_policy_mujoco.py",
    "policies/getup.onnx",
    "policies/getup_policy_spec.json",
    "policies/safefall.onnx",
    "policies/safefall_policy_spec.json",
)

#: Which vendor's continuous torque limit to enforce. Not a style choice: see
#: the table in `limits`.
SOURCES = ("urdf", "openhorizon")

#: Picks the enforced `effort` when nothing says. "urdf" is Menlo's own number
#: for their own robot, so it is the default; `openhorizon` is what THEIR
#: policy was trained inside and is tighter on four leg groups.
SOURCE_ENV = "MICRODUCK_ASIMOV_MOTOR_SOURCE"


def cache_dir() -> Path:
    """Where `fetch()` puts the release. Beside the other fetched robots."""
    from . import asimov
    return asimov.cache_dir().parent / "openhorizon_getup"


def spec_path(name: str = "getup") -> Path:
    return cache_dir() / "policies" / f"{name}_policy_spec.json"


def require() -> Path:
    d = cache_dir()
    if not spec_path().is_file():
        raise FileNotFoundError(
            f"{REPO} is not fetched ({d} has no policies/getup_policy_spec"
            ".json). `peak_torque` is published there and nowhere in Menlo's "
            "files, so it cannot be guessed:\n    uv run python -m "
            "microduck_local.robots.asimov_motor --fetch")
    return d


def fetch(dest: Path | None = None) -> Path:
    """Download the release at `SHA` from the Hub, and print the licence."""
    from huggingface_hub import hf_hub_download
    out = dest or cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name in FETCH_FILES:
        print(f"[asimov-motor] {REPO}/{name}")
        hf_hub_download(repo_id=REPO, filename=name, revision=SHA,
                        local_dir=str(out))
    print(f"[asimov-motor] at {out} — {REPO} @ {SHA[:12]} ({LICENCE})")
    return out


# ------------------------------------------------------- the published numbers

@lru_cache(maxsize=1)
def urdf_limits() -> dict[str, tuple[float, float]]:
    """`{joint: (effort N*m, rated speed rad/s)}` from Menlo's own URDF.

    Read from the vendor's file, never typed here: a revision that re-rates a
    motor must change this repo's physics, not disagree with it silently.
    """
    from . import asimov
    path = asimov.cache_dir() / "urdf" / "asimov_1.urdf"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — `uv run fetch-robot asimov1` brings the URDF "
            "down with the MJCF, and the torque/speed ratings live only there")
    out: dict[str, tuple[float, float]] = {}
    for j in re.finditer(r'<joint[^>]*name="([^"]+)"[^>]*>(.*?)</joint>',
                         path.read_text(), re.S):
        lim = re.search(r"<limit([^>]*)/>", j.group(2))
        if not lim:
            continue
        eff = re.search(r'effort="([-\d.eE]+)"', lim.group(1))
        vel = re.search(r'velocity="([-\d.eE]+)"', lim.group(1))
        if eff and vel:
            out[j.group(1)] = (float(eff.group(1)), float(vel.group(1)))
    return out


@lru_cache(maxsize=1)
def openhorizon_gains() -> dict[str, dict[str, float]]:
    """`{joint: pd_gains group}` from openhorizon's spec, expanded per joint.

    Their `pd_gains` keys joints by regex, exactly as their `Policy.__init__`
    resolves them — `re.fullmatch` against each pattern, first group wins.
    """
    spec = json.loads(spec_path().read_text())
    names = spec["action"]["joint_order"]
    out = {}
    for n in names:
        out[n] = next(
            g for g in spec["pd_gains"].values()
            if any(re.fullmatch(p, n) for p in g["joints"]))
    return out


@dataclass(frozen=True)
class MotorLimits:
    """One array per published quantity, in `asimov_walk.JOINT_NAMES` order."""

    effort: np.ndarray    #: continuous torque limit, N*m
    peak: np.ndarray      #: torque available at zero speed, N*m
    vlim: np.ndarray      #: rated (no-load) speed, rad/s
    source: str           #: which vendor's `effort` is enforced

    def vmax(self) -> np.ndarray:
        """Where the envelope bottoms out — their `vmax`, above which a motor
        can only brake. `qd` is clipped to it before the window is computed,
        which is what keeps `lo <= hi` for every speed."""
        return self.vlim * (1.0 + self.effort / self.peak)

    def note(self) -> str:
        return (f"effort={self.source}, peak_torque={REPO}@{SHA[:12]}, "
                f"vlim=Menlo URDF")


@lru_cache(maxsize=len(SOURCES))
def limits(source: str | None = None) -> MotorLimits:
    """The envelope, in `JOINT_NAMES` order, from the vendors' own files.

    `source` chooses whose continuous `effort` is enforced, because the two
    publications disagree on four leg groups and neither is ours to average:

    | group      | Menlo URDF | openhorizon | peak |
    |------------|-----------:|------------:|-----:|
    | hip_pitch  |         45 |          40 |  120 |
    | hip_roll   |         45 |          30 |   90 |
    | hip_yaw    |         28 |          20 |   60 |
    | knee       |         45 |          25 |   75 |

    Everything else agrees to the digit. "urdf" is Menlo's rating for Menlo's
    robot and is the default; "openhorizon" is the envelope their 24/24 get-up
    was trained and measured inside, so it is the one to quote when comparing
    against them. `peak` and `vlim` are the same either way.
    """
    src = (source or os.environ.get(SOURCE_ENV, "") or "urdf").strip().lower()
    if src not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {src!r}")
    require()
    from . import asimov_walk as W

    urdf, oh = urdf_limits(), openhorizon_gains()
    missing = [n for n in W.JOINT_NAMES if n not in urdf or n not in oh]
    if missing:
        raise RuntimeError(
            f"no published rating for {missing} — a revision added a joint "
            "this envelope does not describe, and driving it with a "
            "neighbour's limits would be a guess")

    eff = np.array([(urdf[n][0] if src == "urdf" else oh[n]["effort_limit"])
                    for n in W.JOINT_NAMES], float)
    peak = np.array([oh[n]["peak_torque"] for n in W.JOINT_NAMES], float)
    vlim = np.array([urdf[n][1] for n in W.JOINT_NAMES], float)

    # Two independent publications of the rated speed. They agree today; if a
    # revision ever parts them, that is a different robot and not a rounding.
    theirs = np.array([oh[n]["velocity_limit"] for n in W.JOINT_NAMES], float)
    bad = ~np.isclose(vlim, theirs, rtol=0, atol=5e-3)
    if bad.any():
        raise RuntimeError(
            "Menlo's URDF and openhorizon's spec disagree on the rated speed "
            f"of {[W.JOINT_NAMES[i] for i in np.flatnonzero(bad)]}: "
            f"{vlim[bad].tolist()} vs {theirs[bad].tolist()} rad/s")
    if not (peak > eff).all():
        raise RuntimeError(
            "peak_torque must exceed the continuous effort for the envelope "
            "to mean anything: "
            f"{[W.JOINT_NAMES[i] for i in np.flatnonzero(peak <= eff)]}")
    return MotorLimits(eff, peak, vlim, src)


def gains_cross_check() -> dict[str, tuple[float, float]]:
    """Every joint where our `GAIN_GROUPS` and openhorizon's `pd_gains` differ
    on kp, kd, armature or friction. Empty today, and a test keeps it so —
    "mirror it, cross-check two independent publications of it"."""
    from . import asimov_walk as W

    oh, out = openhorizon_gains(), {}
    for n in W.JOINT_NAMES:
        kp, kd, _eff, arm, fric = W.gains_for(n)
        g = oh[n]
        for key, ours in (("kp", kp), ("kd", kd), ("armature", arm),
                          ("friction", fric)):
            if not np.isclose(ours, g[key], rtol=0, atol=1e-6):
                out[f"{n}.{key}"] = (ours, float(g[key]))
    return out


# ------------------------------------------------------------- the model itself

def torque_window(qd: np.ndarray, lim: MotorLimits
                  ) -> tuple[np.ndarray, np.ndarray]:
    """`(lo, hi)` N*m available at joint speeds `qd` — mjlab's dc_motor_clip.

    A pure function of the speed and the ratings, so it can be checked against
    openhorizon's `Policy.torque` without a simulator in the way.
    """
    v = np.clip(qd, -lim.vmax(), lim.vmax())
    hi = np.minimum(lim.effort, lim.peak * (1.0 - v / lim.vlim))
    lo = np.maximum(-lim.effort, -lim.peak * (1.0 + v / lim.vlim))
    return lo, hi


class DcMotorClip:
    """Rewrite `actuator_forcerange` from the live `qvel`, once per substep.

    Nothing else: the position servos already compute `kp*(ctrl - q) - kd*qd`
    and MuJoCo already clamps that to `actuator_forcerange`, so moving the
    clamp is the whole model. `before_step` is called before every
    `mj_step` — per SUBSTEP and not per control step, because that is where
    openhorizon's `_physics` recomputes it and because 5 ms of a 1.2 m robot
    falling is 40 mrad of hip travel.
    """

    def __init__(self, model, data, lim: MotorLimits | None = None,
                 joint_names=None):
        from . import asimov_walk as W

        self.model, self.data = model, data
        self.lim = lim if lim is not None else limits()
        names = list(joint_names if joint_names is not None else W.JOINT_NAMES)
        self.dofadr = np.array([model.joint(n).dofadr[0] for n in names])
        self.actid = np.array([model.actuator(n).id for n in names])
        if (model.actuator_forcelimited[self.actid] == 0).any():
            raise RuntimeError(
                "an actuator here is not force-limited, so MuJoCo would "
                "ignore the envelope entirely — walk_spec sets forcerange on "
                "every joint and the compiler's autolimits turns the flag on")
        # Contiguous out-buffers: this runs 200 times a simulated second.
        self._out = np.empty((len(names), 2), float)
        self._applied = np.zeros((len(names), 2), float)

    def before_step(self) -> None:
        lo, hi = torque_window(self.data.qvel[self.dofadr], self.lim)
        self._out[:, 0], self._out[:, 1] = lo, hi
        self.model.actuator_forcerange[self.actid] = self._out
        # Kept because `headroom` must price the torque against the window
        # that was IN FORCE when it was computed -- see the method.
        self._applied[:] = self._out

    def headroom(self) -> np.ndarray:
        """`|tau|` as a fraction of the ceiling THAT WAS IN FORCE — 1.0 means
        the motor is saturated against its speed-dependent limit rather than
        against the flat rating.

        Priced against `_applied` and NOT against a window recomputed from
        today's `qvel`, because `mj_step` integrates AFTER it computes
        actuation: once the substep loop ends, `actuator_force` is one substep
        older than `qvel`. Recomputing the window post-hoc reported a maximum
        headroom of **4.495** on a run whose true maximum is 1.000 — the
        clamp was exact on all 18,377 joint-substeps — so the naive version
        was a readout that manufactured violations out of a 5 ms frame
        mismatch. The same trap as `_refresh_derived` in `walk_env`.
        """
        lo, hi = self._applied[:, 0], self._applied[:, 1]
        tau = np.asarray(self.data.actuator_force[self.actid], float)
        span = np.where(tau >= 0.0, np.maximum(hi, 1e-9),
                        np.maximum(-lo, 1e-9))
        return np.abs(tau) / span


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true",
                    help=f"download {REPO} @ {SHA[:12]} ({LICENCE})")
    ap.add_argument("--source", default=None, choices=SOURCES)
    a = ap.parse_args(argv)
    if a.fetch:
        fetch()
        return 0
    from . import asimov_walk as W

    lim = limits(a.source)
    print(f"{REPO} @ {SHA[:12]} ({LICENCE})\n{lim.note()}\n")
    diff = gains_cross_check()
    print(f"kp/kd/armature/friction vs openhorizon: "
          f"{'IDENTICAL on all 23 joints' if not diff else diff}\n")
    print(f"{'joint':30s}{'effort':>8s}{'peak':>8s}{'vlim':>8s}"
          f"{'vmax':>8s}{'peak/eff':>10s}")
    vmax = lim.vmax()
    for i, n in enumerate(W.JOINT_NAMES):
        print(f"{n:30s}{lim.effort[i]:8.1f}{lim.peak[i]:8.1f}"
              f"{lim.vlim[i]:8.2f}{vmax[i]:8.2f}"
              f"{lim.peak[i] / lim.effort[i]:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
