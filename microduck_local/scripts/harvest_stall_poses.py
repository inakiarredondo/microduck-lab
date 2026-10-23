"""HARVEST THE STALL — the side-lying state a prone get-up actually dies in.

Measured 2026-10-11: from `prone`, 48 of 48 episodes roll off the face into
the side band and 0 of 48 ever gain height face-down, so phase 3m's
mechanically-blocked push-up route is not what happens. 13 carry the roll to
uprightness +1.00; **35 stall on their side at +0.17** (peak trunk 0.187 m)
and oscillate there until the clock runs out. One transition is missing:
completing the roll from the side onto the back.

Why harvest rather than spawn. `asimov_getup.collect` drops limp robots from
a uniformly random quaternion and yields only **5 side_left + 3 side_right of
500** — and the module already records that "the rare categories need targeted
spawn recipes, not more drops", because a 1.2 m humanoid does not REST on its
side; it rolls to front or back, and poses still moving are rejected. A
synthetic side pose may also sit on no feasible path, which is the mistake
that cost 7.5 M steps ("Train from the TASK's own states"). The states the
policy itself stalls in are reachable by construction.

    uv run python scripts/harvest_stall_poses.py <run> [episodes] [--write]

Without `--write` it only reports. `--write` appends them to
`fallen_poses.npz` under the label `side_stall`, atomically, preserving every
existing pose.

AND IT PRICES THE ANCHOR, because a start distribution nothing recovers from
has no gradient and no reward can give it one: it scores the ZERO-ACTION
policy and the run's own policy from the harvested states before anybody
trains on them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from microduck_local.robots import asimov_getup as G
from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import spec as S
from microduck_local.run_record import replay_kwargs

#: A stall, in the terms the route trace measured it in: off the face but not
#: onto the back, and still low.
STALL_UP = (-0.30, 0.50)
STALL_Z = 0.45
#: It has to have STAYED there — a frame passing through the side band on a
#: successful roll is not a stall. 0.6 s at 50 Hz.
STALL_HOLD = 30


def onnx(p):
    import onnxruntime as ort
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return lambda o: s.run(None, {n: o.reshape(1, -1).astype(np.float32)})[0][0]


def build_env(run: str, pose: str):
    kw = dict(replay_kwargs(Path("runs") / run))
    kw.pop("task", None)
    kw.pop("handoff_pay", None)
    kw["getup_label"] = pose
    return S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)


def harvest(run: str, n_ep: int):
    env = build_env(run, "prone")
    act = onnx(Path("runs") / run / "policy.onnx")
    walk = onnx(".cache/asimov_1/policy/policy.onnx")
    found, rose_n = [], 0
    for ep in range(n_ep):
        o, _ = env.reset(seed=ep)
        env._handoff_streak = 0
        env._handoff_prev_xy = None
        handed = False
        rose = False
        run_len = 0
        best = None
        for _ in range(int(env.max_steps)):
            if not handed and env.handoff_ready():
                handed = True
            o, _r, t, tr, _ = env.step((walk if handed else act)(o))
            up = float(-env._projected_gravity()[2])
            z = float(env.data.xpos[env.trunk_body_id][2])
            rose = rose or R.is_standing(env)
            if STALL_UP[0] < up < STALL_UP[1] and z < STALL_Z:
                run_len += 1
                if run_len >= STALL_HOLD:
                    # keep the LAST qualifying frame of the longest stall:
                    # the state it is actually stuck in, not the one it
                    # entered on.
                    best = (run_len, env.data.qpos.copy(), up, z)
            else:
                run_len = 0
        if rose:
            rose_n += 1
        elif best is not None:
            found.append(best)
    return env, found, rose_n, n_ep


def price_anchor(run: str, poses: list[np.ndarray]) -> None:
    """Score DOING NOTHING and the policy itself from these states.

    `AGENTS.md`: "Before choosing a start distribution, score the do-nothing
    policy ON IT; if nothing succeeds from those states, the curriculum has
    no anchor and no reward will give it one."
    """
    import mujoco
    env = build_env(run, "prone")
    act = onnx(Path("runs") / run / "policy.onnx")
    walk = onnx(".cache/asimov_1/policy/policy.onnx")
    zero = np.zeros(env.action_space.shape, np.float32)
    for name, pol in (("zero action", lambda _o: zero), ("this policy", act)):
        up = 0
        for q in poses:
            env.reset(seed=0)
            env.data.qpos[:] = q
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            o = env._obs() if hasattr(env, "_obs") else env._get_obs()
            handed = False
            rose = False
            for _ in range(int(env.max_steps)):
                if not handed and env.handoff_ready():
                    handed = True
                o, _r, t, tr, _ = env.step((walk if handed else pol)(o))
                rose = rose or R.is_standing(env)
                if t or tr:
                    break
            up += bool(rose)
        print(f"    {name:12} rises from {up}/{len(poses)} of the stalls")


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    run = argv[0]
    n_ep = int(argv[1]) if len(argv) > 1 and not argv[1].startswith("--") else 48
    write = "--write" in argv

    env, found, rose_n, tried = harvest(run, n_ep)
    print(f"{run}: {rose_n}/{tried} rose from prone; "
          f"{len(found)} stalls harvested from the {tried - rose_n} failures")
    if not found:
        print("no stall held long enough — nothing to harvest")
        return 1
    hold = [f[0] for f in found]
    print(f"  stall length: median {int(np.median(hold))} steps "
          f"({np.median(hold)/50:.1f} s), max {max(hold)} "
          f"({max(hold)/50:.1f} s)")
    print(f"  uprightness {np.mean([f[2] for f in found]):+.2f} "
          f"| trunk {np.mean([f[3] for f in found]):.3f} m")
    qs = [f[1] for f in found]

    # WHAT DOES THE EXISTING CLASSIFIER CALL THEM? If these are already
    # `side_left`/`side_right`, the cache simply never sampled them; if they
    # are `other`, they are a posture the cache has no name for at all.
    import collections

    import mujoco
    m, d = env.model, env.data
    names = []
    for q in qs:
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        names.append(G.label(m, d)[0])
    print(f"  the existing classifier calls them: "
          f"{dict(collections.Counter(names))}")

    print("  PRICING THE ANCHOR (a start nothing recovers from has no gradient):")
    price_anchor(run, qs[:12])

    if not write:
        print("\n(report only — pass --write to append them to the cache)")
        return 0

    path = G.cache_path()
    d0 = np.load(path, allow_pickle=True)
    qpos = np.concatenate([d0["qpos"], np.stack(qs)])
    label = np.concatenate([d0["label"],
                            np.array(["side_stall"] * len(qs))])
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:          # a HANDLE: savez appends .npz to a path
        np.savez(fh, qpos=qpos, label=label)
    os.replace(tmp, path)
    print(f"\nwrote {len(qs)} 'side_stall' poses into {path.name} "
          f"({len(d0['qpos'])} -> {len(qpos)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
