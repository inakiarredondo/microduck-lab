"""HOW FAST DOES IT SWING ITS LIMBS, and when?

The safety number this repo did not have. Measured 2026-10-13 on
`teach-asimov_getup_turn-967072`: walking (the shipped walker in control)
median 3 deg/s and p99 566; RISING (the get-up in control) median 532, p99
1171, max 1753. A human elbow in a punch is roughly 500-900 deg/s, so the
get-up swings limbs at or above punch speed every time it stands — which
matters far more than the fall, since that topples away from the pusher
19/19.

Split by WHO IS DRIVING, because the two phases differ by two orders of
magnitude and an episode average hides it entirely.

    uv run python scripts/probe_limb_speed.py <run> [<run> ...] [--episodes=N]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import asimov_walk as W
from microduck_local.robots import spec as S
from microduck_local.run_record import replay_kwargs


def onnx(p):
    import onnxruntime as ort
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return lambda o: s.run(None, {n: o.reshape(1, -1).astype(np.float32)})[0][0]


def measure(run: str, n_ep: int, pose: str):
    kw = dict(replay_kwargs(Path("runs") / run))
    kw.pop("task", None)
    kw.pop("handoff_pay", None)
    kw["getup_label"] = pose
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    getup = onnx(Path("runs") / run / "policy.onnx")
    walk = onnx(".cache/asimov_1/policy/policy.onnx")
    m, d = env.model, env.data
    adr = np.array([m.joint(n).qposadr[0] for n in W.JOINT_NAMES])
    rise, hold, rose = [], [], 0
    for ep in range(n_ep):
        o, _ = env.reset(seed=ep)
        env._handoff_streak = 0
        env._handoff_prev_xy = None
        handed = False
        prev = None
        up = False
        for _ in range(int(env.max_steps)):
            if not handed and env.handoff_ready():
                handed = True
            o, _r, t, tr, _ = env.step((walk if handed else getup)(o))
            q = np.asarray(d.qpos[adr], float)
            if prev is not None:
                (hold if handed else rise).append(
                    float(np.abs(q - prev).max()) * 50)
            prev = q
            up = up or R.is_standing(env)
            if t or tr:
                break
        rose += int(up)
    return (np.degrees(np.array(rise)) if rise else np.array([0.0]),
            np.degrees(np.array(hold)) if hold else np.array([0.0]),
            rose, n_ep)


def main(argv):
    runs = [a for a in argv if not a.startswith("--")]
    n_ep = next((int(a.split("=")[1]) for a in argv
                 if a.startswith("--episodes=")), 16)
    if not runs:
        print(__doc__)
        return 2
    print(f"peak joint rate, {n_ep} episodes per pose "
          f"(a human punch is ~500-900 deg/s)\n")
    print(f"{'run':30} {'pose':7} {'rose':>6} "
          f"{'RISING med':>11}{'p99':>7}{'max':>7}  {'holding med':>12}{'max':>7}")
    for run in runs:
        for pose in ("supine", "prone"):
            a, b, rose, n = measure(run, n_ep, pose)
            print(f"{run[-28:]:30} {pose:7} {rose:3}/{n:<2} "
                  f"{np.median(a):11.0f}{np.percentile(a, 99):7.0f}"
                  f"{a.max():7.0f}  {np.median(b):12.0f}{b.max():7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
