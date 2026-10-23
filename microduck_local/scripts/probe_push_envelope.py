"""HOW HARD A PUSH DOES THE WALKER SURVIVE, AND WHAT FLOORS IT?

The acceptance gate shoves at 0.20 bodyweight for 0.15 s and the handoff
survives it 180 of 181 times, so nothing in this repo has measured where the
recovery ENDS. Two things need that number: whether a dedicated push-recovery
mode is worth having at all (if the robot goes straight from "caught" to "on
the floor" there is no band for one to live in), and a magnitude that
reliably floors the robot, without which a get-up-after-a-push loop cannot
be tested.

Each episode: rise from supine, hand off to Menlo's walker, let it settle,
then shove the trunk and watch what happens.

    uv run python scripts/probe_push_envelope.py <run> [episodes]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from microduck_local.robots import asimov_getup as G
from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import spec as S
from microduck_local.run_record import replay_kwargs

#: Bodyweights of lateral trunk force, held for `PUSH_S`.
GRID = (0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.2, 3.0)
PUSH_S = 0.15
SETTLE_S = 1.0          # after the handoff, before shoving
WATCH_S = 4.0           # after the shove


def onnx(p):
    import onnxruntime as ort
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return lambda o: s.run(None, {n: o.reshape(1, -1).astype(np.float32)})[0][0]


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    run = argv[0]
    n_ep = int(argv[1]) if len(argv) > 1 else 12
    kw = dict(replay_kwargs(Path("runs") / run))
    kw.pop("task", None)
    kw.pop("handoff_pay", None)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    getup = onnx(Path("runs") / run / "policy.onnx")
    walk = onnx(".cache/asimov_1/policy/policy.onnx")
    m, d = env.model, env.data
    TB = env.trunk_body_id
    MASS = float(m.body_subtreemass[TB])

    print(f"{run}: {n_ep} episodes per magnitude, "
          f"{PUSH_S}s lateral trunk shove once standing\n")
    print(f"{'bodyweights':>12} {'newtons':>8} {'still up':>9} "
          f"{'on the floor':>13} {'in between':>11}")
    for g in GRID:
        up = floored = between = 0
        for ep in range(n_ep):
            o, _ = env.reset(seed=ep)
            env._handoff_streak = 0
            env._handoff_prev_xy = None
            handed = False
            stood_at = None
            for k in range(int(env.max_steps)):
                d.xfrc_applied[:] = 0.0
                if not handed and env.handoff_ready():
                    handed = True
                if handed and stood_at is None and R.is_standing(env):
                    stood_at = k + int(SETTLE_S * 50)
                if (stood_at is not None
                        and stood_at <= k < stood_at + int(PUSH_S * 50)):
                    d.xfrc_applied[TB, 1] = g * MASS * 9.81
                o, _r, t, tr, _ = env.step((walk if handed else getup)(o))
                if (stood_at is not None
                        and k > stood_at + int(WATCH_S * 50)):
                    break
                if t or tr:
                    break
            if stood_at is None:
                continue                      # never got up; not this test
            lab = G.label(m, d)[0]
            if R.is_standing(env):
                up += 1
            elif lab in ("prone", "supine", "side_left", "side_right"):
                floored += 1
            else:
                between += 1
        print(f"{g:12.1f} {g * MASS * 9.81:8.0f} {up:9} {floored:13} "
              f"{between:11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
