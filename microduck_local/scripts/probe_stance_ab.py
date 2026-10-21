"""WHO holds the stand better under a shove: Menlo's walker, or ours?

The stumble happens while the WALKER is in control, because the get-up hands
over the moment it is up. The idea on the table is to train an IDLE/STANCE
mode of our own instead — so the first thing to know is the starting point:
does our get-up policy, kept in control, resist a push better or worse than
the shipped walker?

It has never been trained to stand still (the `home` term pays for REACHING
the idle pose, not for holding it against a force), and the walker is trained
on Menlo's GPU stack — so a deficit here is expected. The size of it is what
decides whether training a stance mode starts from a reasonable place or from
behind.

    uv run python scripts/probe_stance_ab.py <run> [episodes]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import spec as S
from microduck_local.run_record import replay_kwargs

GRID = (0.4, 0.8, 1.2, 1.6)
#: "went to the floor", as distinct from staggering — half standing height.
FLOOR_Z = 0.30
PUSH_S, SETTLE_S, WATCH_S = 0.15, 1.0, 4.0


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
    kw["fell_cut"] = False
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    getup = onnx(Path("runs") / run / "policy.onnx")
    walk = onnx(".cache/asimov_1/policy/policy.onnx")
    m, d = env.model, env.data
    TB = env.trunk_body_id
    MASS = float(m.body_subtreemass[TB])

    def trial(g, keep_getup):
        """Returns (never hit the floor, up at the end, n).

        BOTH, because "up at the end" conflates resisting with recovering:
        our get-up policy starts rising the moment it is floored, so a 4 s
        window can score a fall-then-rise identically to a clean catch. That
        confound is how a raw count once read 13-15 "hand-backs" that were
        episode resets.
        """
        up = 0
        clean = 0
        n = 0
        for ep in range(n_ep):
            o, _ = env.reset(seed=ep)
            env._handoff_streak = 0
            env._handoff_prev_xy = None
            # Always RISE on the get-up; only who holds the stand differs.
            handed = False
            at = None
            went_down = False
            for k in range(int(env.max_steps)):
                d.xfrc_applied[:] = 0.0
                if not handed and env.handoff_ready():
                    handed = True
                if handed and at is None and R.is_standing(env):
                    at = k + int(SETTLE_S * 50)
                if at is not None and at <= k < at + int(PUSH_S * 50):
                    d.xfrc_applied[TB, 1] = g * MASS * 9.81
                hold = (not keep_getup) and handed
                o, _r, t, tr, _ = env.step((walk if hold else getup)(o))
                if at is not None and k > at + int(PUSH_S * 50):
                    # ON THE FLOOR, not merely "not standing". `is_standing`
                    # wants the trunk above 0.86 x standing, so a recovery
                    # STAGGER trips it: the walker scored 2/12 "never down"
                    # at 0.8 bw while catching 12/12. Half standing height is
                    # unambiguous — nothing staggers that low and comes back
                    # without a get-up.
                    went_down = went_down or float(d.xpos[TB][2]) < FLOOR_Z
                if at is not None and k > at + int(WATCH_S * 50):
                    break
                if tr:
                    break
            if at is None:
                continue              # never got up; not this comparison
            n += 1
            up += int(R.is_standing(env))
            clean += int(not went_down)
        return clean, up, n

    print(f"{run}: {n_ep} episodes per cell, lateral shove 1 s after standing\n")
    print(f"{'':16}{'MENLO WALKER':>22}{'OUR GET-UP HOLDING':>26}")
    print(f"{'bw':>5} {'newtons':>9} {'stayed up':>12}{'up at end':>10}"
          f"{'stayed up':>14}{'up at end':>12}")
    for g in GRID:
        a = trial(g, False)
        b = trial(g, True)
        print(f"{g:5.1f} {g * MASS * 9.81:9.0f} {a[0]:9}/{a[2]:<3}"
              f"{a[1]:7}/{a[2]:<3}{b[0]:11}/{b[2]:<3}{b[1]:9}/{b[2]:<3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
