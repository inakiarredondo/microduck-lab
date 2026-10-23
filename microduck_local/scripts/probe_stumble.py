"""WHY does a shove in the 1.2-1.6 bodyweight band put the robot down?

The envelope (`probe_push_envelope.py`): the walker catches <=0.8 bw 12/12,
1.2 bw 10/12, 1.6 bw 2/12, 2.2 bw 0/12. So the stumble — the band where the
outcome is still in doubt — is 1.2-1.6, and nothing is trained for it: the
get-up learns from SETTLED fallen poses and the walker is Menlo's, untouched.

Before paying for another reward term, measure what the failure IS. Four
prone attempts were spent on a mechanism nobody had measured, and the answer
turned out to be the axis rather than the weights.

The hypothesis worth killing first is a missing PROTECTIVE STEP: a human
shoved sideways throws a foot out along the push. If the robot never moves
its feet, the gap is a behaviour it does not have. If it steps and still
goes down, the gap is capability — and this repo has already measured the
ankle saturating at its +-20 deg stops, which would cap any step.

    uv run python scripts/probe_stumble.py <run> [episodes]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import asimov_walk as W
from microduck_local.robots import spec as S
from microduck_local.run_record import replay_kwargs

GRID = (1.2, 1.4, 1.6)
PUSH_S = 0.15
WATCH_S = 1.2            # the window a protective step would land in
SETTLE_S = 1.0


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
    lf = m.body(m.geom_bodyid[env.foot_geoms["left"]]).id
    rf = m.body(m.geom_bodyid[env.foot_geoms["right"]]).id
    ANK = [i for i, n in enumerate(W.JOINT_NAMES) if "ankle" in n]
    adr = np.array([m.joint(n).qposadr[0] for n in W.JOINT_NAMES])
    rng = np.array([m.jnt_range[m.joint(n).id] for n in W.JOINT_NAMES])

    print(f"{run}: {n_ep} episodes per magnitude, lateral shove once up\n")
    print(f"{'bw':>4} {'caught':>7} {'step (m)':>19} {'ankle at its stop':>19} "
          f"{'span at impact+0.5s':>20}")
    for g in GRID:
        rows = []
        for ep in range(n_ep):
            o, _ = env.reset(seed=ep)
            env._handoff_streak = 0
            env._handoff_prev_xy = None
            handed = False
            at = None
            foot0 = None
            step = 0.0
            sat = 0
            n_watch = 0
            span = 0.0
            for k in range(int(env.max_steps)):
                d.xfrc_applied[:] = 0.0
                if not handed and env.handoff_ready():
                    handed = True
                if handed and at is None and R.is_standing(env):
                    at = k + int(SETTLE_S * 50)
                if at is not None and at <= k < at + int(PUSH_S * 50):
                    d.xfrc_applied[TB, 1] = g * MASS * 9.81
                    if foot0 is None:
                        foot0 = (d.xpos[lf][:2].copy(), d.xpos[rf][:2].copy())
                o, _r, t, tr, _ = env.step((walk if handed else getup)(o))
                if foot0 is not None and k < (at or 0) + int(WATCH_S * 50):
                    step = max(step,
                               float(np.linalg.norm(d.xpos[lf][:2] - foot0[0])),
                               float(np.linalg.norm(d.xpos[rf][:2] - foot0[1])))
                    q = d.qpos[adr]
                    sat += int(sum(1 for i in ANK
                                   if q[i] <= rng[i][0] + 0.02
                                   or q[i] >= rng[i][1] - 0.02) > 0)
                    n_watch += 1
                if at is not None and k == at + int(0.5 * 50):
                    span = env._foot_span()
                if at is not None and k > at + int(4.0 * 50):
                    break
                if tr:
                    break
            if at is None:
                continue
            rows.append((bool(R.is_standing(env)), step,
                         sat / max(1, n_watch), span))
        if not rows:
            continue
        a = np.array([[float(r[0]), r[1], r[2], r[3]] for r in rows])
        up = a[:, 0] > 0.5
        def col(i, mask):
            return np.median(a[mask, i]) if mask.any() else float("nan")
        print(f"{g:4.1f} {int(up.sum()):4}/{len(rows):<3}"
              f"  caught {col(1, up):.3f} / fell {col(1, ~up):.3f}"
              f"   caught {col(2, up)*100:3.0f}% / fell {col(2, ~up)*100:3.0f}%"
              f"      caught {col(3, up):.3f} / fell {col(3, ~up):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
