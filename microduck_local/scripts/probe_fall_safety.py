"""WHEN IT GOES DOWN, WHERE AND HOW HARD? The safety-shaped questions.

"Stays up" is not the safety question — a 32 kg, 1.2 m robot that falls is a
hazard whatever the policy does. The questions that matter for a person
standing next to it are:

  * does it topple TOWARD the push (into whoever pushed it) or away?
  * how fast is the trunk moving when it reaches the floor?
  * does it FLAIL on the way down — large commanded joint motion that could
    strike someone — or does it go slack?
  * how far does it travel from where it was standing?

READ THE CAVEAT IN THE OUTPUT. This is CPU MuJoCo. Nothing in this repo has
run on an Asimov, contact and actuator fidelity here are not validated
against hardware, and these numbers cannot license anybody to stand next to
the real robot.

    uv run python scripts/probe_fall_safety.py <run> [episodes] [bodyweights]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import asimov_walk as W
from microduck_local.robots import spec as S
from microduck_local.run_record import replay_kwargs

PUSH_S, SETTLE_S, WATCH_S = 0.15, 1.0, 3.0
FLOOR_Z = 0.30


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
    n_ep = int(argv[1]) if len(argv) > 1 else 24
    bw = float(argv[2]) if len(argv) > 2 else 1.6
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
    adr = np.array([m.joint(n).qposadr[0] for n in W.JOINT_NAMES])

    rows = []
    for ep in range(n_ep):
        o, _ = env.reset(seed=ep)
        env._handoff_streak = 0
        env._handoff_prev_xy = None
        handed = False
        at = None
        start = None
        hit = None
        peak_v = 0.0
        peak_rate = 0.0
        prev_q = None
        for k in range(int(env.max_steps)):
            d.xfrc_applied[:] = 0.0
            if not handed and env.handoff_ready():
                handed = True
            if handed and at is None and R.is_standing(env):
                at = k + int(SETTLE_S * 50)
            if at is not None and k == at:
                start = d.xpos[TB][:2].copy()
            if at is not None and at <= k < at + int(PUSH_S * 50):
                d.xfrc_applied[TB, 1] = bw * MASS * 9.81
            o, _r, t, tr, _ = env.step((walk if handed else getup)(o))
            if at is not None and k > at:
                q = np.asarray(d.qpos[adr], float)
                if prev_q is not None and float(d.xpos[TB][2]) > FLOOR_Z:
                    peak_rate = max(peak_rate,
                                    float(np.abs(q - prev_q).max()) * 50)
                prev_q = q
                v = float(np.linalg.norm(d.cvel[TB][3:6]))
                if float(d.xpos[TB][2]) > FLOOR_Z:
                    peak_v = max(peak_v, v)
                if hit is None and float(d.xpos[TB][2]) < FLOOR_Z:
                    hit = (v, d.xpos[TB][:2].copy())
            if at is not None and k > at + int(WATCH_S * 50):
                break
            if tr:
                break
        if at is None or start is None:
            continue
        rows.append((hit, start, peak_v, peak_rate))

    fell = [r for r in rows if r[0] is not None]
    print(f"{run}: {len(rows)} shoves at {bw} bw "
          f"({bw * MASS * 9.81:.0f} N), {len(fell)} reached the floor\n")
    if fell:
        # The push is +y, so the pusher stands on the -y side. Displacement
        # with a POSITIVE y means it went away from them.
        dy = np.array([float(h[1][1] - s[1]) for h, s, _, _ in fell])
        dist = np.array([float(np.linalg.norm(h[1] - s)) for h, s, _, _ in fell])
        vhit = np.array([float(h[0]) for h, _, _, _ in fell])
        vmax = np.array([v for _, _, v, _ in fell])
        rate = np.array([r for _, _, _, r in fell])
        print(f"  toppled AWAY from the push (good): "
              f"{int((dy > 0).sum())}/{len(fell)}; "
              f"toward it: {int((dy <= 0).sum())}/{len(fell)}")
        print(f"  travel from where it stood: median {np.median(dist):.2f} m  "
              f"max {dist.max():.2f} m")
        print(f"  trunk speed reaching the floor: median {np.median(vhit):.2f} "
              f"m/s  max {vhit.max():.2f} m/s")
        print(f"  peak trunk speed while still up: median {np.median(vmax):.2f} "
              f"m/s  max {vmax.max():.2f} m/s")
        print(f"  fastest joint motion on the way down: median "
              f"{np.degrees(np.median(rate)):.0f} deg/s  max "
              f"{np.degrees(rate.max()):.0f} deg/s")
    print("\n  CAVEAT, and it outweighs every number above: this is CPU "
          "MuJoCo.\n  Nothing in this repo has run on an Asimov; the contact "
          "and actuator\n  fidelity is not validated against hardware, and a "
          "32 kg robot falling\n  from 1.2 m is a hazard whatever a simulator "
          "says. These numbers\n  compare POLICIES. They do not license "
          "anyone to stand beside one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
