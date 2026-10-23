"""CLOSED-LOOP retargeting: scale the clone's hip-roll OUTPUT and replay it.

The open-loop version of this test was void — replaying a closed-loop policy's
recorded actions diverges, and the control (scale 1.00, the unmodified
demonstration) rose 0/12 with peak trunk 0.407 m where a stand is 0.64. A
demonstration from a feedback controller is not an open-loop trajectory, and
zeroing the recorded abduction channel still left a 0.868 m span, which says
the splay emerges from the dynamics rather than from that channel.

So the policy is run IN THE LOOP and its hip-roll output is scaled before it
reaches the actuators. That keeps the feedback that makes the rise work while
removing the abduction that makes it wide — which is what retargeting a
demonstration means, as opposed to penalising width in training (measured dead)
or filtering demos (measured dead).

The control is scale 1.00 and it has to rise, or the numbers below mean nothing.
A scale that keeps the rise and halves the span IS the narrow demonstration
this track has never had.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?", default="asimov-getup-h5-dag")
    ap.add_argument("--scales", default="1.0,0.8,0.6,0.4,0.2,0.0")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--pose", default="prone")
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    JN = list(W.JOINT_NAMES)
    roll = [i for i, n in enumerate(JN) if "hip_roll" in n]
    kw = dict(replay_kwargs(Path("runs") / a.run))
    kw.pop("task", None)
    kw.update(getup_label=a.pose, fell_cut=False, max_episode_s=10.0)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    pel = env.model.body("pelvis_link").id
    pol = GA.load(a.run)

    print(f"  {a.run} from '{a.pose}', {a.episodes} eps — scaling the "
          f"hip-roll OUTPUT {[JN[i] for i in roll]}\n")
    print(f"  {'scale':>6s}{'rises':>8s}{'held 1s':>9s}{'peak span':>11s}"
          f"{'peak z':>9s}{'verdict':>26s}")
    base = None
    for sc in [float(x) for x in a.scales.split(",")]:
        rises = held = 0
        peaks, zs = [], []
        for s in range(a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            mx, pz, run, best = 0.0, 0.0, 0, 0
            for _k in range(int(env.max_steps)):
                act = np.asarray(pol(o), np.float32).copy()
                act[roll] *= sc
                o, _r, t, tr, _i = env.step(act)
                mx = max(mx, float(env._foot_span()))
                pz = max(pz, float(env.data.xpos[pel][2]))
                if R.is_standing(env):
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
                if t or tr:
                    break
            rises += int(best > 0)
            held += int(best * 0.02 >= 1.0)
            peaks.append(mx)
            zs.append(pz)
        if base is None:
            base = held
            v = "CONTROL" if held else "CONTROL FAILED — void"
        else:
            v = ("narrow AND holds" if held >= 0.7 * max(1, base)
                 and np.median(peaks) <= 0.40 else
                 "holds, still wide" if held >= 0.7 * max(1, base) else
                 "lost the rise")
        print(f"  {sc:6.2f}{f'{rises}/{a.episodes}':>8s}"
              f"{f'{held}/{a.episodes}':>9s}{np.median(peaks):11.3f}"
              f"{np.median(zs):9.3f}{v:>26s}", flush=True)
    print("\n  'held 1s' is a full second of is_standing, not a frame.")
    print("  the gate is a peak span under 0.40 m; unretargeted is ~0.93.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
