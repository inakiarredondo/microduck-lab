"""WHAT are the frames where `is_standing` fires? Held, or a flicker on one foot?

A clamp sweep reported 6/6 episodes "standing" while the best uprightness over
TWO-foot frames was +0.54 — a contradiction, because `standing_report` requires
upright > 0.85. The resolution is that `only_feet` is `floor <= feet`, which ONE
foot satisfies, so the two-foot filter was hiding the frames that passed.

Which makes the question the one this repo has been wrong about before — "a
stand is not a get-up": 23/24 scored stands were once reached by HOPPING, 1.04 s
airborne. So this reports, per episode, the longest UNBROKEN run of standing
frames, how many feet are down during them, the airborne share, and the span —
because a single frame of one-footed vertical on the way past is not a get-up
and no endpoint test can tell the difference.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?",
                    default="teach-asimov_getup_gap-3ec8cc-s2")
    ap.add_argument("--clamp", type=float, default=150.0)
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    cls = S.get("asimov1").env_class("getup_ref")
    cls.ATTITUDE_MAX = a.clamp
    kw = dict(replay_kwargs(Path("runs") / a.run))
    kw.pop("task", None)
    kw.update(fell_cut=False, max_episode_s=8.0, getup_attitude=120.0,
              fold_pay=20.0, arm_unload=2.0, ground_cost=6.0)
    env = cls(domain_rand=False, obs_noise=True, action_delay=True, seed=0,
              **kw)
    pel = env.model.body("pelvis_link").id
    pol = GA.load(a.run)

    print(f"  {a.run}, clamp {a.clamp:g} N.m, {a.episodes} eps")
    print(f"  {'ep':>3s}{'stand frames':>13s}{'longest':>9s}{'feet':>12s}"
          f"{'upright':>9s}{'trunk z':>9s}{'span':>7s}{'air %':>7s}")
    rows = []
    for s in range(a.episodes):
        o, _ = env.reset(seed=s)
        if hasattr(pol, "reset"):
            pol.reset()
        n_st, run, best_run, air, tot = 0, 0, 0, 0, 0
        feet_hist, ups, zs, spans = [], [], [], []
        for _k in range(int(env.max_steps)):
            o, _r, t, tr, _i = env.step(pol(o))
            tot += 1
            parts = [str(x) for x in R.floor_parts(env)]
            if not parts:
                air += 1
            if R.is_standing(env):
                n_st += 1
                run += 1
                best_run = max(best_run, run)
                feet_hist.append(sum(1 for x in parts if "foot" in x))
                ups.append(float(-env._projected_gravity()[2]))
                zs.append(float(env.data.xpos[pel][2]))
                spans.append(float(env._foot_span()))
            else:
                run = 0
            if t or tr:
                break
        f = (f"{min(feet_hist)}-{max(feet_hist)}" if feet_hist else "-")
        print(f"  {s:3d}{n_st:>8d}/{tot:<4d}{best_run * 0.02:8.2f}s{f:>12s}"
              f"{(np.median(ups) if ups else float('nan')):+9.2f}"
              f"{(np.median(zs) if zs else float('nan')):9.3f}"
              f"{(np.median(spans) if spans else float('nan')):7.3f}"
              f"{100.0 * air / tot:7.1f}", flush=True)
        rows.append((n_st, best_run, feet_hist, ups, spans))
    held = [r for r in rows if r[1] * 0.02 >= 0.50]
    two = [r for r in rows if r[2] and min(r[2]) >= 2]
    print(f"\n  episodes with ANY standing frame: "
          f"{sum(1 for r in rows if r[0]):d}/{len(rows)}")
    print(f"  episodes holding it >= 0.50 s:      {len(held)}/{len(rows)}")
    print(f"  episodes standing on TWO feet throughout: {len(two)}/{len(rows)}")
    allsp = [x for r in rows for x in r[4]]
    if allsp:
        print(f"  span while standing: median {np.median(allsp):.3f} m, "
              f"max {max(allsp):.3f} m   (the gate is 0.40)")
    print("\n  a single frame of one-footed vertical on the way past is not a"
          "\n  get-up — 'a stand is not a get-up', 23/24 once by hopping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
