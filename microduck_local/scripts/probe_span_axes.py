"""Is the "splits" LATERAL or FORE-AFT? They are not the same posture.

`_foot_span` is `hypot(dx, dy)` — lateral and fore-aft collapsed into one
number, deliberately ("a robot in a front-to-back splits is in the splits").
But those are different postures with different consequences. Legs out to the
SIDES is abduction: the hip is loaded sideways, it is the splits, and it is
what looks alarming and risks the joint. Feet apart FRONT-TO-BACK is a LUNGE —
a normal human stride is 0.7-0.8 m, and a 1.2 m robot with 0.85 m between its
feet fore-aft is mid-stride, not injured.

So the one number has been carrying a verdict it cannot support. This splits it
into its axes, in the body's OWN frame (not the world's, or a robot that has
turned reads its stride as abduction), across the crossing band where the width
is spent.

If most of the 0.85 m is fore-aft, "it does the splits" is the wrong
description and the posture may already be acceptable.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

DROP = ("task", "clip_file", "track_tol", "ref_speed", "free", "start_phase",
        "handoff_pay", "force_lag", "getup_label", "prologue")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", default=["asimov-getup-h5pd2"])
    ap.add_argument("--episodes", type=int, default=12)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    print(f"  foot separation decomposed in the BODY frame, across the "
          f"crossing (pelvis 0.30-0.52 m)\n")
    print(f"  {'policy':<34s}{'total':>8s}{'LATERAL':>9s}{'fore-aft':>10s}"
          f"{'lat share':>11s}")
    for run in a.runs:
        kw = dict(replay_kwargs(Path("runs") / run))
        for k in DROP:
            kw.pop(k, None)
        kw.update(getup_label="prone", fell_cut=False, max_episode_s=10.0)
        env = S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
        m, d = env.model, env.data
        pel = m.body("pelvis_link").id
        ids = tuple(int(m.geom_bodyid[env.foot_geoms[s]])
                    for s in ("left", "right"))
        pol = GA.load(run)
        lat, fwd, tot = [], [], []
        for s in range(a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            for _k in range(int(env.max_steps)):
                o, _r, t, tr, _i = env.step(np.asarray(pol(o), np.float32))
                z = float(d.xpos[pel][2])
                if not (0.30 <= z < 0.52):
                    if t or tr:
                        break
                    continue
                # THE BODY'S OWN FRAME. In the world frame a robot that has
                # turned reads its stride as abduction and vice versa, which
                # is the whole distinction this probe exists to make.
                q = np.asarray(d.xquat[pel], float)
                w, x, y, zq = q
                # body x-axis (forward) and y-axis (left) in world coords
                fx = np.array([1 - 2 * (y * y + zq * zq),
                               2 * (x * y + zq * w),
                               2 * (x * zq - y * w)])
                fy = np.array([2 * (x * y - zq * w),
                               1 - 2 * (x * x + zq * zq),
                               2 * (y * zq + x * w)])
                v = np.asarray(d.xpos[ids[0]], float) - np.asarray(
                    d.xpos[ids[1]], float)
                lat.append(abs(float(v @ fy)))
                fwd.append(abs(float(v @ fx)))
                tot.append(float(np.hypot(v[0], v[1])))
                if t or tr:
                    break
        if not tot:
            print(f"  {run[-32:]:<34s}{'(never crossed)':>38s}")
            continue
        L, F, T = np.median(lat), np.median(fwd), np.median(tot)
        print(f"  {run[-32:]:<34s}{T:8.3f}{L:9.3f}{F:10.3f}"
              f"{100.0 * L / max(T, 1e-9):10.0f} %")
    print(f"\n  a human stride is 0.7-0.8 m fore-aft and nobody calls that the "
          f"splits.")
    print(f"  LATERAL is the abduction that loads the hip sideways — that is "
          f"the number\n  a 'narrow' requirement should have been written "
          f"against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
