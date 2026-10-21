"""Which STARTING POSTURES can the working get-up rise from — and how?

Every measurement on this track used `prone`. But the pose cache holds supine,
both sides, sitting and kneeling, and a rise from any of those is necessarily a
DIFFERENT route: from supine the robot must roll or sit up before it can push,
and from a side it must get square first. If one of those works, there is
already more than one way up and the "only one move" framing is wrong.

Reports, per posture: whether it rises, and the span DECOMPOSED (lateral vs
fore-aft) through the crossing — because a route that is narrower LATERALLY is
the thing being looked for, and the combined number cannot show it.
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
    ap.add_argument("--run", default="asimov-getup-h5pd2")
    ap.add_argument("--poses",
                    default="prone,supine,side_left,side_right,sitting,"
                            "kneeling,fold_tight")
    ap.add_argument("--episodes", type=int, default=10)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    pol = GA.load(a.run)
    base = dict(replay_kwargs(Path("runs") / a.run))
    for k in DROP:
        base.pop(k, None)

    print(f"  {a.run} — which postures can it rise from? "
          f"{a.episodes} eps each\n")
    print(f"  {'start pose':<14s}{'rises':>8s}{'total':>8s}{'LATERAL':>9s}"
          f"{'fore-aft':>10s}{'end span':>10s}")
    for pose in a.poses.split(","):
        kw = dict(base)
        kw.update(getup_label=pose, fell_cut=False, max_episode_s=10.0)
        try:
            env = S.get("asimov1").env_class("getup")(
                domain_rand=False, obs_noise=True, action_delay=True,
                seed=0, **kw)
        except Exception as e:
            print(f"  {pose:<14s}{'(no cache)':>8s}  {type(e).__name__}")
            continue
        m, d = env.model, env.data
        pel = m.body("pelvis_link").id
        ids = tuple(int(m.geom_bodyid[env.foot_geoms[s]])
                    for s in ("left", "right"))
        rises = 0
        lat, fwd, tot, ends = [], [], [], []
        for s in range(a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            up = False
            for _k in range(int(env.max_steps)):
                o, _r, t, tr, _i = env.step(np.asarray(pol(o), np.float32))
                z = float(d.xpos[pel][2])
                q = np.asarray(d.xquat[pel], float)
                w, x, y, zq = q
                fx = np.array([1 - 2 * (y * y + zq * zq),
                               2 * (x * y + zq * w), 2 * (x * zq - y * w)])
                fy = np.array([2 * (x * y - zq * w),
                               1 - 2 * (x * x + zq * zq),
                               2 * (y * zq + x * w)])
                v = np.asarray(d.xpos[ids[0]], float) - np.asarray(
                    d.xpos[ids[1]], float)
                if 0.30 <= z < 0.52:
                    lat.append(abs(float(v @ fy)))
                    fwd.append(abs(float(v @ fx)))
                    tot.append(float(np.hypot(v[0], v[1])))
                if R.is_standing(env):
                    up = True
                    ends.append(float(np.hypot(v[0], v[1])))
                if t or tr:
                    break
            rises += int(up)
        f = lambda v: f"{np.median(v):.3f}" if v else "-"
        print(f"  {pose:<14s}{f'{rises}/{a.episodes}':>8s}{f(tot):>8s}"
              f"{f(lat):>9s}{f(fwd):>10s}{f(ends):>10s}")
    print(f"\n  a rise from supine or a side is a DIFFERENT ROUTE — it has to "
          f"roll or sit\n  up first. LATERAL is the axis 'narrow' should be "
          f"judged on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
