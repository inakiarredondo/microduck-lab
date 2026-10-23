"""Is there a NARROW TAIL in OpenHorizon's crossings — one that still rises?

Their policy's span, binned by pelvis height, falls monotonically as it climbs
(0.934 at 0.30-0.38, 0.850 at 0.38-0.46, 0.719 at 0.46-0.52, 0.239 standing)
and explodes on ENTRY (0.505 below 0.20 -> 0.881 at 0.20-0.30). Those are
medians over episodes. If the per-episode spread is wide enough that some
crossings are much narrower AND still rise, the narrow tail is a demonstration
of the one transition this track is missing — and distillation is the tool that
has worked on this body before.

So this reports, per episode, the MAX span while inside the band and whether it
stood, then the narrowest successful crossing. One episode at 0.60 m that rises
is worth more than any reward term, because it is an existence proof on the
real envelope.
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--pose", default="prone")
    ap.add_argument("--lo", type=float, default=0.30)
    ap.add_argument("--hi", type=float, default=0.52)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_openhorizon as OH
    from microduck_local.robots import spec as S

    pol = OH.OpenHorizonPolicy()
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0,
        getup_label=a.pose, actuator_force="dc", action_clip=4.0,
        history=1, fell_cut=False, max_episode_s=10.0)
    m, d = env.model, env.data
    pel = m.body("pelvis_link").id

    rows = []
    for s in range(a.episodes):
        o, _ = env.reset(seed=s)
        if hasattr(pol, "reset"):
            pol.reset()
        band_max, up, peak = 0.0, False, 0.0
        for _k in range(int(env.max_steps)):
            o, _r, t, tr, _i = env.step(np.asarray(pol(o), np.float32))
            z = float(d.xpos[pel][2])
            sp = float(env._foot_span())
            peak = max(peak, z)
            if a.lo <= z < a.hi:
                band_max = max(band_max, sp)
            if R.is_standing(env):
                up = True
            if t or tr:
                break
        rows.append((s, band_max, up, peak))

    ok = [r for r in rows if r[2] and r[1] > 0]
    print(f"  OpenHorizon from '{a.pose}', {a.episodes} eps, band "
          f"{a.lo:.2f}-{a.hi:.2f} m: {sum(1 for r in rows if r[2])} stood\n")
    print(f"  {'ep':>4s}{'band max span':>15s}{'peak z':>9s}{'stood':>7s}")
    for s, bm, u, pk in sorted(rows, key=lambda r: r[1]):
        print(f"  {s:4d}{(bm if bm else float('nan')):15.3f}{pk:9.3f}"
              f"{str(u):>7s}")
    if ok:
        w = np.array([r[1] for r in ok])
        print(f"\n  successful crossings: n={len(w)}, narrowest "
              f"**{w.min():.3f} m**, p10 {np.percentile(w, 10):.3f}, "
              f"median {np.median(w):.3f}, max {w.max():.3f}")
        print(f"  the gate is 0.40 m. under it: "
              f"{int((w <= 0.40).sum())}/{len(w)}")
        if (w <= 0.40).any():
            print("  >> A NARROW SUCCESSFUL CROSSING EXISTS — that is the "
                  "demonstration.")
        else:
            print("  >> none under the gate: the narrow tail does not reach "
                  "it, so there is\n     no narrow demonstration to harvest "
                  "from this policy.")
    else:
        print("\n  no successful crossing recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
