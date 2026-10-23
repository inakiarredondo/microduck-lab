"""What SPAN does OpenHorizon's own policy use to cross the band?

Their policy rises 24/24 in this MuJoCo under the real envelope. The narrow
track has been treating the crossing as something no controller does narrowly —
but that was inferred from the TEACHER'S DEMONSTRATION DATA (0.917-0.944 m),
and a teacher's bound is not the robot's. Their POLICY's own trajectory through
pelvis 0.30 -> 0.511 m has never been measured here.

If it crosses narrow, a narrow demonstration of the one missing transition
already exists and can be distilled — which is the thing this track has been
missing all day. If it sprawls, that is the strongest evidence yet that the
crossing cannot be done narrowly on this body.

Reports the span BINNED BY HEIGHT, because a peak over the whole episode says
nothing about where the width is spent: the sprawl could all be below the band
(where it does not matter) or inside it (where it does).
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--pose", default="prone")
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

    BINS = [(0.00, 0.20), (0.20, 0.30), (0.30, 0.38), (0.38, 0.46),
            (0.46, 0.52), (0.52, 0.65)]
    per_bin = {b: [] for b in BINS}
    rises = 0
    peaks = []
    for s in range(a.episodes):
        o, _ = env.reset(seed=s)
        if hasattr(pol, "reset"):
            pol.reset()
        mx, up = 0.0, False
        for _k in range(int(env.max_steps)):
            o, _r, t, tr, _i = env.step(np.asarray(pol(o), np.float32))
            z = float(d.xpos[pel][2])
            sp = float(env._foot_span())
            mx = max(mx, sp)
            for b in BINS:
                if b[0] <= z < b[1]:
                    per_bin[b].append(sp)
                    break
            if R.is_standing(env):
                up = True
            if t or tr:
                break
        rises += int(up)
        peaks.append(mx)

    print(f"  OpenHorizon's policy from '{a.pose}', {a.episodes} eps: "
          f"stands {rises}/{a.episodes}, median PEAK span "
          f"{np.median(peaks):.3f} m\n")
    print(f"  {'pelvis band':>16s}{'frames':>8s}{'median span':>13s}"
          f"{'p90':>8s}{'max':>8s}")
    for b in BINS:
        v = per_bin[b]
        if not v:
            print(f"  {f'{b[0]:.2f}-{b[1]:.2f}':>16s}{0:>8d}"
                  f"{'-':>13s}{'-':>8s}{'-':>8s}")
            continue
        tag = f"{b[0]:.2f}-{b[1]:.2f}"
        print(f"  {tag:>16s}{len(v):>8d}{np.median(v):13.3f}"
              f"{np.percentile(v, 90):8.3f}{max(v):8.3f}")
    band = [x for b in BINS if b[0] >= 0.30 and b[1] <= 0.52
            for x in per_bin[b]]
    if band:
        print(f"\n  INSIDE THE BAND (0.30-0.52 m): {len(band)} frames, median "
              f"span {np.median(band):.3f} m, p90 "
              f"{np.percentile(band, 90):.3f}, max {max(band):.3f}")
        print(f"  the acceptance gate is 0.40 m; `fold` is 0.215 m")
    else:
        print("\n  it never entered the band — nothing to say about its width "
              "there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
