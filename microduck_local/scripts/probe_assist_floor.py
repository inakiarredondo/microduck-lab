"""Calibrate the assist on STAYING ON THE FLOOR, not on peak height.

`asimov_getup_gap` picked assist 0.85 because the peak pelvis height climbed
with it (0.266 at 0.00, 0.294 at 0.20, 0.334 at 0.50, 0.385 at 0.70) and "at
0.85 the crossing appears". Height was the wrong criterion: measured on the
trained policy, that crossing is pelvis 0.678 m at uprightness **-1.00** with
**NOTHING** on the floor, and 97 % of all frames have no foot down. The assist
is not helping it push up, it is lifting an inverted robot clear of the ground.

Nothing charged for that: the ref env HAS the term (`flight_penalty`, which
calls `_flight_charge` on `self._ground_cost`) but the gap recipe never set
`MICRODUCK_ASIMOV_GETUP_GROUND`, so it was 0.0 for all 8,400,000 steps.

So the assist is re-priced here against the DO-NOTHING policy, on the
quantity that matters: what fraction of the episode is airborne. The largest
assist that keeps the robot supported is the honest anchor.
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--assists", default="0.0,0.2,0.4,0.55,0.7,0.85,1.0")
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S

    print("  assist priced against ZERO ACTION, prologue 'pushup', "
          f"{a.episodes} eps")
    print(f"  {'assist':>7s}{'peak z':>9s}{'up at pk':>10s}{'airborne':>10s}"
          f"{'span':>8s}{'verdict':>22s}")
    for s in [float(x) for x in a.assists.split(",")]:
        env = S.get("asimov1").env_class("getup_ref")(
            domain_rand=False, obs_noise=False, action_delay=False, seed=0,
            start_phase="prone", prologue="pushup", getup_assist=s,
            history=5, actuator_force="dc", action_clip=4.0,
            fell_cut=False, max_episode_s=8.0)
        m, d = env.model, env.data
        pelvis = m.body("pelvis_link").id
        zero = np.zeros(env.action_space.shape, np.float32)
        pk, upk, air, tot, spans = [], [], 0, 0, []
        for e in range(a.episodes):
            env.reset(seed=e)
            best = (-1.0, 0.0)
            for _k in range(int(env.max_steps)):
                env.step(zero)
                z = float(d.xpos[pelvis][2])
                tot += 1
                if not env._on_floor_contacts():
                    air += 1
                if z > best[0]:
                    best = (z, float(-env._projected_gravity()[2]))
                spans.append(float(env._foot_span()))
            pk.append(best[0])
            upk.append(best[1])
        frac = air / max(1, tot)
        verdict = ("supported" if frac < 0.05 else
                   "some flight" if frac < 0.25 else "MOSTLY AIRBORNE")
        print(f"  {s:7.2f}{np.median(pk):9.3f}{np.median(upk):+10.2f}"
              f"{frac:10.1%}{np.median(spans):8.3f}{verdict:>22s}", flush=True)
    print("\n  fold is pelvis 0.511 m at uprightness +0.83")
    print("  the recipe's own ladder starts at 0.85")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
