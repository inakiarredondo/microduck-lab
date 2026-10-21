"""Assist the ROTATION, not the height.

The lift assist cannot make this move. Priced against zero action, raising it
buys pelvis height and LOSES attitude — uprightness at the peak goes -0.53,
-0.36, -0.42, -0.48, -0.78, -1.00 as the assist goes 0.00 -> 0.85 — because a
vertical force at the pelvis tips the trunk over the planted forearms instead
of rotating it up. At 0.85 the "crossing" is a handstand produced by the assist
with no policy involved (peak 0.643 m at uprightness -1.00, 19.5 % airborne).

`fold` is not a height, it is pelvis 0.511 m AND uprightness +0.83. So the
help the move actually needs is a PITCH TORQUE about the trunk, and this asks
whether one exists that produces an upright crossing where the lift produces an
inversion. Applied as `xfrc_applied[trunk, 4]` (torque about y) with the lift
assist at zero, under zero action, so whatever happens is the ASSIST's doing
and not a policy's.

Both signs are swept because the sign that pitches this trunk up is a fact
about the model's frame, not something to assume — and getting it backwards
would read as "rotation does not help".
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--torques", default="0,25,50,75,100,150,220")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--lift", type=float, default=0.0,
                    help="lift assist to run ALONGSIDE the torque (0 isolates "
                         "the torque; 0.55 is the largest lift that stayed "
                         "supported)")
    a = ap.parse_args(argv)

    from microduck_local.robots import spec as S

    print(f"  pitch-torque assist, lift {a.lift:g}, ZERO ACTION, "
          f"{a.episodes} eps")
    print(f"  {'N.m':>7s}{'peak z':>9s}{'up at pk':>10s}{'best up':>9s}"
          f"{'z at best up':>13s}{'airborne':>10s}{'span':>8s}")
    for T in [float(x) for x in a.torques.split(",")]:
        for sign in ((1.0, -1.0) if T else (1.0,)):
            env = S.get("asimov1").env_class("getup_ref")(
                domain_rand=False, obs_noise=False, action_delay=False, seed=0,
                start_phase="prone", prologue="pushup", getup_assist=a.lift,
                history=5, actuator_force="dc", action_clip=4.0,
                fell_cut=False, max_episode_s=8.0)
            m, d = env.model, env.data
            pelvis = m.body("pelvis_link").id
            tb = env.trunk_body_id
            zero = np.zeros(env.action_space.shape, np.float32)
            pk, upk, bu, zbu, air, tot, spans = [], [], [], [], 0, 0, []
            for e in range(a.episodes):
                env.reset(seed=e)
                hz, hu = (-1.0, 0.0), (-9.0, 0.0)
                for _k in range(int(env.max_steps)):
                    # Set AFTER reset and every step: the env writes
                    # xfrc_applied[trunk, 2] itself each step (the lift) and
                    # leaves the torque slot alone, so this persists.
                    d.xfrc_applied[tb, 4] = sign * T
                    env.step(zero)
                    z = float(d.xpos[pelvis][2])
                    u = float(-env._projected_gravity()[2])
                    tot += 1
                    if not env._on_floor_contacts():
                        air += 1
                    if z > hz[0]:
                        hz = (z, u)
                    if u > hu[0]:
                        hu = (u, z)
                    spans.append(float(env._foot_span()))
                pk.append(hz[0]); upk.append(hz[1])
                bu.append(hu[0]); zbu.append(hu[1])
            lbl = f"{sign * T:+.0f}" if T else "0"
            print(f"  {lbl:>7s}{np.median(pk):9.3f}{np.median(upk):+10.2f}"
                  f"{np.median(bu):+9.2f}{np.median(zbu):13.3f}"
                  f"{air / max(1, tot):10.1%}{np.median(spans):8.3f}",
                  flush=True)
    print("\n  TARGET: `fold` is pelvis 0.511 m at uprightness +0.83, "
          "span 0.215 m")
    print("  the lift assist's best attitude at any level was -0.36, and its")
    print("  'crossing' at 0.85 was -1.00. 'best up' and 'z at best up' are")
    print("  ONE instant, not two maxima read together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
