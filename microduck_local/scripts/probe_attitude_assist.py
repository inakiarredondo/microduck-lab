"""Does an ATTITUDE-holding assist give a supported UPRIGHT crossing?

The lift assist raises the trunk and inverts it (uprightness -1.00 at 0.85, 97 %
of frames airborne, and zero action reproduces it). A constant pitch torque
rotates it upright but only at pelvis 0.111 m, and lift plus constant torque
launches the robot. `getup_attitude` is a PD on uprightness instead: it fades as
the trunk arrives, damps the rate, and is clamped, so the lift can raise while
it keeps the robot from going over the top.

Everything here is ZERO ACTION, so whatever happens is the SCAFFOLD's doing and
not a policy's — which is the check the original ladder never had. The torque
actually written is reported alongside, because a knob that reads back correctly
and reaches nothing is this repo's most repeated bug.

The bar: pelvis near `fold`'s 0.511 m AND uprightness near +0.83 AND nothing
airborne AND a span at or under 0.215 m, all at ONE instant.
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gains", default="0,20,40,80,120,200")
    ap.add_argument("--lifts", default="0.0,0.4,0.55,0.7")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--clamp", type=float, default=None,
                    help="override ATTITUDE_MAX — it BINDS at the default 60 "
                         "N.m for every gain tried, so the gain sweep alone "
                         "cannot tell authority from stiffness apart")
    a = ap.parse_args(argv)

    from microduck_local.robots import spec as S

    print(f"  ZERO ACTION, prologue 'pushup', {a.episodes} eps. "
          f"TARGET: pelvis 0.511 m at upright +0.83, span <= 0.215, 0 % air")
    print(f"  {'lift':>5s}{'gain':>6s}{'|tau|':>10s}{'max z':>8s}"
          f"{'--z':>8s}{'up':>8s}{'span':>7s}{'lift':>7s}{'air':>7s}"
          f"{'JOINT':>8s}")
    print("        (the four columns after 'max z' are ONE INSTANT: the "
          "joint-max frame)")
    best_overall = None
    for lift in [float(x) for x in a.lifts.split(",")]:
        for g in [float(x) for x in a.gains.split(",")]:
            cls = S.get("asimov1").env_class("getup_ref")
            if a.clamp is not None:
                # Set on the CLASS, before construction, so it is read the
                # same way a shipped default would be.
                cls.ATTITUDE_MAX = a.clamp
            env = cls(
                domain_rand=False, obs_noise=False, action_delay=False, seed=0,
                start_phase="prone", prologue="pushup", getup_assist=lift,
                getup_attitude=g, history=5, actuator_force="dc",
                action_clip=4.0, fell_cut=False, max_episode_s=8.0)
            m, d = env.model, env.data
            pelvis = m.body("pelvis_link").id
            zero = np.zeros(env.action_space.shape, np.float32)
            hz, tmax, air, tot = [], 0.0, 0, 0
            # Report the state AT the joint maximum — ONE instant. Reporting
            # max(z) beside max(upright) is how three stages of progress were
            # invented here before; and the joint score itself is only honest
            # for a trajectory that is not launched, so the airborne fraction
            # is printed beside it rather than folded in.
            shots = []
            for e in range(a.episodes):
                env.reset(seed=e)
                shot = (-9.0, 0.0, 0.0, 0.0, 0.0)
                for _k in range(int(env.max_steps)):
                    env.step(zero)
                    tmax = max(tmax, abs(float(
                        d.xfrc_applied[env.trunk_body_id, 4])))
                    z = float(d.xpos[pelvis][2])
                    u = float(-env._projected_gravity()[2])
                    sp = float(env._foot_span())
                    sup = bool(env._on_floor_contacts())
                    tot += 1
                    if not sup:
                        air += 1
                    j = (min(z / 0.511, 1.0) * max(0.0, (u + 1.0) / 1.83)
                         * (1.0 if sup else 0.0)
                         * (1.0 if sp <= 0.30 else 0.0))
                    if j > shot[0]:
                        shot = (j, z, u, sp, float(abs(d.qvel[2])))
                    hz.append(z)
                shots.append(shot)
            frac = air / max(1, tot)
            arr = np.array(shots)
            jm = float(np.median(arr[:, 0]))
            k = int(np.argmax(arr[:, 0]))
            print(f"  {lift:5.2f}{g:6.0f}{tmax:10.1f}{max(hz):8.3f}"
                  f"{arr[k, 1]:8.3f}{arr[k, 2]:+8.2f}{arr[k, 3]:7.3f}"
                  f"{arr[k, 4]:7.2f}{frac:7.1%}{jm:8.3f}", flush=True)
            if best_overall is None or jm > best_overall[0]:
                best_overall = (jm, lift, g)
    print(f"\n  best JOINT score {best_overall[0]:.3f} at lift "
          f"{best_overall[1]:g}, gain {best_overall[2]:g}")
    print("  JOINT = (z/0.511) x (upright+1)/1.83, zeroed unless SUPPORTED and")
    print("  span <= 0.30 — one instant. 1.000 would be fold, supported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
