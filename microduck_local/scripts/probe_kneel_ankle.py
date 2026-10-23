"""How much ankle plantarflexion would the Asimov need to KNEEL?

`AGENTS.md` records that it cannot kneel — "in every legal configuration the
FOOT touches 72-115 mm before the shin, because a 20 deg ankle cannot point the
toe out of the way" — and that is the single mechanical reason the human get-up
route (hands-and-knees -> kneel -> one foot forward -> stand) is unavailable,
which is what forces the sprawl.

What is NOT recorded is HOW MUCH range would be enough. That is a number worth
having: it is a hardware recommendation rather than a verdict, and the Asimov
is somebody else's robot.

Method, pure geometry: fold the knee hard so the shin lies toward the floor,
then sweep `ankle_pitch` past its limit and measure the lowest point of the
FOOT against the lowest point of the SHIN. The shin reaches the floor first
exactly when (foot_low - shin_low) turns positive, and the angle where that
happens is the plantarflexion the kneel needs.

(`knee_link` IS the shin — an MJCF link is named for the joint ABOVE it, which
has produced a false "no shin geom" finding here before. The parts come from
the chain, not from the name.)
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--side", default="right")
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0,
        getup_label="prone", actuator_force="dc", fell_cut=False)
    m, d = env.model, env.data
    JN = list(W.JOINT_NAMES)
    qs = {n: int(m.joint(n).qposadr[0]) for n in JN}

    shin_b = m.body(f"{a.side}_knee_link").id        # the SHIN
    foot_b = m.body(f"{a.side}_ankle_roll_link").id  # the FOOT
    def geoms(bid):
        return [g for g in range(m.ngeom)
                if m.geom_bodyid[g] == bid and int(m.geom_contype[g])]
    shin_g, foot_g = geoms(shin_b), geoms(foot_b)

    knee = f"{a.side}_knee_joint"
    ank = f"{a.side}_ankle_pitch_joint"
    hip = f"{a.side}_hip_pitch_joint"
    kr = m.jnt_range[m.joint(knee).id]
    ar = m.jnt_range[m.joint(ank).id]
    print(f"  {a.side} leg — shipped ranges: knee {np.degrees(kr[0]):+.1f} .."
          f"{np.degrees(kr[1]):+.1f} deg, ankle_pitch "
          f"{np.degrees(ar[0]):+.1f} ..{np.degrees(ar[1]):+.1f} deg\n")

    # NO GEOMETRIC BOUND. The first version of this ranked foot against shin
    # with `geom_rbound`, whose error (up to 63 mm on a lying pose) is an order
    # of magnitude larger than the 1-11 mm margins it was reporting — so it
    # claimed a kneel was available inside the shipped range, which contradicts
    # the measured 72-115 mm. `AGENTS.md`: use MuJoCo's own contact distances,
    # because every bound is wrong somewhere. So the leg is SETTLED onto the
    # floor and the question becomes "what actually touches".
    from microduck_local.robots import asimov_getup_ref as _R

    # Fold the knee as hard as the model allows and pitch the hip so the thigh
    # is vertical — the kneel posture. Then sweep the ankle.
    knee_max = kr[1] if abs(kr[1]) > abs(kr[0]) else kr[0]
    print(f"  {'ankle deg':>10s}{'settled z':>11s}   what touches the floor")
    first_ok = None
    for deg in range(-20, 95, 5):
        d.qpos[:] = 0.0
        d.qpos[2] = 0.60
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # BOTH legs — a kneel is both knees down. Folding one leg left the
        # other straight, so its foot was the lowest point and the settle
        # landed on THAT every time; the folded shin never came near the
        # floor and the sweep reported "never", which was the probe, not the
        # robot.
        for sd in ("left", "right"):
            d.qpos[qs[f"{sd}_knee_joint"]] = float(knee_max)
            d.qpos[qs[f"{sd}_hip_pitch_joint"]] = 0.0
            d.qpos[qs[f"{sd}_ankle_pitch_joint"]] = float(np.radians(deg))
        d.qvel[:] = 0.0
        _R.settle_onto_floor(m, d, clearance=-0.001)
        mujoco.mj_forward(m, d)
        touch = set()
        for c in range(int(d.ncon)):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            for g in (g1, g2):
                b = int(m.geom_bodyid[g])
                nm = m.body(b).name if b else ""
                if "knee" in nm:
                    touch.add("SHIN")
                elif "ankle_roll" in nm:
                    touch.add("foot")
                elif b:
                    touch.add(m.body(b).name.replace("_link", ""))
        inside = ar[0] <= np.radians(deg) <= ar[1]
        ok = "SHIN" in touch and "foot" not in touch
        print(f"  {deg:10d}{float(d.qpos[2]):11.3f}   "
              f"{', '.join(sorted(touch)) or 'nothing':34s}"
              f"{'' if inside else '[outside shipped range]'}")
        if ok and first_ok is None and deg > 0:
            first_ok = deg

    print()
    if first_ok is None:
        print("  the shin NEVER reaches the floor first, at any ankle angle —")
        print("  so the ankle is not the whole story and the geometry forbids")
        print("  a kneel outright.")
    else:
        need = first_ok - np.degrees(ar[1])
        print(f"  the SHIN alone takes the floor from {first_ok} deg of "
              f"plantarflexion.")
        print(f"  the robot has {np.degrees(ar[1]):.1f} deg, so a kneel needs "
              f"about {need:.0f} deg MORE.")
        print(f"  (a human ankle plantarflexes roughly 50 deg.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
