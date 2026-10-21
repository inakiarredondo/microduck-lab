"""Are there NARROW, STABLE poses between the floor and `fold`?

The geometric argument against a narrow get-up was derived from the AUTHORED
route: along it the CoM sits 0.50-0.60 m ahead of the feet against a 0.540 m
leg, so the support cannot be stepped under the mass and widening is all that
is left. But that is one route's bound, not the robot's — "a teacher's bound is
not the robot's" is an error this track has already made once.

The CoM is a function of CONFIGURATION: pulling the arms and torso back moves
it toward the feet. So search configuration space directly for poses that are
NARROW (span <= --span), still up ON BOTH FEET after a real settle, and sitting
at intermediate trunk heights. If a chain of them exists from near the floor up
to `fold`, there is a narrow static ladder that nobody has climbed. If none
exists below `fold`, the geometric argument stands on its own evidence rather
than on one route's.

Settled for `--settle-s` and judged on the pose that SURVIVES, because a 0.05 s
filter and a peak-valued reading are exactly what hid the pike problem.
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--span", type=float, default=0.30)
    ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local import contract as C
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="fold_tight",
        actuator_force="dc")
    m, d = env.model, env.data
    ADR = np.array([int(m.joint(n).qposadr[0]) for n in W.JOINT_NAMES])
    JID = [m.joint(n).id for n in W.JOINT_NAMES]
    lo_j, hi_j = m.jnt_range[JID, 0], m.jnt_range[JID, 1]
    fb = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
    decim = int(C.DECIMATION)

    # Seeded from the rungs AND from the stand, so the search explores around
    # every pose this robot is known to adopt rather than uniformly over a
    # 23-dim box (where almost nothing stands).
    seeds = [np.asarray(R.BY_NAME[n].joints, float)
             for n in ("quadruped", "pike", "rock_back", "fold", "crouch")]
    seeds.append(np.asarray(env.robot.default_pose, float))
    pitches = [R.BY_NAME[n].root_pitch
               for n in ("quadruped", "pike", "rock_back", "fold", "crouch")]
    pitches.append(0.0)

    rng = np.random.default_rng(a.seed)
    n_settle = max(1, int(a.settle_s / 0.02)) * decim
    keep = []
    for i in range(a.samples):
        j = rng.integers(len(seeds))
        q = np.clip(seeds[j] + rng.normal(0, rng.uniform(0.1, 0.6), 23),
                    lo_j, hi_j)
        pitch = pitches[j] + rng.normal(0, 0.35)
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        h = 0.5 * pitch
        d.qpos[3:7] = [np.cos(h), 0.0, np.sin(h), 0.0]
        d.qpos[ADR] = q
        lo, hi = -0.3, 1.2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            d.qpos[2] = mid
            mujoco.mj_forward(m, d)
            if d.ncon > 0:
                lo = mid
            else:
                hi = mid
        d.qpos[2] = lo
        mujoco.mj_forward(m, d)
        if float(env._foot_span()) > a.span:
            continue
        for _ in range(n_settle):
            mujoco.mj_step(m, d)
        span = float(env._foot_span())
        if span > a.span:
            continue
        parts = [str(p) for p in R.floor_parts(env)]
        if sum(1 for p in parts if "foot" in p) < 2:
            continue
        if sum(1 for p in parts if "foot" not in p) > 0:
            continue                      # feet ONLY: a real stance
        z = float(d.xpos[env.trunk_body_id][2])
        com = float(np.sum(m.body_mass[:, None] * d.xipos, axis=0)[0]
                    / m.body_mass.sum())
        gap = com - float(np.mean([d.xpos[b][0] for b in fb]))
        keep.append((z, span, gap, float(-env._projected_gravity()[2])))

    print(f"{len(keep)} of {a.samples} candidates are NARROW (<= {a.span:g} m), "
          f"on BOTH FEET ONLY, still up after {a.settle_s:g} s\n")
    if not keep:
        print("  none — no narrow feet-only stance exists at any height here")
        return 1
    k = np.asarray(keep)
    print(f"  {'trunk z band':18s}{'n':>5s}{'median span':>13s}"
          f"{'median CoM gap':>16s}{'median upright':>16s}")
    for lo_z, hi_z in ((0.0,0.15),(0.15,0.25),(0.25,0.35),(0.35,0.45),
                       (0.45,0.55),(0.55,0.70)):
        msk = (k[:,0] >= lo_z) & (k[:,0] < hi_z)
        if not msk.any():
            print(f"  {lo_z:.2f} - {hi_z:.2f}{'':8s}{0:5d}{'—':>13s}{'—':>16s}{'—':>16s}")
            continue
        print(f"  {lo_z:.2f} - {hi_z:.2f}{'':8s}{int(msk.sum()):5d}"
              f"{np.median(k[msk,1]):13.3f}{np.median(k[msk,2]):+16.3f}"
              f"{np.median(k[msk,3]):+16.2f}")
    print(f"\n  `fold` sits at trunk 0.541, span 0.230, CoM gap -0.004")
    print(f"  a band with members BELOW 0.45 would be a narrow rung under fold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
