"""Play the authored get-up reference OPEN LOOP and measure what it does.

Not a success criterion — a dynamic motion tracked by nothing but position
servos has no business working, and `rock_back` is explicitly a span where
nothing holds. It is a SANITY check on the authoring: a reference the robot
cannot even begin to follow is geometrically wrong, and that is cheap to find
out here rather than after 600 k steps of training.

    uv run python scripts/probe_asimov_getup_ref.py
    uv run python scripts/probe_asimov_getup_ref.py --verify   # pin the table
"""

from __future__ import annotations

import argparse

import numpy as np

from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import asimov_walk as W


def _env():
    from microduck_local.robots.asimov_env import AsimovGetupEnv
    return AsimovGetupEnv(domain_rand=False, obs_noise=False,
                          getup_blend=0.0, getup_assist=0.0)


def place(env, q, root_pitch, clearance=0.002):
    """Put the robot in `q` at `root_pitch`, dropped onto the floor."""
    import mujoco
    m, d = env.model, env.data
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.qpos[0:3] = [0.0, 0.0, 1.0]
    half = float(root_pitch) / 2.0
    d.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]
    for n, v in zip(W.JOINT_NAMES, q):
        d.qpos[m.joint(n).qposadr[0]] = v
    mujoco.mj_forward(m, d)
    gs = [g for g in range(m.ngeom)
          if int(m.geom_contype[g]) != 0 and m.geom_bodyid[g] != 0]
    low = min(float(d.geom_xpos[g][2] - m.geom_rbound[g]) for g in gs)
    d.qpos[2] -= low - clearance
    mujoco.mj_forward(m, d)


def _action(q, clip):
    a = (np.asarray(q) - np.asarray(W.DEFAULT_POSE)) / W.ACTION_SCALE
    return a.astype(np.float32).clip(-clip, clip)


def contacts(env) -> list[str]:
    m, d = env.model, env.data
    out = set()
    for i in range(d.ncon):
        for g in (d.contact[i].geom1, d.contact[i].geom2):
            b = int(m.geom_bodyid[g])
            if b:
                out.add(m.body(b).name.replace("_link", ""))
    return sorted(out)


def hold(env, w, secs=6.0):
    """Command one waypoint for `secs` and report where it ends up.

    6 s, not 2: a knee-1.00 crouch sits at 0.511 m for two full seconds and
    collapses at three, so a short window reports a slow collapse as a hold.
    """
    place(env, w.joints, w.root_pitch)
    a = _action(w.joints, env.robot.action_clip)
    tb = env.trunk_body_id
    rated = np.array([env.model.actuator(n).forcerange[1] for n in W.JOINT_NAMES])
    n = int(secs * 50)
    peak = 0.0
    for k in range(n):
        env.data.xfrc_applied[:] = 0.0
        env.step(a)
        # STEADY torque, over the last second only. Measured across the whole
        # window every pose reads 100 %, because placing the robot 2 mm clear
        # of the floor makes the servos slam on the first step — a placement
        # transient, not what holding the pose costs.
        if k >= n - 50:
            f = np.abs([env.data.actuator_force[env.model.actuator(n2).id]
                        for n2 in W.JOINT_NAMES])
            peak = max(peak, float((f / rated).max()))
    return (float(env.data.xpos[tb][2]), float(-env._projected_gravity()[2]),
            contacts(env), peak)


def chain(env):
    """Ramp the servos from waypoint to waypoint, no policy, no feedback — and
    report how far it gets. `VERIFIED_THROUGH` is this answer."""
    w0 = R.WAYPOINTS[0]
    place(env, w0.joints, w0.root_pitch)
    tb = env.trunk_body_id
    cur = np.array(w0.joints, float)
    rows = []
    for i, w in enumerate(R.WAYPOINTS[1:]):
        secs = R.LEGS[i]
        n = int(secs * 50)
        for k in range(n):
            f = (k + 1) / n
            f = f * f * (3.0 - 2.0 * f)
            env.data.xfrc_applied[:] = 0.0
            env.step(_action(cur + (np.asarray(w.joints) - cur) * f,
                             env.robot.action_clip))
        cur = np.asarray(w.joints, float)
        rows.append((w.name, float(env.data.xpos[tb][2]),
                     float(-env._projected_gravity()[2]), contacts(env),
                     w.settled_z))
    return rows


def playback(env, dt=0.02):
    """Follow the reference open loop, sampling it at the control rate."""
    q0, rp0, _ = R.pose_at(0.0)
    place(env, q0, rp0)
    tb = env.trunk_body_id
    rows, t = [], 0.0
    total = R.duration()
    last = None
    while t < total + 1.0:
        q, _, phase = R.pose_at(t)
        env.data.xfrc_applied[:] = 0.0
        env.step(_action(q, env.robot.action_clip))
        if phase != last:
            rows.append((t, phase, float(env.data.xpos[tb][2]),
                         float(-env._projected_gravity()[2]), contacts(env)))
            last = phase
        t += dt
    rows.append((t, "END", float(env.data.xpos[tb][2]),
                 float(-env._projected_gravity()[2]), contacts(env)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="check each waypoint against its recorded numbers")
    ap.add_argument("--secs", type=float, default=6.0)
    args = ap.parse_args()

    env = _env()
    env.reset(seed=0)

    # EQUILIBRIA are checked by placing them fresh and holding; a TRANSIT pose
    # has no fresh-placement answer to check — placed cold it settles somewhere
    # else entirely, which is what "transit, not equilibrium" means — so its
    # recorded numbers come off the chain below instead.
    print(f"{'equilibrium':<16} {'z':>6} {'u':>6} {'Tq':>5}  contacts")
    bad = 0
    for w in R.WAYPOINTS:
        if not w.equilibrium:
            continue
        z, u, con, peak = hold(env, w, args.secs)
        held = abs(z - w.settled_z) < 0.02 and abs(u - w.settled_u) < 0.06
        print(f"{w.name:<16} {z:>6.3f} {u:>+6.2f} {peak*100:>4.0f}%  "
              f"{','.join(c[:9] for c in con)[:30]}  {'ok' if held else 'DRIFTED'}")
        if args.verify and not held:
            print(f"                 recorded z {w.settled_z:.3f} u {w.settled_u:+.2f}")
            bad += 1

    print("\nopen-loop chain, servos only — how far does the reference go "
          "with no policy at all?")
    reached = None
    for name, z, u, con, want in chain(env):
        near = abs(z - want) < 0.04
        if near:
            reached = name
        print(f"  {name:<16} z {z:>6.3f} (authored {want:.3f})  u {u:>+5.2f}  "
              f"{'ON REFERENCE' if near else 'off':<12} "
              f"{','.join(c[:9] for c in con)[:30]}")
    print(f"\n  reached: {reached}   (module says VERIFIED_THROUGH = "
          f"{R.VERIFIED_THROUGH!r})")
    if args.verify and reached != R.VERIFIED_THROUGH:
        print("  MISMATCH: update VERIFIED_THROUGH or find out what changed")
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
