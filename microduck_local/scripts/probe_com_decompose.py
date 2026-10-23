"""WHERE is the CoM, limb by limb — and how much of the gap is the ARMS?

Every crawl search here scored `com_x - mean(foot_x)` with the forearms
planted well forward of the shoulders, because that is the posture the crawl
is in. But a rise TUCKS the arms, and the arms are not massless: scored with
them out front, the gap charges the crawl for mass it is about to move.

So the gap is decomposed by limb group at a fixed LEG pose, and then recomputed
with the arms kinematically replaced by the STAND keyframe's arm joints. If the
gap collapses when the arms come in, the crawl's 471 mm was mostly the arms and
the feet are closer to under the mass than the number said. If it does not, the
gap is the TRUNK, and no amount of stepping or hand-walking addresses it.

Reports no stepping at all — this is kinematics (`mj_forward`), so nothing here
depends on a controller or on anything holding.
"""
from __future__ import annotations

import argparse
import numpy as np

CLIP = 4.0

GROUPS = {
    "arms": ("shoulder", "elbow", "wrist"),
    "legs": ("hip", "knee", "ankle"),
    "trunk+head": ("pelvis", "waist", "torso", "head", "neck", "base"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pose", default="fold",
                    help="waypoint to drive the servos to before measuring")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=a.seed,
        getup_label="prone", actuator_force="dc", action_clip=CLIP,
        fell_cut=False, max_episode_s=30.0)
    m, d = env.model, env.data
    JN = list(W.JOINT_NAMES)
    default = np.asarray(env.robot.default_pose, float)
    scale = np.asarray(env.robot.action_scale, float)
    names = [w.name for w in R.WAYPOINTS]
    route = names[:names.index(a.pose) + 1]

    env.reset(seed=a.seed)
    w0 = R.BY_NAME[route[0]]
    env._place(np.asarray(w0.joints, float), w0.root_pitch)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    cur = np.asarray(w0.joints, float)
    for nm in route[1:]:
        tgt = np.asarray(R.BY_NAME[nm].joints, float)
        for k in range(13):
            q = cur + (tgt - cur) * min(1.0, (k + 1) / 12)
            env.step(np.clip((q - default) / scale, -CLIP, CLIP).astype(np.float32))
        cur = tgt

    FB = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]

    def group_of(bid):
        nm = m.body(bid).name.lower()
        for g, keys in GROUPS.items():
            if any(k in nm for k in keys):
                return g
        return "trunk+head"

    def com_x():
        wm = m.body_mass
        return float((wm * d.xipos[:, 0]).sum() / wm.sum())

    def gap():
        return com_x() - float(np.mean([d.xpos[b][0] for b in FB]))

    # The PELVIS gap is the foot-placement question; the CoM gap is not.
    # With the trunk horizontal, the whole-body CoM sits ~half a body length
    # ahead of the feet NO MATTER where the feet are, so minimising it asks
    # for a trunk rotation and reads a stepping failure where there is none.
    # A deep tucked squat — chest on the knees — has the CoM well forward and
    # is a perfectly good configuration to rise from, because the trunk comes
    # up afterwards. What has to be under the feet FIRST is the pelvis.
    # NOT a silent fallback: a wrong body here reports a different robot.
    pelvis = m.body("pelvis_link").id

    def pgap():
        return float(d.xpos[pelvis][0]) - float(
            np.mean([d.xpos[b][0] for b in FB]))

    total = float(m.body_mass.sum())
    print(f"pose '{a.pose}': trunk {float(d.xpos[env.trunk_body_id][2]):.3f} m, "
          f"span {env._foot_span():.3f} m, CoM-to-foot gap {gap():+.3f} m, "
          f"PELVIS-to-foot gap {pgap():+.3f} m")
    print(f"  total mass {total:.2f} kg, foot x "
          f"{float(np.mean([d.xpos[b][0] for b in FB])):+.3f} m\n")
    print(f"  {'group':<12s}{'mass kg':>9s}{'share':>8s}{'CoM x':>9s}"
          f"{'x - foot':>10s}{'contrib':>9s}")
    foot_x = float(np.mean([d.xpos[b][0] for b in FB]))
    for g in list(GROUPS) + ["other"]:
        ids = [b for b in range(m.nbody) if b and group_of(b) == g]
        if not ids:
            continue
        mg = float(m.body_mass[ids].sum())
        if mg <= 0:
            continue
        cx = float((m.body_mass[ids] * d.xipos[ids, 0]).sum() / mg)
        print(f"  {g:<12s}{mg:9.2f}{mg / total:8.1%}{cx:+9.3f}"
              f"{cx - foot_x:+10.3f}{(mg / total) * (cx - foot_x):+9.3f}")

    # The counterfactual: the SAME legs, the arms at the STAND keyframe.
    arm_j = [i for i, n in enumerate(JN)
             if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    stand = np.asarray(R.BY_NAME["stand"].joints, float)
    qs = [int(m.joint(n).qposadr[0]) for n in JN]
    before = gap()
    saved = d.qpos.copy()
    for i in arm_j:
        d.qpos[qs[i]] = stand[i]
    mujoco.mj_forward(m, d)
    after = gap()
    print(f"\n  arms AS PLANTED      gap {before:+.3f} m")
    print(f"  arms at STAND pose   gap {after:+.3f} m   "
          f"({after - before:+.3f} m)")
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    # And the same two numbers at the keyframe that IS a stand, for scale.
    st = np.asarray(R.BY_NAME["stand"].joints, float)
    env._place(st, R.BY_NAME["stand"].root_pitch)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    print(f"\n  at the STAND keyframe: CoM gap {gap():+.3f} m, "
          f"pelvis gap {pgap():+.3f} m, span {env._foot_span():.3f} m")
    print("\n  (kinematic — mj_forward only, nothing was asked to hold it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
