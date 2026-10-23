"""Fallen poses for Asimov 1 — the reset distribution a get-up policy trains on.

A get-up policy is an EXPLORATION problem before it is a reward problem: from a
sprawl, random torques essentially never reach standing, so the payoff is never
seen and no amount of reward tuning finds it (`AGENTS.md`, "Reward design
rules"). What makes it findable is the state the episode STARTS in — so the
first artefact is not a reward, it is a cache of fallen poses to reset into.

**The procedure is Menlo's, ported from Isaac to MuJoCo.** `cyclotron` PR #7
(`scripts/getup/build_fallen_cache.py`, BSD-3-Clause) adapts it in turn from
NVIDIA's WBC-AGILE `fallen_state_dataset.py` (Apache-2.0). Drop a LIMP robot
from a random orientation, let it settle, keep the states that came to rest,
and label them by how the torso ended up. Their numbers, kept here because they
are measured on this robot and guessing them again would be wasteful:

* limp means `Kp 0, Kd 0.5` — NOT zero torque. A zero-torque drop lets the
  limbs flail to the joint limits and settles into poses no fallen robot holds.
* drops of 0.3-0.8 m for the sprawls, 0-0.1 m for the propped/kneeling seeds.
* 2 s of settling, then keep only what is finite, interpenetration-free and
  AT REST (root speed < 0.05 m/s).

The labeller's thresholds are theirs too, and the rule ORDER matters — a
kneeling robot also satisfies the sitting test if you check sitting first.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import asimov_walk as W

#: Bodies the labeller asks about. The torso is the waist link, which is what
#: `label_fallen_states` orients on — not the pelvis, which is below the hips
#: and rotates with the legs.
TORSO_BODY = "waist_yaw_link"
PELVIS_BODY = "pelvis_link"
THIGH_BODIES = ("left_hip_pitch_link", "right_hip_pitch_link",
                "left_hip_yaw_link", "right_hip_yaw_link")
KNEE_BODIES = ("left_knee_link", "right_knee_link")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

#: A body is "touching" when its lowest collision point is within this of the
#: floor. Menlo's `LABEL_CONTACT_TOL`.
CONTACT_TOL = 0.02
#: Kneeling needs the knee this far below the pelvis origin, and the pelvis
#: this high — otherwise a collapsed heap labels as kneeling.
KNEEL_KNEE_BELOW_PELVIS = 0.10
KNEEL_PELVIS_MIN = 0.18
#: Sitting is an upright torso with the pelvis down.
SIT_PELVIS_MAX = 0.20
#: |torso up-axis z| below this is LYING; below -0.5 is inverted.
UPRIGHT_UPZ = 0.5

LABELS = ("supine", "prone", "side_left", "side_right", "sitting", "kneeling")
OTHER = "other"

#: Limp: no position term, a little damping. Menlo's `Kp 0, Kd 0.5`.
LIMP_KD = 0.5
#: Settle time and the speed below which a state counts as at rest.
SETTLE_S = 2.0
AT_REST_SPEED = 0.05


@dataclass(frozen=True)
class FallenPose:
    """One settled pose: the full `qpos`, its label, and why it was kept."""

    qpos: np.ndarray
    label: str
    torso_upz: float
    pelvis_z: float
    root_speed: float


def limp_model(joint_friction: bool = True) -> mujoco.MjModel:
    """The walk model with every servo made limp.

    Built by editing the compiled model's actuator gains rather than the spec,
    so it is the SAME model the policy will act on — a cache collected on a
    different body is a cache of poses the trainer can never reset into.
    """
    model = W.walk_spec(True, joint_friction).compile()
    for a in range(model.nu):
        model.actuator_gainprm[a, 0] = 0.0
        model.actuator_biasprm[a, 1] = 0.0
        model.actuator_biasprm[a, 2] = -LIMP_KD
    return model


def body_lowest(model: mujoco.MjModel, data: mujoco.MjData,
                body: str) -> float:
    """Lowest point of a body's collision geoms above the floor, or +inf.

    The geom AABB in WORLD axes, the arithmetic `robots/body.geom_aabb_world`
    uses — not the bounding sphere, which on a 0.4 m shin reports a point
    0.1 m below the shin and calls everything "touching".
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        return float("inf")
    low = float("inf")
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or not model.geom_contype[g]:
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        centre = data.geom_xpos[g] + rot @ model.geom_aabb[g, :3]
        ext = np.abs(rot) @ model.geom_aabb[g, 3:]
        low = min(low, float(centre[2] - ext[2]))
    return low


def label(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[str, dict]:
    """Menlo's `label_fallen_states`, rule for rule and in their ORDER."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
    rot = data.xmat[tid].reshape(3, 3)
    f_z, l_z, u_z = float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2])
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)
    pelvis_z = float(data.xpos[pelvis_id][2])

    def touching(names) -> bool:
        return any(body_lowest(model, data, b) < CONTACT_TOL for b in names)

    knee_low = min(body_lowest(model, data, b) for b in KNEE_BODIES)
    feat = {"u_z": u_z, "f_z": f_z, "l_z": l_z, "pelvis_z": pelvis_z,
            "knee_low": knee_low}

    if u_z < -UPRIGHT_UPZ:                                  # 1. inverted
        return OTHER, feat
    if (knee_low < CONTACT_TOL and not touching([PELVIS_BODY])
            and pelvis_z >= KNEEL_PELVIS_MIN
            and knee_low < pelvis_z - KNEEL_KNEE_BELOW_PELVIS):
        return "kneeling", feat                             # 2. kneeling
    if u_z >= UPRIGHT_UPZ:
        if (pelvis_z <= SIT_PELVIS_MAX
                and (touching([PELVIS_BODY]) or touching(THIGH_BODIES))
                and not touching([TORSO_BODY])):
            return "sitting", feat                          # 3. sitting
        return OTHER, feat                                  # 4. upright, other
    # 5. lying: which way is the chest facing?
    return LABELS[int(np.argmax([f_z, -f_z, -l_z, l_z]))], feat


def _random_spawn(model: mujoco.MjModel, data: mujoco.MjData,
                  rng: np.random.Generator, drop: tuple[float, float],
                  vel: float, alpha: float) -> None:
    """A limp robot above the floor, random orientation and joints.

    `alpha` blends the STAND pose toward a uniform draw inside each joint's
    range: 0 keeps the stand pose (a robot that fell while standing), 1 is
    fully random (a robot that fell from anything).
    """
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, W.STAND_KEY)
    mujoco.mj_resetDataKeyframe(model, data, key)
    quat = rng.normal(size=4)
    data.qpos[3:7] = quat / np.linalg.norm(quat)
    for name in W.JOINT_NAMES:
        adr = int(model.joint(name).qposadr[0])
        lo, hi = model.jnt_range[model.joint(name).id]
        target = rng.uniform(lo, hi) if hi > lo else data.qpos[adr]
        data.qpos[adr] = (1 - alpha) * data.qpos[adr] + alpha * target
    data.qvel[:] = 0.0
    data.qvel[0:3] = rng.normal(scale=vel, size=3)
    data.qvel[3:6] = rng.normal(scale=vel, size=3)
    # Lift so the LOWEST collision point sits at the requested drop height.
    mujoco.mj_forward(model, data)
    lowest = min(body_lowest(model, data, model.body(b).name)
                 for b in range(1, model.nbody))
    data.qpos[2] += rng.uniform(*drop) - lowest
    mujoco.mj_forward(model, data)


def collect(n: int = 400, seed: int = 0, drop: tuple[float, float] = (0.3, 0.8),
            vel: float = 1.0, alpha: float = 1.0,
            settle_s: float = SETTLE_S) -> tuple[list[FallenPose], dict]:
    """Drop `n` limp robots and keep the ones that settled. Returns
    `(poses, stats)`; `stats` is the yield table, which is the number that
    says whether a cache is usable."""
    model = limp_model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    steps = int(settle_s / model.opt.timestep)
    poses: list[FallenPose] = []
    stats = {"tried": 0, "nonfinite": 0, "moving": 0, "kept": 0,
             **{k: 0 for k in LABELS}, OTHER: 0}
    for _ in range(n):
        stats["tried"] += 1
        _random_spawn(model, data, rng, drop, vel, alpha)
        data.ctrl[:] = 0.0
        for _ in range(steps):
            mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            stats["nonfinite"] += 1
            continue
        speed = float(np.linalg.norm(data.qvel[0:3]))
        if speed >= AT_REST_SPEED:
            stats["moving"] += 1
            continue
        name, feat = label(model, data)
        stats[name] += 1
        if name == OTHER:
            continue
        stats["kept"] += 1
        poses.append(FallenPose(data.qpos.copy(), name, feat["u_z"],
                                feat["pelvis_z"], speed))
    return poses, stats


# ------------------------------------------------------------- the cache

#: Where the collected poses live. Beside the model, because they are only
#: valid FOR that model: a pose is a `qpos`, and a revision that renames or
#: reorders a joint makes every cached pose a different posture.
def cache_path(dest=None):
    from . import asimov

    return (dest or asimov.cache_dir()) / "fallen_poses.npz"


#: Drops to collect when building. 500 yields ~240 supine and ~220 prone in
#: 23 s (MEASURED); the rare categories need targeted spawn recipes, not more
#: drops — see the module docstring.
BUILD_DROPS = 500
BUILD_SEED = 1


def fall_ladder(n: int = 8, push_n: float = 260.0, seed: int = 0):
    """Rungs sampled from the robot FALLING, labelled `fall_0` (lowest) upward.

    Every ladder before this one used poses that are not on any feasible path.
    `blend_to_stand` INTERPOLATES toward standing, and MEASURED 2026-10-04 the
    result is not an intermediate pose: at alpha 0.8 the robot is one-footed
    (the other foot 5 cm up), lands both feet only at height 0.885 / upright
    0.83 — under the 0.90/0.90 `up` gate — and has collapsed to 0.292 by
    t = 1.0 s even with half its weight carried. The authored ladder uses the
    pike, whose trunk reaches 98 deg against the 50 deg the trained stand can
    recover from. So `standing` was never once sampled in 9.1 M steps and an
    unsampled state's value is never learned.

    A rung does not need to be STABLE — no feet-only equilibrium exists below
    trunk 0.516 m, so there are none to find. It needs to be REACHABLE. A state
    on a fall is reachable by construction, and a get-up is a fall run
    backwards. Pushed over from standing the robot passes 0.638 -> 0.009 m with
    only ~8 frames inside the 0.25-0.52 m gap, which also says what the motion
    has to be: a ballistic crossing, not a staged one.
    """
    import mujoco

    from . import asimov_walk as AW
    from .spec import get as _get

    env = _get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=seed,
        getup_blend=0.0, getup_assist=0.0, getup_label="prone")
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    for name, v in zip(AW.JOINT_NAMES, AW.DEFAULT_POSE):
        d.qpos[m.joint(name).qposadr[0]] = v
    mujoco.mj_forward(m, d)
    d.qvel[:] = 0.0
    traj = []
    for k in range(400):
        d.ctrl[:len(AW.JOINT_NAMES)] = AW.DEFAULT_POSE
        d.xfrc_applied[:] = 0.0
        if k < 30:
            d.xfrc_applied[env.trunk_body_id, 0] = push_n
        mujoco.mj_step(m, d, 4)
        traj.append((float(d.xpos[env.trunk_body_id][2]), d.qpos.copy()))
    # evenly in HEIGHT, not in time: the fall spends most of its frames already
    # on the floor, and the rungs that matter are the ones in the gap.
    lo, hi = min(z for z, _ in traj), max(z for z, _ in traj)
    out = []
    for i in range(n):
        want = lo + (hi - lo) * (i + 0.5) / n
        z, q = min(traj, key=lambda t: abs(t[0] - want))
        out.append((f"fall_{i}", q, z))
    return out


def load_cache(label: str = "supine", dest=None, rebuild: bool = False):
    """`[N, nq]` of settled poses with this label, building the cache if needed.

    Written atomically (temp + `os.replace`) because a training run builds 32
    workers and any of them may be the one that finds the file missing —
    `AGENTS.md`, "Atomic writes and live imports". A worker that loses the
    race writes its own temp and replaces; the content is identical, since the
    seed is fixed.
    """
    import os

    path = cache_path(dest)
    if rebuild or not path.is_file():
        poses, stats = collect(n=BUILD_DROPS, seed=BUILD_SEED)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        # A FILE HANDLE, not a path: `np.savez` appends `.npz` to any name
        # that lacks it, so a `.tmp` path is saved as `.tmp.npz` and the
        # `os.replace` below then fails on a file that was never written.
        with open(tmp, "wb") as fh:
            np.savez(fh, qpos=np.stack([p.qpos for p in poses]),
                     label=np.array([p.label for p in poses]))
        os.replace(tmp, path)
        print(f"[asimov] built {path.name}: {stats['kept']} poses from "
              f"{stats['tried']} drops", flush=True)
    d = np.load(path, allow_pickle=True)
    have = sorted(set(d["label"].tolist()))
    # A SET of postures, comma-separated, or "all" — the same spelling the
    # reference env's `start_phase` uses. One label was the old behaviour and
    # is still what a bare name means.
    #
    # Why it matters: the cache holds 477 poses across `kneeling, prone,
    # side_left, side_right, sitting, supine`, and training drew only from
    # `supine` — both narrower than the task and, per this repo's own ledger,
    # the hardest single posture ("it cannot rise from supine at all"). The
    # one policy known to get up on this model
    # (openhorizon-labs/asimov1-getup-safefall, 99.8 %, 1.70 s on the stock
    # MJCF) is trained on exactly the set above: "back, front, side, sitting,
    # kneeling". Training on one sixth of the distribution and concluding the
    # task is impossible is not a measurement of the task.
    names = have if label.strip() == "all" else [
        x.strip() for x in label.split(",") if x.strip()]
    missing = [n for n in names if n not in have]
    if missing:
        raise RuntimeError(
            f"no {missing!r} poses in {path} — it holds {have}")
    want = np.isin(d["label"], names)
    return d["qpos"][want]


def blend_to_stand(model, qpos: np.ndarray, alpha: float) -> np.ndarray:
    """A pose `alpha` of the way from `qpos` back to standing. 0 = as it fell.

    The REVERSE CURRICULUM, and the lever the reward could not supply. Run
    990c5d got the reward right — a product no single trick can farm, which
    scored the previous run's shoulder stand at -55 — and then learned almost
    nothing (`stand` 64 of a possible 3200), because a correct reward that is
    ~0 everywhere has no gradient. The skill has to be SAMPLED before it can
    be paid for, and the way to sample it is to start near the goal and walk
    backwards (`AGENTS.md`: "fix the physics curriculum, not the reward").

    Orientation is SLERPed and joints are lerped toward the stand pose, then
    the whole body is dropped so its lowest collision point rests on the
    floor — an interpolated pose floating 10 cm up would train a get-up that
    begins with a fall.
    """
    from . import asimov_walk as AW

    out = np.array(qpos, dtype=np.float64).copy()
    a = float(min(max(alpha, 0.0), 1.0))
    if a <= 0.0:
        return out

    # root orientation: slerp toward upright (identity quaternion)
    q0 = out[3:7] / max(np.linalg.norm(out[3:7]), 1e-9)
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = np.arccos(min(max(dot, -1.0), 1.0))
        q = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    out[3:7] = q / max(np.linalg.norm(q), 1e-9)

    # joints toward the stand pose
    for i, name in enumerate(AW.JOINT_NAMES):
        adr = int(model.joint(name).qposadr[0])
        out[adr] = (1 - a) * out[adr] + a * float(AW.DEFAULT_POSE[i])

    # sit it on the floor rather than wherever the interpolation left it
    data = mujoco.MjData(model)
    data.qpos[:] = out
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    lowest = min(body_lowest(model, data, model.body(b).name)
                 for b in range(1, model.nbody))
    out[2] -= lowest
    return out
