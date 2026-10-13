"""The contract Asimov 1's shipped policy speaks, and the driver that runs it.

    78 observations -> 23 joint-position actions, at 50 Hz.

Every constant here is READ from the official artifact — `env.yaml` inside
[Menlo/asimov1-locomotion-0818](https://huggingface.co/Menlo/asimov1-locomotion-0818)
at the pinned revision — and `tests/test_asimov_contract.py` re-reads that file
and fails if any of them drifts. They are mirrored rather than parsed at import
for `contract.py`'s reason: the deployment contract has to be readable in one
place, and a 62 kB YAML with `!!python/object/apply` tags in it is not that
place. The test is what keeps the mirror honest.

**Why this file exists at all.** The point of bringing a stranger's robot in
is not that it renders — it is that OUR environment and THEIR policy agree
about what an observation is. 78 floats in the wrong order still produce a
number from the graph, and the robot still moves; it just falls over, and
every downstream measurement describes a different robot. So the first thing
built here is their layout exactly, and the first thing measured is whether
their own shipped policy walks in our MuJoCo. If it does, the layout is right
and everything after it can be trusted. That check is `scripts/probe_asimov_walk.py`.

**Two sources, cross-checked.** The viewer's `reference_policy_config.json`
publishes the same layout as integer index lists into the 23-joint action
order; `env.yaml` publishes it as joint NAMES. They agree, and both agree with
the MJCF's own joint order — which is not a given, since the policy was
trained in Isaac on the URDF and runs here on the MJCF. `JOINT_NAMES` is
asserted against the compiled model rather than trusted.

**The slots are CAN buses, not anatomy.** `slot01`, `slot23` and `slot45`
interleave left and right and put the waist between a hip and a shoulder,
because they are motor positions on the robot's five CAN buses — Menlo's own
blog says the hip motors read 6-9 ms stale by the time the ankle is sampled,
and a per-slot observation term is how that latency is modelled per group.
Guessing this grouping is not possible; it is why the layout is copied and
asserted rather than inferred.

**What is NOT modelled here.** The 0-5 control-step actuator delay and the
observation noise both exist in `env.yaml` and both are training-time domain
randomisation. The driver takes them as arguments and defaults them OFF, so a
check of "does their policy walk" is deterministic and repeatable
(`AGENTS.md`, "Verify policies deterministically").
"""

from __future__ import annotations

import os
import struct
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from . import asimov
from .spec import RobotSpec as _SPEC_BASE

# --------------------------------------------------------------- the policy

#: The shared run this contract was read from, pinned. A different revision is
#: a different policy AND possibly a different layout — `--share` uploads the
#: env.yaml beside the graph precisely so the two travel together.
POLICY_REPO = "Menlo/asimov1-locomotion-0818"
POLICY_SHA = "18114a730668a983423e6361f0522eb62c76cb74"
#: BSD-3-Clause (the model card's own `license:`), unlike the MJCF's GPL-2.0 /
#: CERN-OHL-S-2.0 — so the two halves of this robot are not under one licence
#: and both are fetched rather than vendored.
POLICY_LICENCE = "BSD-3-Clause"
POLICY_FILES = ("policy.onnx", "env.yaml", "agent.yaml")


def policy_dir(dest: Path | None = None) -> Path:
    return (dest or asimov.cache_dir()) / "policy"


def policy_onnx(dest: Path | None = None) -> Path:
    return policy_dir(dest) / "policy.onnx"


def policy_env_yaml(dest: Path | None = None) -> Path:
    return policy_dir(dest) / "env.yaml"


def policy_ready(dest: Path | None = None) -> bool:
    return all((policy_dir(dest) / f).is_file() for f in POLICY_FILES)


def fetch_policy(dest: Path | None = None) -> Path:
    """Download the shipped policy from the Hub at `POLICY_SHA`.

    `huggingface_hub` rather than this module's own HTTP, unlike
    `robots/asimov.fetch`: the Hub serves `policy.onnx` through LFS with a
    redirect to a CDN, and the integrity check is the Hub's own ETag/sha256
    against the pinned revision. `setup.sh` already downloads the duck's
    shipped policies this way.
    """
    from huggingface_hub import hf_hub_download

    out = policy_dir(dest)
    out.mkdir(parents=True, exist_ok=True)
    for name in POLICY_FILES:
        print(f"[asimov] {POLICY_REPO}/{name}")
        hf_hub_download(repo_id=POLICY_REPO, filename=name,
                        revision=POLICY_SHA, local_dir=str(out))
    print(f"[asimov] policy at {out} — {POLICY_REPO} "
          f"@ {POLICY_SHA[:12]} ({POLICY_LICENCE})")
    return out


def require_policy(dest: Path | None = None) -> Path:
    if not policy_ready(dest):
        raise RuntimeError(
            "Asimov 1's shipped policy is not downloaded — run "
            "`uv run python -m microduck_local.robots.asimov_walk --fetch` "
            f"({POLICY_REPO} @ {POLICY_SHA[:12]}, {POLICY_LICENCE})")
    return policy_dir(dest)


# -------------------------------------------------------------- the contract

OBS_DIM = 78
NUM_JOINTS = 23
PHYSICS_DT = 0.005                 # env.yaml sim.dt
DECIMATION = 4                     # env.yaml decimation
CTRL_DT = PHYSICS_DT * DECIMATION  # 0.02 s -> 50 Hz
EPISODE_S = 20.0                   # env.yaml episode_length_s

#: The ACTION order: 23 joints, and the order `last_action` is written in.
#: Asserted against the compiled model in `joint_ids` — the MJCF's own tree
#: order happens to match, which is luck worth checking rather than relying on.
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
)

#: The three observation slots, in `env.yaml`'s order — see the module
#: docstring on why these are CAN buses rather than limbs.
SLOT_01: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
)
SLOT_23: tuple[str, ...] = (
    "left_hip_yaw_joint", "left_knee_joint",
    "right_hip_yaw_joint", "right_knee_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
)
SLOT_45: tuple[str, ...] = (
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "right_wrist_yaw_joint", "left_wrist_yaw_joint",
)
SLOTS: tuple[tuple[str, ...], ...] = (SLOT_01, SLOT_23, SLOT_45)

#: Obs block boundaries, in contract order. Written out rather than computed
#: so a reader can check them against `env.yaml` without running anything.
#:   [0:3] ang_vel*0.25  [3:6] gravity  [6:9] twist cmd
#:   [9:32] joint_pos_rel by slot   [32:55] joint_vel*0.1 by slot
#:   [55:78] last_action
OBS_ANG_VEL = (0, 3)
OBS_GRAVITY = (3, 6)
OBS_COMMAND = (6, 9)
OBS_JOINT_POS = (9, 32)
OBS_JOINT_VEL = (32, 55)
OBS_LAST_ACTION = (55, 78)

ANG_VEL_SCALE = 0.25               # env.yaml observations.policy.base_ang_vel
JOINT_VEL_SCALE = 0.1              # ...joint_vel_slot*.scale

#: Observation noise, uniform half-widths (env.yaml, enable_corruption: true).
#: Training-time DR; the driver defaults it off.
NOISE_ANG_VEL = 0.01
NOISE_GRAVITY = 0.02
NOISE_JOINT_POS = 0.01
NOISE_JOINT_VEL = 0.5

#: target = default_joint_pos + ACTION_SCALE * action  (env.yaml
#: actions.joint_pos: scale 0.25, use_default_offset true). NOT the per-joint
#: array in the viewer's `reference_policy_config.json`, which that file
#: labels a leftover from an older run that `env.yaml` overrides.
ACTION_SCALE = 0.25
#: The clip stays `RobotSpec`'s inherited 4.0, and that is a MEASUREMENT, not
#: an oversight. Theirs is `clip_actions: null` (policy/agent.yaml line 35) and
#: 4.0 is the duck's, so at scale 0.25 this robot is capped at +/-1.000 rad from
#: standing -- 76 % of its joint range on average, 32 % on a wrist -- and six of
#: the nine authored get-up waypoints need more on 2-6 joints each
#: (`plant_forearms` 6.00 on its elbows, `quadruped` 5.20, `pike` and
#: `draw_knees_in` 4.80, `fold` 5.00). TRIED 2026-10-04: a 13.0 clip, derived as
#: max|jnt_range - DEFAULT_POSE| / ACTION_SCALE so the whole physically
#: reachable set is commandable. It made things WORSE -- `fold` stopped holding
#: (settled 0.242 m at u -0.43, against the recorded 0.517 / +0.83) and
#: `quadruped` drifted, because what every verified hold and the one working
#: transition were measured on is the CLIPPED pose, and the authored pose is a
#: different, unstable one. The cost of the clip is small where it is actually
#: paid: `_joint_err` is a MEAN over 23 joints, so truncating 2-6 of them is a
#: 0.02-0.04 rad floor and an 0.84-0.92 ceiling on the tracking term, a tax and
#: not a wall. The open-loop chain dies at `rock_back` under BOTH clips, which
#: is what says the route past the pike is the problem and the command range is
#: not. `test_every_waypoint_is_commandable_enough_to_track` pins the numbers.

#: env.yaml scene.robot.init_state.joint_pos, resolved onto JOINT_NAMES.
DEFAULT_POSE_BY_NAME: dict[str, float] = {
    "left_hip_pitch_joint": -0.15, "right_hip_pitch_joint": 0.15,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.45, "right_knee_joint": -0.45,
    "left_ankle_pitch_joint": -0.30, "right_ankle_pitch_joint": 0.30,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": -0.25, "right_shoulder_pitch_joint": 0.25,
    "left_shoulder_roll_joint": -0.05, "right_shoulder_roll_joint": 0.05,
    "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.40, "right_elbow_joint": -0.40,
    "left_wrist_yaw_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}
DEFAULT_POSE = np.array([DEFAULT_POSE_BY_NAME[n] for n in JOINT_NAMES],
                        np.float32)

#: Spawn height (cyclotron `ASIMOV_1_STANDING_INIT_STATE.pos`). The model's
#: own qpos0 stands at 0.630 with straight legs; this pose has bent knees.
STAND_Z = 0.639
STAND_KEY = "STAND"

#: Episode cut: `bad_orientation(limit_angle=70 deg)` — the same cut the
#: duck's `fall_gravity_z = -0.342` encodes, which is a coincidence worth
#: naming rather than a shared convention.
FALL_TILT_RAD = np.deg2rad(70.0)
FALL_GRAVITY_Z = float(-np.cos(FALL_TILT_RAD))      # -0.342

#: Command ranges the policy was trained over (cyclotron velocity_env_cfg
#: `UniformVelocityCommandCfg.Ranges`), m/s and rad/s.
LIN_VEL_X_RANGE = (-0.6, 0.8)
LIN_VEL_Y_RANGE = (-0.5, 0.5)
ANG_VEL_Z_RANGE = (-0.8, 0.8)

#: The slowest forward command this policy actually WALKS at. MEASURED on the
#: shipped policy, 8 s a command, distance travelled against what was asked:
#:
#:     cmd 0.100 -> 0.001 m/s ( 1 %)    cmd 0.250 -> 0.218 m/s (87 %)
#:     cmd 0.125 -> 0.003 m/s ( 3 %)    cmd 0.300 -> 0.268 m/s (89 %)
#:     cmd 0.150 -> 0.008 m/s ( 6 %)    cmd 0.400 -> 0.359 m/s (90 %)
#:     cmd 0.200 -> 0.149 m/s (75 %)    cmd 0.500 -> 0.442 m/s (88 %)
#:
#: A cliff, not a slope: below 0.2 it stands still and above it tracks within
#: ~10 %. `RobotSpec.min_forward_cmd` defaults to the duck's 0.3, which is why
#: this was inherited rather than measured — and inheriting it cost a /sim
#: level where the robot froze mid-tour (`world/arena.WorldPerson._forward`).
MIN_FORWARD_CMD = 0.2

#: Per-joint servo, in `env.yaml` scene.robot.actuators order, as
#: `(name fragment, kp, kd, effort N*m, armature kg*m^2, joint friction N*m)`.
#: Matched by SUBSTRING in the order given, so `hip_pitch` is tried before
#: anything that would also match `hip`; an unmatched joint raises.
GAIN_GROUPS: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("hip_pitch",      150.0, 5.0, 45.0, 0.0698, 0.70),
    ("hip_roll",       150.0, 5.0, 45.0, 0.1400, 0.20),
    ("hip_yaw",        150.0, 5.0, 28.0, 0.0687, 0.70),
    ("knee",           150.0, 5.0, 45.0, 0.0330, 0.70),
    ("ankle_pitch",    110.0, 5.0, 40.0, 0.0484, 0.40),
    ("ankle_roll",     110.0, 5.0, 17.0, 0.0484, 0.40),
    ("waist",           65.0, 5.0, 40.0, 0.0698, 0.70),
    ("shoulder_pitch",  57.0, 5.0, 30.0, 0.1400, 0.20),
    ("shoulder_roll",   86.0, 5.0, 25.0, 0.0330, 0.70),
    ("shoulder_yaw",    96.0, 5.0, 20.0, 0.0687, 0.70),
    ("elbow",           40.0, 2.0, 12.0, 0.0242, 0.40),
    ("wrist_yaw",       40.0, 2.0, 12.0, 0.0242, 0.40),
)


def gains_for(joint: str) -> tuple[float, float, float, float, float]:
    """`(kp, kd, effort, armature, friction)` for one joint name."""
    for frag, kp, kd, effort, armature, friction in GAIN_GROUPS:
        if frag in joint:
            return kp, kd, effort, armature, friction
    raise KeyError(
        f"no actuator group in env.yaml matches {joint!r} — a revision of "
        f"{asimov.REPO} added a joint this contract does not drive, and "
        "driving it with a neighbour's gains would be a guess")


# -------------------------------------------------------------- the look

#: Asimov's shells. MEASURED off menloresearch/asimov-mjlab's own README
#: images by clustering them (the CAD render's 49 % cluster is #141311, the
#: hardware photo's shell clusters are #373939 and #333b40) — and this takes
#: the PHOTO's, not the render's. The CAD sits on a white page where near
#: black is the strongest thing on it; the lab stage is #0c0d11, and a robot
#: at #141311 on it is a silhouette with no shape in it. That is the trade
#: `robots/mars.paint_shell` and `MarsLook.tsx` already measured from the
#: other end, and the photo's colour is also the machine as it stands in
#: their workshop.
ASIMOV_SHELL = (0.200, 0.212, 0.231, 1.0)       # #33363b graphite

#: The burnt orange on the thigh plates, knee discs, hip covers and top
#: plate. The CAD render's two orange clusters are #7c3f08 lit and #562b06
#: shadowed; those are a dim key light's midtones, so the ALBEDO behind them
#: is brighter than either — this is that base colour, checked by rendering
#: it back under MuJoCo's lights beside the crop.
ASIMOV_ACCENT = (0.761, 0.416, 0.078, 1.0)      # #c26a14

#: The shoe. Black in the photograph; the CAD's are clear covers over it
#: (#7b7c82 in the clustering), and a sole that reads lighter than the leg
#: above it reads as a robot standing in snow.
ASIMOV_FOOT = (0.106, 0.114, 0.129, 1.0)        # #1b1d21

#: The chest plate and the head shell. MEASURED off menlo.ai's standing hero
#: shot: 22,484 strongly-red torso pixels have a median of #fb4d2b. Taken
#: down a little here because that median is a LIT surface under a workshop
#: lamp and the albedo behind it is darker — a shell at the literal sample
#: clips to flat vermilion under the stage's own key light.
ASIMOV_RED = (0.839, 0.231, 0.141, 1.0)         # #d63b24

#: The hands. The real robot's grippers are white (#e1e3e3 / #d2d4d2 over
#: ~4,000 px). NOTE this model has no gripper at all — `wrist_yaw_link` is a
#: 19x38x38 mm stub where the claw would mount — so white here stands in for
#: a hand that is not in the mesh. It is the one entry in this table that
#: paints a part the robot does not have.
ASIMOV_HAND = (0.863, 0.871, 0.871, 1.0)        # #dcdede

#: The soles. Magenta on the real robot, and the one colour on it that is
#: not black, bronze, silver or red — cut off the bottom of each foot mesh
#: by `CUT_PARTS`.
#: The soles. Magenta on the real robot, and the one colour on it that is
#: neither black, bronze, silver nor red — it shows in every walking shot,
#: because a raised foot shows its underside. `CUT_PARTS` takes it off the
#: bottom 6 mm of each foot's SOURCE STL.
ASIMOV_SOLE = (0.839, 0.149, 0.345, 1.0)        # #d62658

#: The hip joints. MEASURED off menlo.ai's standing hero shot, where the
#: outer hip plate samples #d4cbc0 — a warm light grey, because the workshop
#: light is warm. Neutralised to machined aluminium here: a hip that stayed
#: warm would read as a second, duller accent beside the bronze rather than
#: as bare metal.
ASIMOV_SILVER = (0.749, 0.765, 0.784, 1.0)      # #bfc3c8

#: Link-name fragments that are an actuator HOUSING rather than a long
#: cover. Identical to `duck-viewer/components/AsimovLook.asimovPartKind`'s
#: rule, and the split is the model's own: every shell mesh is longer than
#: 148 mm and every housing shorter than 136 mm (that file lists the
#: extents). The foot is `ankle_roll`, the one joint-named link that is a
#: shell.
#: What stays bronze. The shoulder ROTATORS came out of this list on
#: 2026-10-04: in menlo.ai's front and back shots the only gold on the
#: shoulder is a small pad on the very top, and the rotator barrel under it
#: is black — painting the whole `shoulder_pitch` / `shoulder_yaw` link
#: bronze put two gold boulders where the robot has none. The model gives one
#: mesh per link and the pad is not separable from the barrel, so the barrel's
#: colour wins: it is the larger surface and the one that reads at distance.
ACCENT_LINKS = ("ankle_pitch", "neck_yaw")

#: The red shells: the torso and the HELMET. Both take the chest's red —
#: MEASURED off menlo.ai's walking shots, where the helmet samples #cf2a18
#: against the chest's #f7472a over 4,914 and 14,381 strongly-red pixels.
#: Same hue; the helmet is simply the shaded side, since it faces away from
#: the key light in both frames.
#:
#: Two compromises, both forced by one mesh per link and both visible:
#: `waist_yaw` is the WHOLE torso, so the back goes red with the chest (the
#: real robot does carry a red plate up there, so this is the closer of the
#: two options), and `neck_pitch` is the whole head, so the black visor goes
#: red with the helmet. Cutting either is what sank the torso 18 cm on
#: 2026-10-04 — see the note on `CUT_PARTS`.
#: `waist_yaw` is NOT here: the torso is CUT, so its red comes from
#: `CUT_PARTS` and only the front half gets it.
RED_LINKS: tuple = ()


#: The hands. `wrist_yaw` was an accent housing until the same look.
HAND_LINKS = ("wrist_yaw",)

#: The hip housings, silver rather than bronze. On the real robot the hip
#: plate is bare metal and the bronze starts at the knee — these two links
#: were accent until the menlo.ai photographs were looked at.
SILVER_LINKS = ("hip_pitch", "hip_roll")
FOOT_LINKS = ("ankle_roll",)


#: Parts CUT out of a link because the robot is two colours there and the
#: model is one mesh. `(link, name, keep, cut colour, rest colour)`, where
#: `keep(centroids, shells, normals)` picks the cut half's triangles **in the source
#: STL's own frame**.
#:
#: That frame is the whole trick, and getting it wrong cost a day. A cut must
#: split the triangles of the FILE on disk — exactly what
#: `mars.split_base_mesh` has always done for MARS's tyres. Splitting
#: `model.mesh_vert` instead looks identical and is not: MuJoCo recentres a
#: mesh on load and `mesh_vert` is the RECENTRED copy, so writing those
#: Parts CUT out of a link because the robot is two colours there and the
#: model gives the link one mesh. `(link, name, keep, cut colour, rest
#: colour)`, where `keep(centroids, shells, normals)` picks the cut half's triangles
#: **in the SOURCE STL's own frame**.
#:
#: That frame is the whole trick. A cut splits the triangles of the file on
#: disk — exactly what `mars.split_base_mesh` has always done for MARS's
#: tyres — and both halves then land within 4e-6 mm of the uncut mesh with no
#: position compensation at all. Splitting `model.mesh_vert` instead looks
#: identical and is not: that is MuJoCo's RECENTRED copy, so writing it back
#: subtracts the recentre twice. Done that way on 2026-10-04 it sank the
#: torso 18 cm, ate the neck column and left the head floating, and four
#: "corrections" were then measured against the wrong premise.
#:
#: The thresholds below are therefore in the FILE's frame, which is NOT the
#: compiled one — a sole threshold read off `mesh_vert` (-74 mm) missed a
#: foot spanning -34..+23 mm entirely and made `walk_spec()` raise. `cut_part`
#: refuses an empty or total selection for that reason, and caches on the
#: source mesh's mtime AND this module's so a threshold edit really re-cuts.
#:
#: Only PLANE-bounded parts belong here. Asimov's bronze knee disc is a
#: raised boss on a curved shell; six cuts failed to isolate it and it stays
#: graphite until somebody segments it by feature edges.
#:
#: MEASURED, in each source STL's own frame:
#:   torso  `WAIST_YAW.STL`, x -117..102 (+x forward), z -3.5..378 mm —
#:          the vest is the front above z = 140 mm, 24 % of its triangles.
#:   foot   `ANKLE_B_NEW.STL`, z -34.3..22.6 mm — the sole is below
#:          z = -28 mm, 2 540 triangles, the bottom 6 mm of the shoe.
CUT_PARTS: tuple = (
    # The chest plate. SHELL 2 is the outer cowl — the only part of the torso
    # that is red on the real robot — and the normal test takes the FRONT and
    # BACK faces of it, which is the robot's own layout: a red chest, a red
    # upper-back plate, black shoulders and black sides. The cowl's creases
    # where the plate meets the sides are what the seam lands on.
    #
    # It used to be `x > 0` (a slab over the entire front) and then
    # `shell + z > 120 mm`. Both were planes across a curved cowl, and
    # MEASURED with the seam probe that is the definition of serrated: 83 %
    # of the 120 mm seam ran through SMOOTH surface at a 2.1 deg median
    # dihedral. Every floor from 80 to 220 mm scored 72-96 %; there is no
    # crease partway up the cowl to cut along. `|nx| > 0.4` is 11.4 % smooth
    # at an 85 deg median — it follows real edges. A second normal test to
    # trim the midriff was tried and dropped: it did not raise the red at all
    # and took the seam back to 15-26 %.
    ("waist_yaw_link", "asimov_torso_vest",
     lambda c, s, n: (s == 2) & (np.abs(n[:, 0]) > 0.4), "RED", "SHELL"),
    # The soles, by which way the face POINTS rather than by height. The shoe
    # is one shell (the other 57 are internal fasteners) so a shell cut cannot
    # reach the sole — but a horizontal plane through a curved sole cuts
    # ACROSS the triangles and leaves a row of serrated teeth, which is
    # exactly what z < -28 mm shipped and what the magenta looked like on the
    # stage. The downward-facing faces end at the welt, which is a real crease
    # in the model, so the boundary is the shoe's own edge and is clean.
    # -0.5 is the measured threshold: it keeps the underside and the thin
    # bottom band the photographs show, and -0.9 loses the band.
    ("left_ankle_roll_link", "asimov_sole_l",
     lambda c, s, n: (s == 0) & (n[:, 2] < -0.5), "SOLE", "FOOT"),
    ("right_ankle_roll_link", "asimov_sole_r",
     lambda c, s, n: (s == 0) & (n[:, 2] < -0.5), "SOLE", "FOOT"),
    # The helmet's VISOR — everything but the outer rim. Not a plane: the
    # black sweeps the face and over the crown and the red is the frame
    # around it. It is a separate SHELL though, so `_triangle_shells` hands
    # it over: shell 0 is the 7 256-triangle rim, and the rest (inner panel,
    # two side plates, one scrap) is the dark.
    ("neck_pitch_link", "asimov_visor",
     lambda c, s, n: s != 0, "SHELL", "RED"),
    # The bronze elbow bands: each is its own shell, so `s != 0` is the
    # whole selection and there is no seam.
    #
    # These were called impossible on 2026-10-04 after plane cuts on the
    # COMPILED mesh left floating gold rings and sliced internal flanges.
    # They were never hard — MuJoCo welds vertices on load and had merged
    # what the CAD keeps apart. On the source STL each is a plate of its own.
    #
    # The shoulder PADS are the one place `s != 0` could not be used, because
    # it is not the same selection on the two sides. Each shoulder carries
    # two 10 x 117 x 67 mm plates — one forward at x = +39 mm, one aft at
    # x = -38 mm — with the rotator barrel between them. On the RIGHT link
    # the barrel is a single component; on the LEFT it arrives already split
    # in half (19 024 + 15 489 mm^2 against the right's one 34 514). Since
    # `_triangle_shells` ranks by TRIANGLE COUNT, "everything but shell 0"
    # meant the two plates on the right but the two plates PLUS half a
    # barrel on the left: the robot's left shoulder wore twice the bronze of
    # its right, and from the front only the left showed any at all, because
    # both of the real plates sit aft of centre.
    #
    # So select the PLATES, which is what the photographs show. The two
    # links are mirrors in y to a median 0.14 mm, and |x| > 37.5 mm lands on
    # the barrel's own rim. MEASURED, left then right: 30 284 and 30 285 mm^2
    # over 4 islands each, and 0.0 % of the seam through smooth surface on
    # BOTH sides. The rim is narrow and the threshold is not free — 37.0
    # scores 84 % smooth and 38.5 scores 23 %.
    *[(f"{side}_shoulder_pitch_link", f"asimov_shoulder_pad_{side[0]}",
       lambda c, s, n: np.abs(c[:, 0]) > 0.0375, "ACCENT", "SHELL")
      for side in ("left", "right")],
    *[(f"{side}_elbow_link", f"asimov_elbow_band_{side[0]}",
       lambda c, s, n: s != 0, "ACCENT", "SHELL") for side in ("left", "right")],
)


def _mesh_file(link: str, dest: Path | None = None) -> Path:
    """The STL on disk behind `link`'s visual mesh.

    Resolved through the MJCF's own asset entry and `meshdir` rather than
    guessed from the link name: Asimov's mesh asset is `waist_yaw_link.STL`
    and the file it names is `WAIST_YAW.STL`, so the two are not the same
    string and a guess would miss.
    """
    root = Path(dest) if dest else asimov.cache_dir()
    xml = asimov.xml_path(root)
    spec = mujoco.MjSpec.from_file(str(xml))
    meshdir = getattr(spec, "meshdir", "") or ""
    base = xml.parent if os.path.isabs(meshdir) else xml.parent
    for mesh in spec.meshes:
        if mesh.name == f"{link}.STL":
            return (Path(meshdir) if os.path.isabs(meshdir)
                    else (base / meshdir)).resolve() / mesh.file
    raise KeyError(f"{asimov.REPO} has no mesh asset for {link!r}")


def _read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """`(normals[n, 3], verts[n, 3, 3])` from a binary STL.

    The count in the 84-byte header is checked against the file length,
    which is the only honest way to tell a binary STL from an ASCII one and
    is also the corruption check. Same reader as `mars._read_binary_stl`.
    """
    blob = path.read_bytes()
    if len(blob) < 84:
        raise ValueError(f"{path} is {len(blob)} bytes — not an STL")
    count = int(struct.unpack("<I", blob[80:84])[0])
    if len(blob) != 84 + count * 50:
        raise ValueError(
            f"{path} is not a binary STL of {count} triangles: {len(blob)} "
            f"bytes, expected {84 + count * 50}")
    rec = np.frombuffer(blob, dtype=np.uint8, offset=84).reshape(count, 50)
    return (rec[:, 0:12].copy().view("<f4").reshape(count, 3),
            rec[:, 12:48].copy().view("<f4").reshape(count, 3, 3))


def _triangle_shells(verts: np.ndarray) -> np.ndarray:
    """One label per triangle: which connected SHELL of the mesh it belongs
    to, numbered 0 upward by DESCENDING triangle count — so shell 0 is always
    the biggest piece and a predicate can say `shells != 0` without knowing
    the mesh.

    Vertices are welded at 1e-5 m first: an STL is a triangle soup with every
    vertex repeated per face, so without the weld every triangle is its own
    component. Same approach as `mars._triangle_shells`, which cuts MARS's
    tyres out of its base.

    This is the cut to reach for when a part is NOT plane-bounded but IS a
    separate piece of geometry. MEASURED on Asimov's head: 5 shells — a
    7 256-triangle outer rim, a 6 310-triangle inner panel that fills the
    face and the crown (the visor), two side plates of ~1 500, and a 152
    scrap. The visor is unreachable by any plane and falls straight out of
    this.
    """
    key = np.round(verts.reshape(-1, 3), 5)
    _uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    faces = np.asarray(inverse).reshape(len(verts), 3)
    parent = np.arange(int(faces.max()) + 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for tri in faces:
        root = find(int(tri[0]))
        for other in tri[1:]:
            r2 = find(int(other))
            if r2 != root:
                parent[r2] = root
    roots = np.array([find(int(f[0])) for f in faces])
    order = [r for r, _n in sorted(
        ((r, int((roots == r).sum())) for r in np.unique(roots)),
        key=lambda kv: -kv[1])]
    rank = {r: i for i, r in enumerate(order)}
    return np.array([rank[r] for r in roots], dtype=np.int32)


# The smallest a piece of a cut can be and still be a PART and not a speck,
# in mm^2. MEASURED across every cut on this robot: the island areas run
# ... 72, 90, 109, 109, 130, 130 | 516, 516, 1187, 1188, 3014, 3817 ... —
# a 4.0x gap with nothing inside it, and 250 sits in the middle of it (the
# geometric mean of the two sides is 259). Below the gap are only the flecks
# a face-normal test strews across a curved shell; above it is every real
# plate, panel and sole layer on the machine.
SPECK_MM2 = 250.0


def _edge_pairs(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of triangles sharing an edge, welded at 1e-5 m, as two
    index arrays. The same weld `_triangle_shells` uses, and for the same
    reason: an STL is a soup with every vertex repeated per face.

    EVERY pair within an edge's group, not just the consecutive ones. These
    meshes are not manifold — the shoe alone has 2 136 edges shared by four
    triangles and five shared by six — and chaining only neighbours in the
    sorted order links two faces on such an edge solely when they happen to
    sort next to each other. A caller that then keeps pairs whose BOTH ends
    are selected loses the rest, which is how a first version of this split
    the sole into 106 islands instead of 17 and dropped 2 067 mm^2 of it as
    specks. Groups are at most six, so all of C(k, 2) is a few thousand rows.
    """
    key = np.round(verts.reshape(-1, 3), 5)
    _uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    faces = np.asarray(inverse).reshape(len(verts), 3)
    edge = np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edge[:, 1], edge[:, 0]))
    edge, owner = edge[order], owner[order]
    if not len(edge):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    start = np.flatnonzero(np.concatenate(
        ([True], np.any(edge[1:] != edge[:-1], axis=1))))
    size = np.diff(np.append(start, len(edge)))
    left: list = []
    right: list = []
    for far in range(1, int(size.max())):
        group = start[size > far]
        for near in range(far):
            left.append(owner[group + near])
            right.append(owner[group + far])
    if not left:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(left), np.concatenate(right)


def _islands(count: int, mask: np.ndarray, left: np.ndarray,
             right: np.ndarray) -> np.ndarray:
    """One label per triangle: which connected island of `mask` it lies in,
    or -1 when it is outside `mask`. Islands are numbered arbitrarily.

    This is `_triangle_shells` restricted to a SUBSET — the shells of the
    whole mesh say nothing about whether a selection drawn across them is
    one piece, which is exactly the thing that went wrong on the chest.
    """
    parent = np.arange(count)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    both = mask[left] & mask[right]
    for a, b in zip(left[both].tolist(), right[both].tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    label = np.full(count, -1, dtype=np.int64)
    inside = np.flatnonzero(mask)
    if len(inside):
        roots = np.array([find(int(i)) for i in inside])
        _uniq, idx = np.unique(roots, return_inverse=True)
        label[inside] = idx
    return label


def _tidy_selection(verts: np.ndarray, picked: np.ndarray,
                    floor_mm2: float = SPECK_MM2) -> np.ndarray:
    """Drop the specks a cut strews, then fill the holes it leaves.

    A face-normal test over a CURVED shell does not select one patch. It
    selects the patch plus every stray triangle whose normal happens to tip
    past the threshold, and it punches pinholes wherever the surface dips
    back. MEASURED on the chest before this ran: 39 islands of which 36 were
    specks (1 107 mm^2 between them, some of them at shoulder height, which
    is what put red flecks around the shoulders), 38 lone black triangles
    stranded inside the red and 18 lone red ones stranded in the black, and
    a seam 5.2x longer than a clean outline of the same area would be.

    The two passes are deliberately NOT symmetric:

    * a picked island under `floor_mm2` is dropped wherever it is — a speck
      of paint on the far side of the robot is wrong whether or not anything
      encloses it;
    * an unpicked island under `floor_mm2` is filled only when it shares an
      edge with painted surface, i.e. it is a hole IN the paint. Without
      that test this would also swallow the small internal fasteners that
      live inside these shells (the shoe alone has 57) and have nothing to
      do with the cut — they touch nothing and must stay as they are.
    """
    area = np.linalg.norm(np.cross(verts[:, 1] - verts[:, 0],
                                   verts[:, 2] - verts[:, 0]), axis=1) / 2
    floor = floor_mm2 * 1e-6
    left, right = _edge_pairs(verts)
    out = np.array(picked, dtype=bool)

    label = _islands(len(verts), out, left, right)
    live = label >= 0
    if live.any():
        areas = np.bincount(label[live], weights=area[live])
        out[live] = ~(areas < floor)[label[live]]

    label = _islands(len(verts), ~out, left, right)
    hole = label >= 0
    if hole.any():
        areas = np.bincount(label[hole], weights=area[hole])
        touches = np.zeros(len(areas), dtype=bool)
        for near, far in ((left, right), (right, left)):
            edged = (label[near] >= 0) & (label[far] < 0)
            touches[label[near][edged]] = True
        out[hole] |= ((areas < floor) & touches)[label[hole]]
    return out


def _write_stl(path: Path, normals: np.ndarray, verts: np.ndarray) -> None:
    """One binary STL, written atomically — a worker spawned mid-write must
    never import a truncated mesh (`AGENTS.md`, "Atomic writes and live
    imports")."""
    count = len(verts)
    rec = np.zeros((count, 50), dtype=np.uint8)
    rec[:, 0:12] = np.ascontiguousarray(normals, "<f4").view(np.uint8)
    rec[:, 12:48] = (np.ascontiguousarray(verts, "<f4")
                     .view(np.uint8).reshape(count, 36))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(b"microduck derived".ljust(80, b" ")
                    + struct.pack("<I", count) + rec.tobytes())
    os.replace(tmp, path)


def cut_part(link: str, name: str, keep, dest: Path | None = None
             ) -> tuple[Path, Path]:
    """Cut `link`'s source STL into (kept, rest). Cached on the source's mtime.

    A triangle goes to the kept half when its CENTROID satisfies `keep`, so
    the seam is one triangle wide and no vertex is invented. Both halves stay
    in the file's frame, which is what makes them land where the whole mesh
    did — see `CUT_PARTS` for what happens otherwise.
    """
    root = Path(dest) if dest else asimov.cache_dir()
    source = _mesh_file(link, dest)
    a, b = root / f"{name}.stl", root / f"{name}_rest.stl"
    # Fresh against the SOURCE *and* against this module, because the cut
    # predicate lives here: keying on the mesh alone means editing a
    # threshold silently reuses the old halves, which is a change that
    # appears not to apply and sends you looking in the wrong place.
    newest = max(source.stat().st_mtime, Path(__file__).stat().st_mtime)
    if (a.is_file() and b.is_file()
            and min(a.stat().st_mtime, b.stat().st_mtime) >= newest):
        return a, b
    normals, verts = _read_stl(source)
    unit = np.asarray(normals, dtype=np.float64)
    length = np.linalg.norm(unit, axis=1, keepdims=True)
    length[length == 0] = 1.0
    picked = np.asarray(
        keep(verts.mean(1), _triangle_shells(verts), unit / length), dtype=bool)
    if not picked.any() or picked.all():
        raise ValueError(
            f"the cut for {name!r} selected {picked.sum()} of {len(verts)} "
            "triangles — a plane that misses the mesh is a threshold in the "
            "wrong frame, not a colour choice")
    # Every cut goes through this, not just the ones known to need it: a
    # predicate that is one clean patch on one link is confetti on the next,
    # and the chest passed the seam test for weeks while wearing 36 specks.
    picked = _tidy_selection(verts, picked)
    if not picked.any() or picked.all():
        raise ValueError(
            f"the cut for {name!r} had nothing left after specks under "
            f"{SPECK_MM2:.0f} mm^2 were dropped — the predicate selected "
            "scatter, not a part")
    _write_stl(a, normals[picked], verts[picked])
    _write_stl(b, normals[~picked], verts[~picked])
    return a, b


def add_cut_parts(spec: mujoco.MjSpec) -> None:
    """Apply every `CUT_PARTS` entry: one visual geom becomes two.

    No position compensation and none needed — both halves are in the source
    file's frame, so MuJoCo places them exactly as it placed the whole mesh.
    MEASURED at 4e-6 mm, and `tests/test_asimov_look` re-derives every visual
    body's world bounding box against the uncut model to hold it there.

    VISUAL ONLY: `contype`/`conaffinity`/`density` all 0, and the collision
    capsules are a separate set this never touches. The same test file
    measures that a 1500-step rollout stays BIT-identical.
    """
    colours = {"RED": ASIMOV_RED, "SHELL": ASIMOV_SHELL, "ACCENT": ASIMOV_ACCENT,
               "SOLE": ASIMOV_SOLE, "FOOT": ASIMOV_FOOT}
    by_name = {b.name: b for b in spec.bodies}
    for link, name, keep, cut_colour, rest_colour in CUT_PARTS:
        body = by_name.get(link)
        if body is None:
            continue
        cut, rest = cut_part(link, name, keep)
        for geom in list(body.geoms):
            if geom.contype == 0 and geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(geom)
        for mesh_name, path, key in ((name, cut, cut_colour),
                                     (f"{name}_rest", rest, rest_colour)):
            mesh = spec.add_mesh()
            mesh.name = mesh_name
            mesh.file = str(path)
            # A cut can leave a piece too thin for a volume integral — a
            # 9 mm shoulder pad does — and MuJoCo then refuses to compile
            # with "mesh volume is too small". These carry no mass anyway.
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            geom = body.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh_name
            geom.group, geom.contype, geom.conaffinity, geom.density = 2, 0, 0, 0
            geom.material = ""
            geom.rgba = list(colours[key])


def paint_shell(spec: mujoco.MjSpec) -> None:
    """Asimov's colours, ON THE SPEC rather than on the viewer's dump.

    The MJCF paints all 26 visual geoms with ONE material — `visual_grey`,
    rgba 0.7 0.7 0.7, on a defaults class — so out of the box he is 26
    identically flat grey parts, which reads as a mannequin rather than a
    machine. `duck-viewer/components/AsimovLook.tsx` already had a table for
    the `/sim` page; this is the same split in the place EVERY consumer
    compiles, which is the lesson `robots/mars.paint_shell` records: a look
    only the dump can see is why every MuJoCo sheet of a MARS run came out
    the URDF's uniform grey while the lab stage showed a painted robot.
    `probe_mjcf_body.py`, `record-world` and the lab stage's generic paint
    all read this.

    **The colours are read off Menlo's published RENDERS, not off published
    materials** — the model states one grey and nothing in it can say what
    the machine is painted. If they ever ship per-link materials, this table
    and `AsimovLook.tsx` are what those replace.

    `geom.material` is cleared and not only `geom.rgba` set: a material wins
    over a geom's own rgba both in MuJoCo's renderer and in the mesh dump
    (`g1.extract_visual_scene` reads `mat_rgba` whenever `matid >= 0`), so
    painting rgba alone would change nothing anywhere and look like the
    change had not been applied.
    """
    for body in spec.bodies:
        for geom in body.geoms:
            # `contype == 0` IS the definition of a geom that exists to be
            # looked at (`mjcf_body.visual_geom_group` detects the visual
            # group the same way), so this needs no group number and cannot
            # go stale against one.
            if geom.contype or geom.type != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            name = body.name.lower()
            if any(f in name for f in FOOT_LINKS):
                rgba = ASIMOV_FOOT
            elif any(f in name for f in RED_LINKS):
                rgba = ASIMOV_RED
            elif any(f in name for f in HAND_LINKS):
                rgba = ASIMOV_HAND
            elif any(f in name for f in SILVER_LINKS):
                rgba = ASIMOV_SILVER
            elif any(f in name for f in ACCENT_LINKS):
                rgba = ASIMOV_ACCENT
            else:
                rgba = ASIMOV_SHELL
            geom.material = ""
            geom.rgba = list(rgba)


def _painted_spec() -> mujoco.MjSpec:
    """The robot MJCF with its meshes resolvable and its colours on.

    The ROBOT alone — no servos, no keyframe, no floor stripped — which is
    what the two readers that only want geometry need (`robot_model`, the
    viewer dump). `walk_spec` does the same load and then adds the rest.
    """
    asimov.require_asimov()
    spec = mujoco.MjSpec.from_file(str(asimov.xml_path()))
    base = asimov.xml_path().parent
    for attr in ("meshdir", "texturedir", "assetdir"):
        rel = getattr(spec, attr, "") or ""
        if rel and not os.path.isabs(rel):
            setattr(spec, attr, str((base / rel).resolve()))
    paint_shell(spec)
    add_cut_parts(spec)
    return spec


# ------------------------------------------------------------- the model

def walk_spec(standalone: bool = True, joint_friction: bool = True
              ) -> mujoco.MjSpec:
    """The MJCF with servos, armature and a STAND keyframe — ready to act on.

    The shipped model has **`nu = 0`**: no actuators at all, because cyclotron
    adds them in Python from the `env.yaml` block mirrored above. So does
    this, and it is the one place the two stacks can silently disagree — a
    policy trained against kp 150 and driven at MuJoCo's default would flail
    and the only symptom is a robot that falls over.

    `standalone` keeps the floor, the lights and the skybox the file carries,
    which is what a single-robot check wants. False strips all three, because
    attaching a model that brings its own ground into a shared `/sim` room
    gives that room two floors (`robots/mjcf_body._generated_scene` hit the
    same thing from the other side).

    `joint_friction` is the ONE quantity here whose translation is an
    interpretation rather than a copy: Isaac's `DelayedPDActuatorCfg.friction`
    is PhysX joint friction and MuJoCo's nearest equivalent is
    `frictionloss` in N*m. It is applied by default and exposed so the walk
    probe can measure both, rather than asserted to be right.
    """
    asimov.require_asimov()
    spec = mujoco.MjSpec.from_file(str(asimov.xml_path()))
    base = asimov.xml_path().parent
    for attr in ("meshdir", "texturedir", "assetdir"):
        rel = getattr(spec, attr, "") or ""
        if rel and not os.path.isabs(rel):
            setattr(spec, attr, str((base / rel).resolve()))

    paint_shell(spec)
    add_cut_parts(spec)

    if not standalone:
        for g in list(spec.worldbody.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
                spec.delete(g)
        for light in list(spec.worldbody.lights):
            spec.delete(light)

    by_name = {j.name: j for j in spec.joints}
    missing = [n for n in JOINT_NAMES if n not in by_name]
    if missing:
        raise RuntimeError(
            f"{asimov.REPO} at {asimov.ASIMOV_SHA[:12]} has no joint(s) "
            f"{missing} — the 23-joint action order this policy speaks does "
            "not fit this model")

    for name in JOINT_NAMES:
        kp, kd, effort, armature, friction = gains_for(name)
        j = by_name[name]
        j.armature = armature
        if joint_friction:
            j.frictionloss = friction
        # A MuJoCo position servo: force = kp*(target - q) - kd*qdot, clamped
        # at the motor's peak torque. `biasprm` is (0, -kp, -kd) by MuJoCo's
        # own definition of an affine bias, which is what makes gainprm[0]
        # and biasprm[1] cancel into a position error term.
        a = spec.add_actuator()
        a.name = name
        a.target = name
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[1] = -kp
        a.biasprm[2] = -kd
        a.forcerange[0], a.forcerange[1] = -effort, effort

    key = spec.add_key()
    key.name = STAND_KEY
    probe = mujoco.MjSpec.from_file(str(asimov.xml_path())).compile()
    qpos = np.array(probe.qpos0, np.float64).copy()
    qpos[0:3] = [0.0, 0.0, STAND_Z]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name in JOINT_NAMES:
        adr = int(probe.joint(name).qposadr[0])
        qpos[adr] = DEFAULT_POSE_BY_NAME[name]
    key.qpos = qpos
    key.qvel = np.zeros(probe.nv)
    key.ctrl = np.array([DEFAULT_POSE_BY_NAME[n] for n in JOINT_NAMES],
                        np.float64)
    return spec


@lru_cache(maxsize=2)
def walk_model(standalone: bool = True,
               joint_friction: bool = True) -> mujoco.MjModel:
    return walk_spec(standalone, joint_friction).compile()


def joint_ids(model: mujoco.MjModel, prefix: str = ""
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(qpos addresses, qvel addresses, actuator ids)` in `JOINT_NAMES` order.

    By NAME, never by index: the contract's order is Isaac's articulation
    order and this model's is the MJCF tree's. They agree today — asserted
    here — and a revision that reorders a limb would otherwise drive the
    knees with the elbows' targets and change nothing a test could see.
    """
    qpos = np.array([int(model.joint(prefix + n).qposadr[0])
                     for n in JOINT_NAMES])
    qvel = np.array([int(model.joint(prefix + n).dofadr[0])
                     for n in JOINT_NAMES])
    act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      prefix + n) for n in JOINT_NAMES])
    if np.any(act < 0):
        raise KeyError(
            f"model has no actuator for {[n for n, a in zip(JOINT_NAMES, act) if a < 0]}"
            " — build it with walk_spec(), which injects them")
    return qpos, qvel, act


def slot_order() -> tuple[int, ...]:
    """The 23 joint indices in observation order: slot01, then 23, then 45."""
    index = {n: i for i, n in enumerate(JOINT_NAMES)}
    return tuple(index[n] for slot in SLOTS for n in slot)


# -------------------------------------------------------------- the driver

@lru_cache(maxsize=2)
def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


class AsimovWalker:
    """50 Hz velocity-commanded walk on one Asimov, shipped policy or ours.

    The same shape as `robots/g1.G1Walker` so the lab and `/sim` can drive
    either: `spawn`, `step`, `pose`, `fallen`.
    """

    def __init__(self, model: mujoco.MjModel, prefix: str = "",
                 policy: str | Path | None = None,
                 *, obs_noise: bool = False, action_delay: int = 0,
                 seed: int = 0):
        self.model = model
        self.prefix = prefix
        self.qpos_adr, self.qvel_adr, self.act_id = joint_ids(model, prefix)
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                 prefix + "floating_base")
        if root < 0:
            raise KeyError(f"no {prefix}floating_base free joint in the model")
        self.root_qpos = int(model.jnt_qposadr[root])
        self.root_qvel = int(model.jnt_dofadr[root])
        self.pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        prefix + "pelvis_link")
        if self.pelvis < 0:
            raise KeyError(f"no body {prefix + 'pelvis_link'!r} in the model")
        #: What a `world/arena.WorldPerson` tracks this walker by — the half
        #: of `G1Walker`'s interface that is not a method.
        self.base_body_id = self.pelvis
        #: Every body of THIS robot in the composed model. The arena uses it
        #: to own geoms for the bump sense (`World._sense_bumps`), so a duck
        #: that walks into a shin knows what it hit.
        self.body_ids = [b for b in range(model.nbody)
                         if model.body(b).name.startswith(prefix)] if prefix \
            else [b for b in range(model.nbody) if b]
        #: The order the viewer's mesh dump is streamed in: world first, then
        #: this robot's tree with the prefix stripped — `G1Walker`'s shape,
        #: because the viewer draws both the same way.
        self.scene_bodies = ["world"] + [
            model.body(b).name[len(prefix):] for b in self.body_ids]
        self.slot = np.array(slot_order())
        self.last_action = np.zeros(NUM_JOINTS, np.float32)
        self.cmd = np.zeros(3, np.float32)
        self._queue: list[np.ndarray] = []
        self.action_delay = int(action_delay)
        self.obs_noise = bool(obs_noise)
        self._rng = np.random.default_rng(seed)
        self._sess = _session(str(policy or require_policy() / "policy.onnx"))
        self._in = self._sess.get_inputs()[0].name
        want = self._sess.get_inputs()[0].shape[-1]
        if int(want) != OBS_DIM:
            raise ValueError(
                f"{policy} takes {want} observations, this contract writes "
                f"{OBS_DIM} — they are not the same policy interface")

    # ------------------------------------------------------------- state
    def spawn(self, data: mujoco.MjData, x: float = 0.0, y: float = 0.0,
              yaw: float = 0.0) -> None:
        q = self.root_qpos
        data.qpos[q:q + 3] = [x, y, STAND_Z]
        data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        data.qpos[self.qpos_adr] = DEFAULT_POSE
        data.qvel[self.root_qvel:self.root_qvel + 6] = 0.0
        data.qvel[self.qvel_adr] = 0.0
        data.ctrl[self.act_id] = DEFAULT_POSE
        self.last_action[:] = 0.0
        self._queue.clear()

    def pose(self, data: mujoco.MjData) -> tuple[float, float, float]:
        p = data.xpos[self.pelvis]
        w, x, y, z = data.xquat[self.pelvis]
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return float(p[0]), float(p[1]), yaw

    def projected_gravity(self, data: mujoco.MjData) -> np.ndarray:
        """Gravity's unit vector in the pelvis frame: `R^T (0,0,-1)`, which is
        minus the third ROW of the body's rotation matrix. Upright reads
        (0, 0, -1) and `[2]` rising past -0.342 is the 70 deg fall cut."""
        rot = data.xmat[self.pelvis].reshape(3, 3)
        return (-rot[2, :]).astype(np.float32)

    def fallen(self, data: mujoco.MjData) -> bool:
        return bool(self.projected_gravity(data)[2] > FALL_GRAVITY_Z)

    def bodies_payload(self, data: mujoco.MjData,
                       scene_bodies: list[str]) -> list[list[float]]:
        """Pose per body, in the order the viewer asked for — `G1Walker`'s
        payload, to the rounding. A name this robot does not have streams the
        identity pose rather than raising: the viewer asks for whatever its
        cached mesh dump holds, which can be a frame behind a reload."""
        out = []
        for name in scene_bodies:
            if name in ("", "world"):
                out.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                continue
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                  self.prefix + name)
            if b < 0:
                out.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                continue
            pos, quat = data.xpos[b], data.xquat[b]
            out.append([round(float(v), 4) for v in (*pos, *quat)])
        return out

    # --------------------------------------------------------------- obs
    def obs(self, data: mujoco.MjData) -> np.ndarray:
        # A free joint's angular qvel is already in the BODY frame, which is
        # what an IMU on the pelvis reads.
        ang_vel = data.qvel[self.root_qvel + 3:self.root_qvel + 6]
        gravity = self.projected_gravity(data)
        jpos = data.qpos[self.qpos_adr] - DEFAULT_POSE
        jvel = data.qvel[self.qvel_adr]
        if self.obs_noise:
            r = self._rng
            ang_vel = ang_vel + r.uniform(-NOISE_ANG_VEL, NOISE_ANG_VEL, 3)
            gravity = gravity + r.uniform(-NOISE_GRAVITY, NOISE_GRAVITY, 3)
            jpos = jpos + r.uniform(-NOISE_JOINT_POS, NOISE_JOINT_POS, NUM_JOINTS)
            jvel = jvel + r.uniform(-NOISE_JOINT_VEL, NOISE_JOINT_VEL, NUM_JOINTS)
        out = np.empty(OBS_DIM, np.float32)
        out[slice(*OBS_ANG_VEL)] = np.asarray(ang_vel) * ANG_VEL_SCALE
        out[slice(*OBS_GRAVITY)] = gravity
        out[slice(*OBS_COMMAND)] = self.cmd
        out[slice(*OBS_JOINT_POS)] = np.asarray(jpos)[self.slot]
        out[slice(*OBS_JOINT_VEL)] = np.asarray(jvel)[self.slot] * JOINT_VEL_SCALE
        out[slice(*OBS_LAST_ACTION)] = self.last_action
        return out

    # -------------------------------------------------------------- step
    def act(self, data: mujoco.MjData) -> np.ndarray:
        """One policy tick: read the state, run the graph, write `data.ctrl`.

        Call once per CONTROL step and `mj_step` `DECIMATION` times after it.
        """
        obs = self.obs(data)[None, :]
        action = np.asarray(
            self._sess.run(None, {self._in: obs})[0], np.float32).reshape(-1)
        self.last_action = action.copy()
        self._queue.append(action)
        # The 0-5 step lag env.yaml randomises per actuator group. Off by
        # default; a fixed lag here is the deterministic version of it.
        if len(self._queue) > self.action_delay:
            applied = self._queue.pop(0)
        else:
            applied = np.zeros(NUM_JOINTS, np.float32)
        data.ctrl[self.act_id] = DEFAULT_POSE + ACTION_SCALE * applied
        return action

    #: `G1Walker`'s name for `act`, and the one `world/arena.WorldPerson`
    #: calls: write `ctrl` and do NOT step. The world owns the clock — it
    #: runs `substeps = CTRL_DT / model.opt.timestep` physics steps per tick,
    #: read off the ROOM's timestep rather than this contract's decimation,
    #: so a room on a 2 ms clock still ticks this policy at 50 Hz.
    control = act

    def step(self, data: mujoco.MjData) -> np.ndarray:
        """`control` plus this contract's own `DECIMATION` steps — for a
        standalone rollout, where nothing else owns the clock."""
        action = self.act(data)
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, data)
        return action


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true",
                    help="download the shipped policy from the Hub")
    args = ap.parse_args(argv)
    if args.fetch:
        fetch_policy()
        return 0
    print(f"{POLICY_REPO} @ {POLICY_SHA[:12]} ({POLICY_LICENCE}): "
          f"{'ready' if policy_ready() else 'not fetched'} at {policy_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


# ------------------------------------------------------- level 1: the spec

#: Four 5 mm capsules a foot, and MEASURED flat: at `STAND` all four
#: undersides sit at exactly the same height (spread 0.00000 m), unlike the
#: duck's filleted sole where a tolerance has to pick the flat out.
FOOT_GEOMS: dict[str, tuple[str, ...]] = {
    "left": tuple(f"left_foot{i}_collision" for i in range(1, 5)),
    "right": tuple(f"right_foot{i}_collision" for i in range(1, 5)),
}

#: The sole is flat to 0.0 mm, so any tolerance picks the whole footprint;
#: the duck's 1 mm is kept rather than inventing a smaller number.
SOLE_TOL = 0.001
#: MEASURED under the shipped policy at 0.5 m/s over 20 s: the two soles'
#: height difference runs median 0.0103 m, p90 0.0242, max 0.0276. A 1 cm cut
#: therefore splits the gait near its own median — one foot airborne about
#: half the time, which is what a walk is. The G1's 1 cm on a 1.3 m body
#: agrees, and that agreement is a check rather than the reason.
GROUND_TOL = 0.01
#: MEASURED: limp from STAND, the pelvis settles at 0.110 m against 0.639
#: standing. 0.25 m is clear of both — below any crouch this robot holds,
#: well above lying down. The real cut is the 70 deg tilt above.
FALL_HEIGHT = 0.25


def walk_scene_xml(dest: Path | None = None, joint_friction: bool = True) -> Path:
    """Write (if changed) and return the walk scene: robot + servos + STAND.

    Atomic, and rewritten only when the content differs — `g1.g1_scene_xml`'s
    pattern, for the reason `AGENTS.md` gives under "Atomic writes and live
    imports": a parallel training worker must never import a half-written
    scene.
    """
    out = (dest or asimov.cache_dir()) / (
        "scene_walk_asimov.xml" if joint_friction
        else "scene_walk_asimov_nofric.xml")
    xml = walk_spec(True, joint_friction).to_xml()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.read_text() != xml:
        tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
        tmp.write_text(xml)
        os.replace(tmp, out)
    return out


class AsimovBody(_SPEC_BASE):
    """Asimov 1 at level 1 — `spec()` builds the one instance."""

    def contract(self):
        """Their DEPLOYMENT contract, and the honesty is in the sentence.

        Unlike the G1's lab contract, every one of these 78 floats is
        available on the real robot — Menlo's own model card says so, and the
        absence of base LINEAR velocity (which the G1's layout opens with) is
        the design decision their blog describes making deliberately. So this
        is a deployment contract in SHAPE.

        It is not a deployment claim by this repo. Nothing here has run on an
        Asimov; the sim2real evidence is Menlo's, measured on their stack, and
        the `deploy` sentence says exactly that so it travels with any `.onnx`
        exported from here.
        """
        from .policy_contract import declare

        return declare(
            self, id=CONTRACT_ID, rate_hz=round(1.0 / CTRL_DT, 6),
            slots=obs_slots(),
            deploy="Menlo's own 78-obs deployment layout (all of it is "
                   "observable on hardware); NOTHING here has run on an "
                   "Asimov, and the sim2real evidence is theirs, not ours")

    def ready(self) -> bool:
        return asimov.ready() and policy_ready()

    def fetch(self):
        asimov.fetch()
        return fetch_policy()

    def setup_hint(self) -> str:
        return asimov.setup_hint()

    #: What this body can be asked to learn. `train.py` prints these when a
    #: `--task` it does not have is asked for.
    TASKS = ("walk", "getup", "getup_ref", "getup_clip")

    def train_env_kwargs(self, args) -> dict:
        """The get-up curriculum's knobs, recorded ONTO the run.

        These used to be read inside `AsimovGetupEnv.__init__` straight from
        `os.environ`, and `robots/moss.py` already carries the rule that
        breaks: *"an env that reads its own environment trains fine and leaves
        no record of what it trained on."* It cost the first four get-up runs.
        `run.json` said only `{domain_rand, obs_noise}`, so nothing recorded
        which rung each of the four curriculum stages trained at, and every
        later render read its blend and assist out of the ambient shell
        instead — which is why no claim measured against those runs can be
        checked, including the one that sent the next lever off after a donor.

        Read HERE, in the trainer, so the knob travels into `env_kwargs` and
        the run says what it was. `train._seed_command_stats` and the lab's
        preview both route through this method already.
        """
        task = getattr(args, "task", None) or self.default_task
        if task == "getup_clip":
            from .asimov_env import AsimovGetupClipEnv as CE
            from .asimov_env import AsimovGetupEnv as E
            from .asimov_env import AsimovGetupRefEnv as RE
            # The IMITATION task records which demonstration it tracked and how
            # much drift it was allowed. Without this the run recorded
            # `track_tol: None` while the stage knob was plainly in effect at
            # 0.80 — the run that cannot say what it trained under is the bug
            # that cost four runs on 2026-10-04, and the clip PATH matters even
            # more than a rung does: two runs against different harvests are not
            # comparable, and the harvest names its upstream sha.
            return {
                "clip_file": (os.environ.get(CE.CLIP_ENV)
                              or str(asimov.CACHE_DIR / "getup_clips.npz")),
                "track_tol": float(os.environ.get(RE.TOL_ENV, "0.60") or 0.60),
                "force_lag": bool(int(os.environ.get(RE.LAG_ENV, "0") or 0)),
                "history": int(os.environ.get(RE.HISTORY_ENV, "1") or 1),
                # THE ACTION CLIP, on the run because this task raises it off
                # the body's duck-inherited 4.0 and a policy trained at 4.0 and
                # one trained at 13.0 are not the same controller: at 4.0,
                # 92-98 % of the teacher's DAgger corrections are unexecutable.
                # READ FROM THE ENVIRONMENT, not pinned: a constant here is
                # PASSED TO THE CTOR, so it overrode the stage's own
                # `ACTION_CLIP` and the knob was dead on this task. It happened
                # to equal the teacher's value in every recipe that set it, so
                # nothing was measured wrong — but the next value would have
                # been silently ignored. See "One knob, three clip sites".
                "action_clip": float(os.environ.get(E.CLIP_ENV, "")
                                     or CE.TEACHER_ACTION_CLIP),
                # WHICH RUNG the episodes start on, and the SPAN CUTS. All four
                # were read straight from the environment by the ctor and
                # trained with while `run.json` said nothing — the FIFTH time
                # this exact bug has been found on this body. It is not
                # cosmetic: `teach-asimov_getup_tidy-f4b58f` trained under a
                # 0.45 m span cut that ended its episodes after 28 steps, and
                # its `run.json` recorded neither the cut nor the window, so
                # every later evaluation rebuilt a different task and scored it.
                "start_rung": os.environ.get(E.START_RUNG_ENV, "") or "",
                "max_span": float(os.environ.get(E.SPAN_ENV, "") or 0.0),
                "launch_span": float(os.environ.get(E.LAUNCH_SPAN_ENV, "")
                                     or 0.0),
                "fall_depth": float(os.environ.get(E.FALL_DEPTH_ENV, "")
                                    or 1.0),
                # The CONTINUOUS span charge and the width it is measured
                # from. On the run because a run that TERMINATES on a wide
                # stance and one that is CHARGED for it are not two seeds of
                # one experiment — the cut taught the policy to stop standing
                # altogether (`…-rise-126f42-s2`: up 0.0 from 98 k steps,
                # return 35.5 against a do-nothing floor of 39.8).
                # THE ASSIST FORCE, as a fraction of body weight held up at
                # the pelvis. It changes the DYNAMICS, so a run at 0.85 and
                # one at 0.00 are different robots — and on this task it is
                # the whole curriculum: 0.85 is the only setting measured to
                # cross the pike->fold gap (fold reached 12/12 at a 0.355 m
                # span) and 1.00 launches the robot to 1.8 m.
                "getup_assist": float(os.environ.get(E.ASSIST_ENV, "0.0")
                                      or 0.0),
                # THE SCRIPTED PUSH-UP. A run whose episodes begin with
                # 1.56 s of servo-driven prone->pike and one that starts at a
                # placed pike are not the same experiment: the policy inherits
                # a pike it ARRIVED at, with real velocity.
                "prologue": os.environ.get(E.PROLOGUE_ENV, "") or "",
                # GET THE FEET UNDER THE MASS — see `AsimovGetupEnv.TUCK_ENV`.
                "tuck_pay": float(os.environ.get(E.TUCK_ENV, "") or 0.0),
                # ROTATION pay. The get-up's blocker is rotation, not
                # height, and `progress` pays only height — so a run with this
                # on and one without optimise different things entirely.
                "rotate_pay": float(os.environ.get(E.ROTATE_ENV, "") or 0.0),
                # ROLL ONTO THE BACK — recorded on every branch, because a
                # knob a run SETS and does not RECORD is the bug this file
                # has now had five times.
                "roll_pay": float(os.environ.get(E.ROLL_ENV, "") or 0.0),
                "span_cost": float(os.environ.get(E.SPAN_COST_ENV, "")
                                   or 0.0),
                "span_target": float(os.environ.get(E.SPAN_TARGET_ENV, "")
                                     or 0.40),
                # THE FLIGHT CHARGE. Recorded on every branch, because a
                # knob a run SETS and does not RECORD is the bug this file
                # has now had five times.
                "ground_cost": float(os.environ.get(E.GROUND_ENV, "")
                                     or 0.0),
                "balance_cost": float(os.environ.get(E.BALANCE_ENV, "")
                                      or 0.0),
                "home_pay": float(os.environ.get(E.HOME_ENV, "") or 0.0),
                "limit_cost": float(os.environ.get(E.LIMIT_ENV, "") or 0.0),
                "push_bw": float(os.environ.get(E.PUSH_ENV, "") or 0.0),
                "calm_cost": float(os.environ.get(E.CALM_ENV, "") or 0.0),
                "swing_pay": float(os.environ.get(E.SWING_ENV, "") or 0.0),
                "handoff_pay": float(os.environ.get(E.HANDOFF_ENV, "")
                                     or 0.0),
                # FREE mode: the intermediates are not imitated, only the rise
                # is paid. A free run and a tracking one optimise different
                # objectives and are not two seeds of one experiment.
                "free": bool(int(os.environ.get(RE.FREE_ENV, "0") or 0)),
            }
        if task == "getup_ref":
            from .asimov_env import AsimovGetupEnv as E
            from .asimov_env import AsimovGetupRefEnv as RE
            # WHICH RUNG. `start_phase` is the real curriculum here and every
            # rung of it is a configuration measured to hold for 6 s, which is
            # what the retired blend ladder could not offer. `track_tol` is the
            # not-tracking cut — the lever that moved the G1 kick — and its
            # reachable set is measured: at 0.60 the `prone` rung terminates on
            # step ONE, because a cache pose sits 0.78 rad from the first
            # waypoint, so that rung needs 0.90 or looser.
            return {
                "start_phase": os.environ.get(RE.START_ENV) or "pike",
                "track_tol": float(os.environ.get(RE.TOL_ENV, "0.60") or 0.60),
                "ref_speed": float(os.environ.get(RE.SPEED_ENV, "1.0") or 1.0),
                # HOW MANY FRAMES THE POLICY SEES. 1 is the contract; anything
                # above it is a control arm that cannot be deployed, so it has
                # to be on the run or a 390-input policy is indistinguishable
                # from a 78-input one by anything but the ONNX itself.
                "history": int(os.environ.get(RE.HISTORY_ENV, "1") or 1),
                # WHETHER THE COMMAND DELAY IS FORCED. On the run because a
                # policy trained with it and one trained without are not the
                # same controller, and the difference is 7.76 s of hold against
                # 1.3 s.
                "force_lag": bool(int(os.environ.get(RE.LAG_ENV, "0") or 0)),
                # FREE mode: no imitation of the intermediates, just the
                # product rise. Recorded because a free policy and a tracking
                # one optimise different objectives entirely.
                "free": bool(int(os.environ.get(RE.FREE_ENV, "0") or 0)),
                # PRONE by default, and the reason is measured: from supine
                # this robot cannot rise at all, at any speed, by any scripted
                # route. See `asimov_getup_ref.WAYPOINTS[0]`.
                "getup_label": os.environ.get(E.POSE_ENV) or "prone",
                "action_clip": float(os.environ.get(E.CLIP_ENV, "")
                                     or spec().action_clip),
                "arm_push": float(os.environ.get(E.ARM_PUSH_ENV, "")
                                  or E.W_ARM_PUSH),
                "arm_effort": float(os.environ.get(E.ARM_EFFORT_ENV, "")
                                    or E.W_ARM_EFFORT),
                "head_up": float(os.environ.get(E.HEAD_ENV, "") or E.W_HEAD_UP),
                # WHICH RUNG the episodes start on. Recorded because a run that
                # starts on the floor and one that starts on the arm plateau are
                # two different halves of the task, not two seeds of one.
                "start_rung": os.environ.get(E.START_RUNG_ENV, "") or "",
                # The SPRAWL cut. Unrecorded, the env still read it from the
                # environment and trained with it while `run.json` said
                # nothing — so every later evaluation rebuilt the env WITHOUT
                # it and scored a different task. Measured: the staged chain's
                # eval reached a 0.877 m foot span under a cut set to 0.70.
                # This file's own docstring states the rule.
                "max_span": float(os.environ.get(E.SPAN_ENV, "") or 0.0),
                # How far down the fall `start_rung="falling"` may reset. THE
                # curriculum axis for the one leg nothing learns from a static
                # start — unrecorded, every later evaluation would rebuild a
                # different task.
                "fall_depth": float(os.environ.get(E.FALL_DEPTH_ENV, "")
                                    or 1.0),
                # The CONTINUOUS span charge and the width it is measured
                # from. On the run because a run that TERMINATES on a wide
                # stance and one that is CHARGED for it are not two seeds of
                # one experiment — the cut taught the policy to stop standing
                # altogether (`…-rise-126f42-s2`: up 0.0 from 98 k steps,
                # return 35.5 against a do-nothing floor of 39.8).
                # THE ASSIST FORCE, as a fraction of body weight held up at
                # the pelvis. It changes the DYNAMICS, so a run at 0.85 and
                # one at 0.00 are different robots — and on this task it is
                # the whole curriculum: 0.85 is the only setting measured to
                # cross the pike->fold gap (fold reached 12/12 at a 0.355 m
                # span) and 1.00 launches the robot to 1.8 m.
                "getup_assist": float(os.environ.get(E.ASSIST_ENV, "0.0")
                                      or 0.0),
                # THE SCRIPTED PUSH-UP. A run whose episodes begin with
                # 1.56 s of servo-driven prone->pike and one that starts at a
                # placed pike are not the same experiment: the policy inherits
                # a pike it ARRIVED at, with real velocity.
                "prologue": os.environ.get(E.PROLOGUE_ENV, "") or "",
                # GET THE FEET UNDER THE MASS — see `AsimovGetupEnv.TUCK_ENV`.
                "tuck_pay": float(os.environ.get(E.TUCK_ENV, "") or 0.0),
                # ROTATION pay. The get-up's blocker is rotation, not
                # height, and `progress` pays only height — so a run with this
                # on and one without optimise different things entirely.
                "rotate_pay": float(os.environ.get(E.ROTATE_ENV, "") or 0.0),
                # ROLL ONTO THE BACK — recorded on every branch, because a
                # knob a run SETS and does not RECORD is the bug this file
                # has now had five times.
                "roll_pay": float(os.environ.get(E.ROLL_ENV, "") or 0.0),
                "span_cost": float(os.environ.get(E.SPAN_COST_ENV, "")
                                   or 0.0),
                "span_target": float(os.environ.get(E.SPAN_TARGET_ENV, "")
                                     or 0.40),
                # THE FLIGHT CHARGE. Recorded on every branch, because a
                # knob a run SETS and does not RECORD is the bug this file
                # has now had five times.
                "ground_cost": float(os.environ.get(E.GROUND_ENV, "")
                                     or 0.0),
                "balance_cost": float(os.environ.get(E.BALANCE_ENV, "")
                                      or 0.0),
                "home_pay": float(os.environ.get(E.HOME_ENV, "") or 0.0),
                "limit_cost": float(os.environ.get(E.LIMIT_ENV, "") or 0.0),
                "push_bw": float(os.environ.get(E.PUSH_ENV, "") or 0.0),
                "calm_cost": float(os.environ.get(E.CALM_ENV, "") or 0.0),
                "swing_pay": float(os.environ.get(E.SWING_ENV, "") or 0.0),
                "handoff_pay": float(os.environ.get(E.HANDOFF_ENV, "")
                                     or 0.0),
            }
        if task != "getup":
            return {}
        from .asimov_env import AsimovGetupEnv as E
        out: dict = {
            # WHICH POSE CACHE: a cache is a set of `qpos` for THIS model, and
            # "supine" and "prone" are different tasks, not two seeds of one.
            "getup_label": os.environ.get(E.POSE_ENV) or "supine",
            # THE ASSIST FORCE, as a fraction of body weight held up at the
            # pelvis for the whole episode. Recorded because it changes the
            # DYNAMICS: a policy that learned its torques against half its
            # weight carried is being scored on a different robot once the
            # rung sets it to 0.
            "getup_assist": float(os.environ.get(E.ASSIST_ENV, "0.0") or 0.0),
            # THE REVERSE-CURRICULUM BLEND — recorded, and measured not to
            # work on this body (2026-10-04). `blend_to_stand` interpolates
            # between a sprawl and the stand, and an interpolation of two
            # poses is a THIRD pose: the passive basin is alpha >= 0.94 and
            # every rung ever shipped (0.8, 0.5, 0.2, 0.0) sat outside it, the
            # easiest one toppling in 0.34 s under ZERO action. Left reachable
            # so those four runs reproduce; a new run leaves it at 0.
            "getup_blend": float(os.environ.get(E.BLEND_ENV, "0.0") or 0.0),
            # ROLL ONTO THE BACK — the axis `rotate` is blind to. Recorded on
            # EVERY branch; this file has had the set-but-not-recorded bug
            # five times, and a sixth the day this term was added.
            "roll_pay": float(os.environ.get(E.ROLL_ENV, "") or 0.0),
            # THE ACTION CLIP. Recorded because it is reachability on this
            # body, not tuning: the inherited 4.0 reaches 1.0 rad from the
            # standing pose and the working get-up needs 2.88. A run at 4.0 and
            # one at 13.0 are not the same experiment — see E.CLIP_ENV.
            "action_clip": float(os.environ.get(E.CLIP_ENV, "")
                                 or spec().action_clip),
            # THE ARMS, both directions. Recorded because a run that pays for
            # pushing off the floor and one that only charges for arm torque
            # are solving different problems — measured, the working get-up
            # carries over half the body weight on its arms up to half standing
            # height. See `AsimovGetupEnv.ARM_PUSH_ENV`.
            "arm_push": float(os.environ.get(E.ARM_PUSH_ENV, "")
                              or E.W_ARM_PUSH),
            "arm_effort": float(os.environ.get(E.ARM_EFFORT_ENV, "")
                                or E.W_ARM_EFFORT),
            "head_up": float(os.environ.get(E.HEAD_ENV, "") or E.W_HEAD_UP),
            # WHICH RUNG the episodes start on. Recorded because a run that
            # starts on the floor and one that starts on the arm plateau are
            # two different halves of the task, not two seeds of one.
            "start_rung": os.environ.get(E.START_RUNG_ENV, "") or "",
            # The SPRAWL cut and the FALL-DEPTH curriculum axis. Both were read
            # from the environment and trained with while `run.json` said
            # nothing, so every later evaluation rebuilt a different task —
            # `teach-asimov_getup_catch-23657c` recorded start_rung="falling"
            # but not WHICH DEPTH, which is the whole curriculum.
            "max_span": float(os.environ.get(E.SPAN_ENV, "") or 0.0),
            "fall_depth": float(os.environ.get(E.FALL_DEPTH_ENV, "") or 1.0),
            "launch_span": float(os.environ.get(E.LAUNCH_SPAN_ENV, "") or 0.0),
            # The CONTINUOUS span charge and the width it is measured
            # from. On the run because a run that TERMINATES on a wide
            # stance and one that is CHARGED for it are not two seeds of
            # one experiment — the cut taught the policy to stop standing
            # altogether (`…-rise-126f42-s2`: up 0.0 from 98 k steps,
            # return 35.5 against a do-nothing floor of 39.8).
            # THE SCRIPTED PUSH-UP — see `AsimovGetupEnv.PROLOGUE_ENV`.
            "prologue": os.environ.get(E.PROLOGUE_ENV, "") or "",
            # ROTATION pay — see `AsimovGetupEnv.ROTATE_ENV`.
            # GET THE FEET UNDER THE MASS — see `AsimovGetupEnv.TUCK_ENV`.
            "tuck_pay": float(os.environ.get(E.TUCK_ENV, "") or 0.0),
            "rotate_pay": float(os.environ.get(E.ROTATE_ENV, "") or 0.0),
            "span_cost": float(os.environ.get(E.SPAN_COST_ENV, "")
                               or 0.0),
            "span_target": float(os.environ.get(E.SPAN_TARGET_ENV, "")
                                 or 0.40),
            "ground_cost": float(os.environ.get(E.GROUND_ENV, "")
                                 or 0.0),
            "balance_cost": float(os.environ.get(E.BALANCE_ENV, "")
                                  or 0.0),
            "home_pay": float(os.environ.get(E.HOME_ENV, "") or 0.0),
            "limit_cost": float(os.environ.get(E.LIMIT_ENV, "") or 0.0),
            "push_bw": float(os.environ.get(E.PUSH_ENV, "") or 0.0),
            "calm_cost": float(os.environ.get(E.CALM_ENV, "") or 0.0),
            "swing_pay": float(os.environ.get(E.SWING_ENV, "") or 0.0),
            "handoff_pay": float(os.environ.get(E.HANDOFF_ENV, "") or 0.0),
        }
        return out

    def env_class(self, task: str = "walk"):
        from .asimov_env import AsimovGetupEnv, AsimovGetupRefEnv, AsimovWalkEnv

        if task in ("walk", ""):
            return AsimovWalkEnv
        if task == "getup":
            return AsimovGetupEnv
        if task == "getup_ref":
            return AsimovGetupRefEnv
        if task == "getup_clip":
            from .asimov_env import AsimovGetupClipEnv
            return AsimovGetupClipEnv
        raise NotImplementedError(
            f"asimov1 has no {task!r} task — it has {', '.join(self.TASKS)}")

    def shipped_policies(self) -> tuple[dict, ...]:
        """The one policy that came with this robot, as a palette entry."""
        if not policy_ready():
            return ()
        return ({
            "id": "asimov1-shipped",
            "label": "Asimov 1 walk (Menlo)",
            "group": "shipped",
            "path": str(policy_onnx()),
            "robot": "asimov1",
            "note": "Menlo's own AMP+PPO locomotion policy, measured HERE "
                    "deterministically: 0.44 m/s at a 0.5 command, 0.68 at "
                    "0.8, turns 0.48 rad/s at 0.6, no falls in 10 s on any "
                    "of six commands",
        },)

    def attach(self, spec_, prefix: str, frame) -> None:
        """Put one Asimov in a `/sim` world, WITHOUT its own floor or lights.

        The shipped MJCF carries both (`robots/asimov.py`), and attaching
        them into a room would give that room two floors and four lights.
        """
        robot = walk_spec(standalone=False)
        robot.option.timestep = spec_.option.timestep
        spec_.attach(robot, prefix=prefix, frame=frame)

    @property
    def visual_group(self) -> int:
        """Which geom group holds the meshes a viewer should DRAW — detected
        off the model (`mjcf_body.visual_geom_group` reads `contype == 0`),
        never assumed to be 2. It answers 2 for this robot; assuming it would
        have been right here and wrong for MARS, whose URDF import emits 1."""
        from .mjcf_body import visual_geom_group

        return visual_geom_group(self.robot_model())

    def robot_model(self):
        """The compiled ROBOT (no floor, no light) — the model
        `visual_scene()` dumps, and what a checker counts meshes against.

        `robots/mjcf_body.MjcfBody` declares the same method, which is what
        lets `tests/test_body_conformance._visual_scene` count a new body's
        meshes with no branch of its own. Declaring it is the cheapest way a
        body opts out of being a special case.
        """
        return _painted_spec().compile()

    def visual_scene(self) -> dict:
        """The viewer's mesh dump — `g1.extract_visual_scene`, unchanged.

        Built from the ROBOT file rather than the walk scene: a scene carries
        the floor plane and the skybox, and the viewer draws its own.
        Group 2 is this model's visual group, DETECTED rather than assumed
        (`mjcf_body.visual_geom_group` reads it off `contype == 0`, and it
        answered 2 for this robot).
        """
        return _visual_scene()

    def driver(self, model, prefix: str):
        """`AsimovWalker` on this model, at its shipped policy.

        `G1Walker`'s slot in the contract. The world calls `control(data)`
        once a 50 Hz tick and does its own `mj_step`s, so the driver must
        write `ctrl` and NOT step — see `AsimovWalker.control`.
        """
        return AsimovWalker(model, prefix, policy=str(policy_onnx()))

    def frames(self):
        """Root link, and no gaze joint: the neck is modelled but has no
        joint in this MJCF (the README's 2-DoF neck is not in `sim-model`),
        so a brain asking to look down reaches nothing and says so."""
        from .body import RobotFrames

        return RobotFrames(base="pelvis_link", head_pitch_joint=None,
                           head_pitch_sign=1.0)


@lru_cache(maxsize=1)
def spec():
    """Asimov 1 as a `RobotSpec` — level 1, the body the walking env can run.

    Level 1 and not 2 or 3: there is an env and an exportable policy
    interface, but no sense channels, no driver in a `/sim` room, and NO
    hardware claim from this repo. The gains, armatures and torque limits
    below are Menlo's own measurements of their motors, not ours.

    `obs_dim` is 78 and `twist_obs_slice` is (6, 9) — the command sits BEFORE
    the joints here, where the duck's and the G1's sit last. That field exists
    for exactly this: `train._seed_command_stats` has to know which slots the
    command occupies, and a layout where it is not at the end is why the slice
    is declared rather than derived from the end of the vector.
    """
    return AsimovBody(
        id="asimov1",
        title="Asimov 1",
        noun="Asimov",
        kind="legged",
        joint_names=JOINT_NAMES,
        joint_groups=asimov.joint_groups(JOINT_NAMES),
        default_pose=DEFAULT_POSE,
        action_scale=np.full(NUM_JOINTS, ACTION_SCALE, np.float32),
        obs_dim=OBS_DIM,
        base_body="pelvis_link",
        gyro_sensor="imu_ang_vel",
        foot_geoms=FOOT_GEOMS,
        floor_geom="floor",
        scene_fn=walk_scene_xml,
        stand_keyframe=STAND_KEY,
        pose_joint_ids=None,              # hold every joint near default
        fall_gravity_z=FALL_GRAVITY_Z,
        fall_height=FALL_HEIGHT,
        com_bodies=("pelvis_link",),
        lin_vel_x_range=LIN_VEL_X_RANGE,
        lin_vel_y_range=LIN_VEL_Y_RANGE,
        ang_vel_z_range=ANG_VEL_Z_RANGE,
        twist_obs_slice=OBS_COMMAND,
        min_forward_cmd=MIN_FORWARD_CMD,
        # MEASURED at STAND by `body.measure_lab_spacing_m`: 0.4560 m across
        # -> 1.605 m. It was 1.495 (0.4248 m) until the look started CUTTING
        # meshes. The robot did not change shape — `widest_horizontal_extent_m`
        # bounds each geom by its own AABB rotated into world, and a rotated
        # box is a BOUND, not an extent, so splitting one mesh into two
        # loosens it. 31 mm of slack on the stage pitch, in the safe
        # direction: two of these stand further apart, never closer.
        lab_spacing_m=1.605,
        sole_tol=SOLE_TOL,
        ground_tol=GROUND_TOL,
        effectors=_effectors(),
        noise_gyro=NOISE_ANG_VEL,
        noise_gravity=NOISE_GRAVITY,
        noise_joint_pos=NOISE_JOINT_POS,
        noise_joint_vel=NOISE_JOINT_VEL,
        extra={
            "repo": asimov.REPO, "sha": asimov.ASIMOV_SHA,
            "policy_repo": POLICY_REPO, "policy_sha": POLICY_SHA,
            "licences": ("model: GPL-2.0 / CERN-OHL-S-2.0",
                         f"policy: {POLICY_LICENCE}"),
        },
    )


def _effectors():
    """The points IK may drag — and they are the points that can CARRY LOAD.

    MEASURED off the model (2026-10-04), because this body's answer differs
    from the G1's in the one way that matters for a get-up: **the wrist links
    carry no collision geom at all.** `left_wrist_yaw_link` has a visual mesh
    and nothing else, so a hand cannot touch the floor, cannot bear weight and
    cannot be planted. The G1's `_effectors` offers hands at (0.06, 0, 0) and
    copying that here would have authored a get-up on a contact that does not
    exist — four phases of it resting on thin air.

    What CAN bear load, from the collision set: pelvis, waist, neck, the two
    hip links, the knee (a 0.13 m half-length capsule down the shin, so the
    whole shin is the kneeling surface), the four 5 mm foot spheres per ankle,
    the shoulder rolls and the ELBOW (a capsule reaching to +0.132 m along the
    forearm). So a get-up on Asimov plants FOREARMS, not hands.

    The knee and elbow points are their joints — body origin, `point=None` —
    which is the natural thing to aim at a floor height, the capsule's radius
    doing the rest once the pose settles. The feet ask for `"sole"`, the
    measured footprint centre at floor level: 0.167 m long (toe +0.122, heel
    -0.045) by 0.056 m wide, which is a lot of room forward of the ankle and
    very little behind it.
    """
    from .spec import Effector
    return (
        Effector("left_foot", "left foot", "left_ankle_roll_link", "foot", "sole"),
        Effector("right_foot", "right foot", "right_ankle_roll_link", "foot", "sole"),
        Effector("left_knee", "left knee", "left_knee_link", "knee"),
        Effector("right_knee", "right knee", "right_knee_link", "knee"),
        Effector("left_elbow", "left forearm", "left_elbow_link", "forearm"),
        Effector("right_elbow", "right forearm", "right_elbow_link", "forearm"),
    )


CONTRACT_ID = "asimov1-78-v0"


def obs_slots() -> tuple:
    """The 78-float layout as a slot table, in contract order."""
    from .policy_contract import Slot

    return (
        Slot("base_ang_vel", *OBS_ANG_VEL),
        Slot("projected_gravity", *OBS_GRAVITY),
        Slot("twist_cmd", *OBS_COMMAND),
        Slot("joint_pos_rel", *OBS_JOINT_POS),
        Slot("joint_vel", *OBS_JOINT_VEL),
        Slot("last_action", *OBS_LAST_ACTION),
    )


@lru_cache(maxsize=1)
def _visual_scene() -> dict:
    """Cached: a 25-mesh dump is heavy and the lab asks per slot.

    Both the lattice and the visual group are MEASURED off the model rather
    than passed, which is the whole point of `mjcf_body`'s two readers: a
    stranger's robot cannot be asked, and the one number that was inherited
    instead of measured is the one that went wrong. This file hard-coded
    `vert_scale=0.001` — the G1's millimetre — on the reasoning that Asimov is
    a G1-sized biped, and the robot arrived in the browser low-detail and
    melted anyway, because the lattice has nothing to do with how TALL a robot
    is and everything to do with how finely its shells are TESSELLATED.

    MEASURED here, as the share of faces whose normal a lattice tilts by more
    than 15 deg (`visual_vert_scale`'s own number, which is what shading is):

    | lattice | faces tilted >15 deg | dump |
    |---|---|---|
    | 1 mm, inherited | **74.0 %** | 26.5 MB |
    | 0.1 mm, MARS's | 24.8 % | 27.9 MB |
    | 0.01 mm, PICKED | **0.6 %** | 29.3 MB |

    Asimov's 10th-percentile triangle edge is 0.140 mm, so `visual_vert_scale`
    lands on the ladder's finest rung, 10 um. Three quarters of his faces were
    being shaded off a normal the quantisation invented, against 40.4 % for
    the Go2 that helper was written on — he is the worst case measured so far,
    and 2.8 MB (+10.6 %) is what it costs to stop it.
    """
    from .g1 import extract_visual_scene
    from .mjcf_body import visual_geom_group, visual_vert_scale

    model = _painted_spec().compile()
    group = visual_geom_group(model)
    # ...and his own normals: there is no reason to let the browser invent
    # what the model already states, and all 26 meshes carry them per vertex,
    # so all 26 ship them. (The G1's and MARS's looks were TUNED against the
    # invented ones and stay off by default; Asimov's look was built after
    # this, so `duck-viewer/components/AsimovLook.tsx` has never seen a
    # normal the model did not author.)
    return extract_visual_scene(model, group=group, normals=True,
                                vert_scale=visual_vert_scale(model, group))
