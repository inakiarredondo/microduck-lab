"""Asimov 1 in the walking env — level 1, and their layout not ours.

`G1WalkEnv`'s shape: the duck's `MicroduckWalkEnv` with this body's
observation written in its place. What differs here, and why it is the
interesting one of the three:

* **The command sits at obs[6:9], before the joints.** The duck's and the
  G1's layouts both end with the twist. `RobotSpec.twist_obs_slice` exists so
  the normaliser can find it wherever it is, and this is the first body that
  actually exercises that.
* **The joints are permuted into three CAN-bus slots**, not written in joint
  order — `robots/asimov_walk.SLOTS`. The permutation is the single thing
  most likely to be got wrong silently, which is why it comes from their
  published `env.yaml` and is cross-checked against the viewer's index lists.
* **This layout already has a trained policy in it.** `scripts/probe_asimov_walk.py`
  runs Menlo's own ONNX through this contract and it walks, which is the only
  real evidence the layout is theirs. Prefer that check over any reward curve
  measured here.

**The reward windows below are NOT measured for this body.** They are the
duck's and the G1's, and `AGENTS.md` is explicit that the structure ports
while the windows do not — the duck's air-time window paid on 0 % of G1
swings until it was re-measured. So this env is honest for PLAYBACK and
preview (which is what the lab asks of it) and the air-time, clearance and
pose windows have to be measured off the shipped policy's own gait before
anything trained here means anything. The one window that IS measured is
`GROUND_TOL`, because a stance/swing split was needed to measure the others.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from ..contract import DECIMATION as C_DECIMATION
from ..walk_env import MicroduckWalkEnv
from . import asimov_walk as W


class AsimovWalkEnv(MicroduckWalkEnv):
    """Velocity-command walking on Asimov 1, in Menlo's own 78-float layout."""

    # No head joints on this body (the MJCF has a neck LINK and no neck
    # joint), and no head slots in the observation.
    W_HEAD_POSE = 0.0
    # Inherited and UNMEASURED for this robot — see the module docstring.
    # The G1's numbers, because a 1.2 m biped is nearer a 1.3 m one than a
    # 25 cm duck, and that is a starting point rather than a result.
    AIR_TIME_MIN = 0.08
    AIR_TIME_MAX = 0.30
    W_POSE = 0.5
    POSE_STD2 = 4.0

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot", W.spec())
        act = kwargs.get("actuator_force") or kwargs.get("actuator") or "xml"
        if str(act).lower() == "bam":
            raise ValueError(
                "actuator='bam' is the duck's XL330 servo model — Asimov runs "
                "the Synapticon servos injected by robots/asimov_walk.walk_spec "
                "('xml')")
        kwargs.setdefault("actuator_force", "xml")
        super().__init__(*args, **kwargs)
        #: The 23 joint indices in observation order (slot01, 23, 45).
        self._slot = np.array(W.slot_order())

    # ----------------------------------------------------------- observations

    def _get_obs(self) -> np.ndarray:
        """78-d, `Menlo/asimov1-locomotion-0818`'s env.yaml order."""
        gyro = self._gyro.astype(np.float32)
        gravity = self._projected_gravity()
        joint_pos = self._joint_pos_rel()
        joint_vel = self._joint_vel()
        self.prev_joint_vel = joint_vel

        if self.obs_noise:
            r = self._rng
            spec = self.robot
            nj = self.nj
            gyro = gyro + r.uniform(-spec.noise_gyro, spec.noise_gyro, 3).astype(np.float32)
            gravity = gravity + r.uniform(
                -spec.noise_gravity, spec.noise_gravity, 3).astype(np.float32)
            joint_pos = joint_pos + r.uniform(
                -spec.noise_joint_pos, spec.noise_joint_pos, nj).astype(np.float32)
            joint_vel = joint_vel + r.uniform(
                -spec.noise_joint_vel, spec.noise_joint_vel, nj).astype(np.float32)

        obs = np.empty(self.robot.obs_dim, np.float32)
        obs[slice(*W.OBS_ANG_VEL)] = gyro * W.ANG_VEL_SCALE
        obs[slice(*W.OBS_GRAVITY)] = gravity
        obs[slice(*W.OBS_COMMAND)] = self.twist_cmd
        obs[slice(*W.OBS_JOINT_POS)] = joint_pos[self._slot]
        obs[slice(*W.OBS_JOINT_VEL)] = joint_vel[self._slot] * W.JOINT_VEL_SCALE
        obs[slice(*W.OBS_LAST_ACTION)] = self.last_action
        return obs

    # -------------------------------------------------------------- commands

    def _sample_commands(self) -> None:
        """Twist only — this body's observation has no head or body slots."""
        r = self._rng
        u = r.uniform()
        stand = self.zero_command_prob
        turn = stand + self.turn_in_place_prob
        fwd = turn + self.forward_command_prob
        spec = self.robot
        if u < stand:
            self.twist_cmd[:] = 0.0
        elif u < turn:
            self.twist_cmd[:] = (0.0, 0.0, r.uniform(*spec.ang_vel_z_range))
        elif u < fwd:
            vx = abs(float(r.uniform(*spec.lin_vel_x_range)))
            self.twist_cmd[:] = (max(vx, spec.min_forward_cmd), 0.0, 0.0)
        else:
            self.twist_cmd[:] = (
                r.uniform(*spec.lin_vel_x_range),
                r.uniform(*spec.lin_vel_y_range),
                r.uniform(*spec.ang_vel_z_range),
            )

    # --------------------------------------------------------------- rewards

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        reward, terms = super()._compute_reward()
        head = terms.pop("head_pose", 0.0)
        return float(reward - head), terms


def make_asimov_env(**kwargs) -> AsimovWalkEnv:
    kwargs.setdefault("max_episode_s", W.EPISODE_S)
    return AsimovWalkEnv(**kwargs)


class AsimovGetupEnv(AsimovWalkEnv):
    """Stand up from a fallen pose, inside the walking contract.

    `docs/roadmap.md` Track 14. Three things differ from the walking env, and
    the first two are the reason a get-up cannot just be "walking with a
    different reward":

    1. **It resets into a fallen pose**, drawn from the cache
       `robots/asimov_getup` collects. A get-up is an EXPLORATION problem —
       from a sprawl, random torques essentially never reach standing, so the
       reward is never sampled and no amount of tuning finds it. The reset
       distribution is the lever, not the pay (`AGENTS.md`, "Reward design
       rules").
    2. **The fall cut is OFF.** The walking env ends an episode when the body
       tilts past 70 deg or the base drops below `fall_height`. A get-up
       STARTS past both, so inheriting them would terminate every episode on
       its first step — the inherited-termination version of the mistake this
       repo warns about most.
    3. **Nothing is commanded.** The twist stays zero, so `obs[6:9]` is zero
       throughout. Those three slots are the free channel inside the 78-float
       contract if a phase or mode input is ever needed (Track 14 phase 2).

    The reward is deliberately plain for a first attempt: get the pelvis up,
    get the chest upright, get the feet under you, then stop moving. The one
    borrowed subtlety is Menlo's — a penalty on ARM effort, so it does not
    learn to stand by hanging off its shoulders, which is both unrealistic and
    the joint group already measured at 100 % of rated torque while walking.
    """

    #: HOW FAR FROM STANDING THE POLICY MAY COMMAND, as a knob, because on this
    #: body it is a REACHABILITY constraint and not a tuning preference. The
    #: body's inherited default is 4.0, which at `ACTION_SCALE` 0.25 reaches
    #: **1.0 rad** from `DEFAULT_POSE` — and the one get-up known to work on
    #: this model needs **2.88 rad** (its action peaks at 11.51) and exceeds 4.0
    #: on 15.5 % of its frames. Every from-scratch chain before 2026-10-06 ran
    #: at 4.0, so none of them could command the motion at all; that was
    #: invisible because the override reached only the reset pose and not
    #: `walk_env.step` (see `walk_env._act_clip`).
    CLIP_ENV = "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP"

    #: THE ARMS. This body cannot get up on its legs alone, and the one get-up
    #: known to work on it does not try: measured over its own demonstration,
    #: its arm joints move **1.25x** as much as its leg joints, and after the
    #: two feet the bodies most often on the floor are `right_elbow_link` (121
    #: contact-steps of 900), `right_shoulder_roll_link` and `left_elbow_link`.
    #: It braces on an elbow and pushes.
    #:
    #: This env only ever charged for that. `arm_effort_penalty` exists so a
    #: policy does not stand by hanging off its shoulders — a real concern, the
    #: arm group already measures at 100 % of rated torque while walking — but
    #: "do not hang off your shoulders" and "never push off the floor" are
    #: different constraints and only the second was implemented. Priced: the
    #: penalty would charge their whole get-up 6.5 a episode against ~135 of
    #: task reward, so it does not FORBID the arms; nothing paid for them
    #: either, and a policy that has to discover elbow-planting by accident had
    #: only a negative signal attached to trying it.
    #:
    #: `arm_push` pays for HEIGHT GAINED WHILE AN ARM CARRIES LOAD, which is
    #: the thing a get-up actually needs, and it is paid on NEW BEST rise only.
    #: Three farms are closed by that shape, all of which this task has already
    #: been caught by: lying on your arms earns nothing (no new best), jittering
    #: up and down earns nothing (only new bests), and holding a pose while
    #: pressing an arm down earns nothing (no height gained).
    #: THE HEAD, as a term of its own, because the TRUNK height cannot see it.
    #: Along the working get-up's rise, head and trunk height track almost
    #: perfectly (0.98 vs 0.97 of standing) and a head term would be redundant
    #: — but that is measured AFTER it is already rising. Across the 485 fallen
    #: START poses, which is where a reward has to bite, the two correlate at
    #: only **r = 0.317**, and in **59 %** of them they differ by more than
    #: 0.12 of standing with the head at 0.05 while the trunk is at 0.21: head
    #: flat on the floor, hips already up. `height` pays full marks for that.
    #: Getting the head up is the first half of sitting up and nothing priced
    #: it.
    #: WHICH RUNG an episode starts on — the second half of the curriculum.
    #:
    #: "" (default) starts where the robot FELL, from the pose cache. The other
    #: values start it on a pose measured to HOLD for 6 s
    #: (`asimov_getup_ref.holdable_rungs`), which is what every previous ladder
    #: on this task lacked: all three built their rungs by interpolating, and an
    #: interpolated pose is not a pose this robot can hold.
    #:
    #: The stable set is two disconnected plateaus — arm-supported 0.212-0.365 m
    #: and feet-only 0.514-0.606 m — with a 15 cm BALLISTIC gap between them.
    #: So the get-up is two rungs and ONE unlearned transition:
    #:
    #:   "" / "floor"  the fallen poses -> the arm plateau. A from-scratch run
    #:                 already does this: it reaches 0.395 m, the top of the
    #:                 plateau, and stalls there because the next stable thing
    #:                 is 15 cm away through the air.
    #:   "arm_top"     START on the highest arm-supported holds and learn the
    #:                 LAUNCH. This is the transition nothing has practised: the
    #:                 episode begins stable, chest down, and the policy only
    #:                 has to cross the gap and catch the crouch plateau.
    #:   "arm"/"feet"  the whole plateau, for scoring and for a wider start mix.
    START_RUNG_ENV = "MICRODUCK_ASIMOV_GETUP_START_RUNG"
    HEAD_ENV = "MICRODUCK_ASIMOV_GETUP_HEAD"
    ARM_PUSH_ENV = "MICRODUCK_ASIMOV_GETUP_ARM_PUSH"
    ARM_EFFORT_ENV = "MICRODUCK_ASIMOV_GETUP_ARM_EFFORT"

    #: Pelvis height and uprightness that together count as "up".
    UP_HEIGHT_FRAC = 0.90
    UP_GRAVITY = 0.90

    #: THE term. `rise = height x uprightness` and `stand = rise x feet`,
    #: paid as PRODUCTS rather than as separate axes, because every version of
    #: this reward that paid an axis on its own was farmed by a trick that
    #: satisfied that axis and nothing else:
    #:
    #:   run be7733  `upright` paid 3.0/step for torso ORIENTATION alone, so
    #:               it sat up with its hips on the floor and never moved
    #:               again: upright 885/1200, height 221/1200, standing 0.00.
    #:   run 08bdf5  gating upright on height killed the sit, and `height`
    #:               paid 3.0/step for PELVIS HEIGHT alone, so it rolled onto
    #:               its head and threw its legs over itself — a shoulder
    #:               stand reaching 0.52 m with the torso INVERTED (u_z -0.57):
    #:               height 747/1200, upright 0.30, standing 0.00.
    #:
    #: A product cannot be farmed by one axis: the sit has h ~ 0.2, the
    #: shoulder stand has u ~ 0, and both go to nearly nothing. `rise` leaves
    #: out the feet on purpose, so a kneel or a crouch — real intermediates on
    #: the way up — still score.
    W_STAND = 8.0
    W_RISE = 2.0
    W_STANDING = 5.0
    W_JOINT_VEL = 0.002
    W_ACTION_RATE = 0.5
    W_ARM_EFFORT = 0.02
    #: Charged for asking for a command the actuator CANNOT be given — the
    #: amount by which the raw network output exceeds the clip, squared.
    #:
    #: `walk_env.step` stores the RAW output in `last_action` precisely so that
    #: unbounded growth is not free, and the action-RATE penalty was carrying
    #: that job. It cannot carry it at a wide clip: the rate penalty's ceiling
    #: is `(2 * clip) ** 2`, so raising the clip to 13.0 multiplied it 10.6x and
    #: a run ended at `action_rate_penalty` **-694** against ~130 of task
    #: reward. Normalising the rate by the clip fixed that and immediately
    #: produced the opposite failure: with the pressure removed, the next run's
    #: raw output reached a mean per-step max of **2055**.
    #:
    #: This term is the targeted replacement. It is EXACTLY ZERO inside the
    #: clip, so it cannot eat the task, and it grows quadratically outside it,
    #: where every unit is a command the robot could never execute.
    W_OVER_CLIP = 0.2
    #: Paid on each new best `rise`, scaled by the fraction of body weight the
    #: arms are carrying at that moment. 0.0 keeps the old economy exactly.
    W_ARM_PUSH = 4.0
    #: Head height as a fraction of standing, paid like `rise` — as a PRODUCT
    #: with uprightness, so a shoulder stand (head high, body inverted) earns
    #: nothing. That failure mode is not hypothetical: run 08bdf5 paid raw
    #: pelvis height and learned to throw its legs over its own head.
    W_HEAD_UP = 60.0
    #: Paid once for each new best `rise`, so there is dense signal toward
    #: standing. On `rise`, NOT on raw height — paying progress on height is
    #: what bought the shoulder stand.
    W_PROGRESS = 40.0

    #: Upward force on the pelvis while learning, as a fraction of body
    #: weight — Menlo's "temporary helper force ... this fades away as the
    #: policy improves, so the final skill doesn't depend on it". Its job is
    #: not to make training gentler: it is to make STANDING get SAMPLED at
    #: all, which is the thing the first run never did. Faded by the
    #: curriculum rather than continuously, so each rung is a run you can
    #: look at; the last rung is 0.0 and is the only one that counts.
    ASSIST_ENV = "MICRODUCK_ASIMOV_GETUP_ASSIST"
    #: How far back toward STANDING each episode starts — the REVERSE
    #: CURRICULUM, and **MEASURED NOT TO WORK on this body**: see
    #: `asimov_getup.blend_to_stand`. Kept only so the four runs that used it
    #: can be reproduced; a new run leaves it at 0.
    BLEND_ENV = "MICRODUCK_ASIMOV_GETUP_BLEND"
    #: Which pose cache the episode resets from.
    POSE_ENV = "MICRODUCK_ASIMOV_GETUP_POSE"

    #: Having been UP, the episode ENDS when the body falls back past these.
    #: Looser than the `up` gate on purpose: `up` flickering false is a
    #: wobble, and terminating on a wobble would charge a recoverable stumble
    #: as a failure. Crossing these is a fall.
    FELL_GRAVITY = 0.50
    FELL_HEIGHT_FRAC = 0.50

    def __init__(self, *args, getup_label: str | None = None,
                 action_clip: float | None = None,
                 getup_blend: float | None = None,
                 arm_push: float | None = None,
                 arm_effort: float | None = None,
                 head_up: float | None = None,
                 start_rung: str | None = None,
                 getup_assist: float | None = None, **kwargs):
        """The curriculum knobs arrive as KWARGS, not as `os.environ` reads.

        They used to be read here, and `robots/moss.py` already says why that
        is wrong: *"an env that reads its own environment trains fine and
        leaves no record of what it trained on."* It cost four runs. Nothing
        in `run.json` said which rung a stage trained at, so every later
        render of those runs took its blend and assist from whatever shell it
        ran in, and no claim about them can be checked. `AsimovBody.
        train_env_kwargs` reads the variables now and returns them, so the
        trainer writes them onto the run.

        The environment variables remain as a FALLBACK for a direct
        construction (a probe, a render) that passes nothing, because the lab
        reaches the trainer subprocess through them — but the trainer no
        longer depends on that path.
        """
        # A get-up is seconds, not a minute. Long enough to fail slowly,
        # short enough that a wasted episode is cheap.
        kwargs.setdefault("max_episode_s", 8.0)
        super().__init__(*args, action_clip=action_clip, **kwargs)

        from . import asimov_getup as G

        getup_label = (getup_label
                       or os.environ.get(self.POSE_ENV)
                       or "supine")
        self._poses = G.load_cache(getup_label)
        self._getup_label = getup_label
        # The inherited cuts never fire: this env BEGINS past both of them, so
        # keeping them would terminate every episode on its first step. The
        # cut that replaces them is armed by `_stood` — see `step`. Set AFTER
        # super(), which reads them off the spec.
        self._fall_gravity_z = 2.0
        self._fall_height = -1.0
        arm = [i for i, n in enumerate(W.JOINT_NAMES)
               if any(k in n for k in ("shoulder", "elbow", "wrist"))]
        self._arm_ids = np.array(arm)
        self._stand_z = float(W.STAND_Z)
        self._assist = float(getup_assist if getup_assist is not None
                             else os.environ.get(self.ASSIST_ENV, "0.0") or 0.0)
        self._blend = float(getup_blend if getup_blend is not None
                            else os.environ.get(self.BLEND_ENV, "0.0") or 0.0)
        #: Standing head height, measured off the model's own STAND keyframe —
        #: never carried across model revisions (AGENTS.md).
        self._head_body_id = self.model.body("neck_pitch_link").id
        _d = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, _d, self.key_stand)
        mujoco.mj_forward(self.model, _d)
        self._head_stand_z = float(_d.xpos[self._head_body_id][2])
        self._w_head_up = float(head_up if head_up is not None
                                else os.environ.get(self.HEAD_ENV, "")
                                or self.W_HEAD_UP)
        self._start_rung = str(start_rung if start_rung is not None
                               else os.environ.get(self.START_RUNG_ENV, "")
                               or "").strip().lower()
        self._rungs = None
        if self._start_rung and self._start_rung != "floor":
            from . import asimov_getup_ref as _R

            z = _R.holdable_rungs()
            sup, zz = z["support"], z["trunk_z"]
            if self._start_rung == "arm_top":
                keep = (sup == "arm") & (zz >= np.quantile(zz[sup == "arm"], 0.75))
            elif self._start_rung in ("arm", "feet"):
                keep = sup == self._start_rung
            elif self._start_rung == "low":
                # The bottom of the ladder: every rung within 15 cm of the
                # floor. This is the band that was missing — a search filter
                # requiring trunk > 0.15 m had excluded it, so the first stage
                # of the curriculum had nothing to aim at between the fallen
                # pose (~0.11 m) and 0.21 m, and it stalled at 0.203 m.
                keep = zz <= 0.26
            elif self._start_rung == "mix":
                # EVERY rung and the floor too, so a final stage practises the
                # whole task rather than only its last rung. A chain whose last
                # stage starts on a crouch learns to hold a crouch; the
                # deliverable starts on the ground.
                keep = np.ones(len(sup), bool)
                #: Drawn per SUPPORT and not per pose. There are 95 arm rungs
                #: and 5 feet ones, so a uniform draw over the cache spends 19
                #: of every 20 rung episodes on the push-up: measured, 27 arm
                #: starts against 3 feet starts in 60 episodes, when the feet
                #: plateau is the one that teaches the HOLD.
                self._rung_groups = tuple(
                    np.flatnonzero(sup == k) for k in ("arm", "feet")
                    if (sup == k).any())
            else:
                    raise ValueError(
                    f"{self._start_rung!r} is not a rung — use floor, arm, "
                    "arm_top, feet or mix")
            if not keep.any():
                raise RuntimeError(f"no {self._start_rung!r} rungs in the cache")
            self._rungs = (z["joints"][keep], z["root_pitch"][keep],
                           zz[keep])
        self._w_arm_push = float(arm_push if arm_push is not None
                                 else os.environ.get(self.ARM_PUSH_ENV, "")
                                 or self.W_ARM_PUSH)
        self._w_arm_effort = float(arm_effort if arm_effort is not None
                                   else os.environ.get(self.ARM_EFFORT_ENV, "")
                                   or self.W_ARM_EFFORT)
        #: Arm geoms that can touch the floor — the forearm/elbow/shoulder
        #: shells. The WRISTS carry no collision geom on this body, which is
        #: why it plants forearms (`add-a-robot` SKILL.md §4b).
        self._arm_geoms = frozenset(
            g for g in range(self.model.ngeom)
            if int(self.model.geom_contype[g]) and self.model.geom_bodyid[g]
            and any(k in self.model.body(self.model.geom_bodyid[g]).name
                    for k in ("shoulder", "elbow", "wrist")))
        #: Body weight in newtons — the denominator for `_arm_load`.
        self._weight_n = float(
            self.model.body_subtreemass[self.trunk_body_id]) * 9.81
        self._assist_n = self._assist * float(
            self.model.body_subtreemass[self.trunk_body_id]) * 9.81
        self._best_rise = 0.0
        #: Has it been `up` yet this episode? Arms the fall cut, and nothing
        #: else: an episode that never stands cannot "fall".
        self._stood = False
        #: Control steps spent `up`: the longest unbroken run, and the total.
        #: `standing` sums the same gate, but a SUM cannot tell 40 scattered
        #: frames from one 0.8 s hold, and the difference is the whole task.
        self._held = 0
        self._held_best = 0
        self._peak_rise = 0.0

    # ------------------------------------------------------------- reset

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        use_rung = self._rungs is not None
        if use_rung and self._start_rung == "mix":
            # Half the episodes from where it FELL. The mix is deliberately
            # even rather than weighted by rung count: there are 95 arm rungs
            # and 5 feet ones, so drawing uniformly over the cache would spend
            # 19 of every 20 rung episodes on the push-up and almost none on
            # the hold.
            use_rung = bool(self._rng.integers(2))
        if use_rung:
            q, _pitch, _z = self._rungs
            groups = getattr(self, "_rung_groups", None)
            if groups:
                g = groups[int(self._rng.integers(len(groups)))]
                i = int(g[int(self._rng.integers(len(g)))])
            else:
                i = int(self._rng.integers(len(q)))
            # The SETTLED qpos, written straight in — not placed and re-settled.
            # `holdable_rungs` stores the state after the full 6 s hold because
            # the commanded configuration and its equilibrium are 25 cm apart:
            # placing the lowest arm rung puts the trunk at 0.462 m and it sinks
            # to 0.212 m. Starting from the configuration starts the episode
            # with a fall in progress, which is the exact defect of the three
            # interpolated ladders this replaces.
            self.data.qpos[:] = np.asarray(q[i])
            self.data.qvel[:] = 0.0
            self.data.xfrc_applied[:] = 0.0
            self.data.xfrc_applied[self.trunk_body_id, 2] = self._assist_n
            mujoco.mj_forward(self.model, self.data)
            self.twist_cmd[:] = 0.0
            h0 = min(max(self.data.xpos[self.trunk_body_id][2]
                         / self._stand_z, 0.0), 1.0)
            self._best_rise = h0 * max(float(-self._projected_gravity()[2]), 0.0)
            self._stood = False
            self._held = 0
            self._held_best = 0
            self._peak_rise = self._best_rise
            self._best_trunk_z = None
            self._best_head = None
            return self._get_obs(), info
        pose = self._poses[int(self._rng.integers(len(self._poses)))]
        if self._blend > 0.0:
            from . import asimov_getup as G

            pose = G.blend_to_stand(self.model, pose, self._blend)
        self.data.qpos[:] = pose
        self.data.qvel[:] = 0.0
        # `xfrc_applied` persists across `mj_step`, so setting it here holds
        # it for the episode. On the PELVIS, which is the body a person lifts
        # a fallen robot by.
        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.trunk_body_id, 2] = self._assist_n
        mujoco.mj_forward(self.model, self.data)
        self.twist_cmd[:] = 0.0
        h0 = min(max(self.data.xpos[self.trunk_body_id][2] / self._stand_z, 0.0), 1.0)
        self._best_rise = h0 * max(float(-self._projected_gravity()[2]), 0.0)
        self._stood = False
        self._held = 0
        self._held_best = 0
        self._peak_rise = self._best_rise
        # Cleared, or the first control step of an episode differences against
        # the LAST frame of the previous one and reads as a huge jump.
        self._best_trunk_z = None
        self._best_head = None
        return self._get_obs(), info


    def _place(self, q, root_pitch, clearance=0.002):
        """Put the robot in `q` at `root_pitch`, dropped onto the floor."""
        m, d = self.model, self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[0:3] = [0.0, 0.0, 1.0]
        half = float(root_pitch) / 2.0
        d.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]
        #  (the base env's) rather than :  now
        # lives on AsimovGetupEnv, which a rung start needs, and  is
        # defined further down on the reference subclass. Same 23 addresses.
        d.qpos[getattr(self, "_q_adr", self.joint_qpos_adr)] = np.asarray(q)
        mujoco.mj_forward(m, d)
        # MuJoCo's own collision detection, not a bound on the geometry — see
        # `asimov_getup_ref.settle_onto_floor` for the 50 mm of float and 25 mm
        # of burial the `geom_rbound` version was leaving at reset.
        # The module, not self._ref: _place lives on AsimovGetupEnv now (a rung
        # start needs it) and _ref is set by the reference subclass.
        from . import asimov_getup_ref as _R

        _R.settle_onto_floor(m, d, clearance=max(clearance, 0.001))

    def _sample_commands(self) -> None:
        """Nothing is asked of it but to stand up."""
        self.twist_cmd[:] = 0.0

    # ------------------------------------------------------------ reward

    def _arm_load(self) -> float:
        """Fraction of body weight currently carried through the ARMS.

        The normal component of every arm-geom/floor contact, over `mg`. Read
        with `mj_contactForce` rather than counted as a boolean, because a
        forearm resting on the floor and a forearm PUSHING are the same contact
        and only the second one gets you up.
        """
        n = int(self.data.ncon)
        if not n:
            return 0.0
        con, floor, arms = self.data.contact, self.floor_geom, self._arm_geoms
        g1, g2 = con.geom1.tolist(), con.geom2.tolist()
        total, buf = 0.0, np.zeros(6)
        for i in range(n):
            a, b = g1[i], g2[i]
            if a == floor:
                other = b
            elif b == floor:
                other = a
            else:
                continue
            if other not in arms:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, buf)
            total += abs(float(buf[0]))          # normal, contact frame
        return total / self._weight_n if self._weight_n else 0.0

    def _action_rate(self) -> float:
        """Mean squared command change, AS A FRACTION OF THE CLIP.

        Divided by the clip because the two are otherwise coupled and the
        coupling is destructive. `walk_env.step` stores the RAW network output
        in `last_action` on purpose, so this penalty prices what the policy
        ASKS for and unbounded output growth is not free. But the reachable
        magnitude of that is `(2 * clip) ** 2`, so raising the clip from the
        body's 4.0 to the 13.0 a get-up needs multiplied this penalty's ceiling
        by **10.6x** — and a 5 M-step run duly found the corner: `ep_rew` was
        +396 at 2.9 M steps and **-689** by 4.3 M, with
        `action_rate_penalty` at **-694** against ~130 of task reward. Every
        checkpoint scored 0 of 12. The penalty ate the task, which is a failure
        this repo has already paid for once on the G1 idle.

        Anchored on the BODY's own clip, not on 1.0: `W_ACTION_RATE` was
        calibrated against `RobotSpec.action_clip` (4.0), so dividing by the
        live clip alone would leave the penalty 169x weaker at 13.0 instead of
        clip-independent — a recalibration disguised as a fix. Scaling by
        `nominal / live` keeps it bit-identical at 4.0 and gives it the same
        ceiling at any wider clip.
        """
        nominal = float(self.robot.action_clip)
        d = ((self.last_action - self.prev_action)
             * (nominal / max(self._act_clip, 1e-9)))
        return float(np.square(d).mean())

    def _head_gain(self) -> float:
        """New head height above the best this episode, as a fraction of
        standing. Zero for a posture that is merely HELD."""
        h = min(max(float(self.data.xpos[self._head_body_id][2])
                    / self._head_stand_z, 0.0), 1.0)
        best = getattr(self, "_best_head", None)
        if best is None:
            self._best_head = h
            return 0.0
        gain = max(h - best, 0.0)
        self._best_head = max(best, h)
        return gain

    def _trunk_up_speed(self) -> float:
        """How fast the trunk is reaching a NEW BEST height, m/s.

        Not `cvel`: that is the body's own frame velocity and this term is
        about the trunk rising in the WORLD, which is the quantity the height
        bands above were measured in.
        """
        z = float(self.data.xpos[self.trunk_body_id][2])
        best = getattr(self, "_best_trunk_z", None)
        if best is None:
            self._best_trunk_z = z
            return 0.0
        # Only motion that takes the trunk ABOVE its own best this episode.
        # Plain per-step up-speed was farmable and measurably so: bouncing
        # scored 32.5 and RANDOM ACTIONS 35.4 against a real get-up's 57.2 —
        # 60 % of the reward for flailing, which is the kind of thing PPO finds
        # in a hundred thousand steps. Monotone height cannot be farmed by
        # oscillation: after the first bounce nothing exceeds the best again.
        gain = max(z - best, 0.0)
        self._best_trunk_z = max(best, z)
        return gain * 50.0

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        pelvis_z = float(self.data.xpos[self.trunk_body_id][2])
        upright = float(-self._projected_gravity()[2])       # +1 standing
        height = min(max(pelvis_z / self._stand_z, 0.0), 1.0)
        contacts = self._foot_contacts()
        feet = float(sum(bool(v) for v in contacts.values())) / 2.0

        up = (height >= self.UP_HEIGHT_FRAC and upright >= self.UP_GRAVITY
              and feet >= 1.0)
        joint_vel = float(np.square(self._joint_vel()).mean())
        rate = self._action_rate()
        arm_effort = float(np.abs(
            self.data.actuator_force[self.act_adr[self._arm_ids]]).mean()) \
            if hasattr(self, "act_adr") else float(np.abs(
                self.data.actuator_force[self._arm_ids]).mean())

        rise = height * max(upright, 0.0)
        stand = rise * feet
        # Paid for exceeding the best RISE this episode reached, and never
        # again for the same value, so it cannot be farmed by bouncing.
        gain = max(rise - self._best_rise, 0.0)
        self._best_rise = max(self._best_rise, rise)
        self._peak_rise = max(self._peak_rise, rise)

        # The latches. `up` is computed here and nowhere else, so this is where
        # they belong; `step` reads them to decide the cut and the info.
        if up:
            self._stood = True
            self._held += 1
            self._held_best = max(self._held_best, self._held)
        else:
            self._held = 0

        pen = self._penalty_scale()
        terms = {
            "stand": self.W_STAND * stand,
            "rise": self.W_RISE * rise,
            "progress": self.W_PROGRESS * gain,
            "standing": self.W_STANDING * float(up),
            "joint_vel_penalty": -self.W_JOINT_VEL * joint_vel * float(up) * pen,
            "action_rate_penalty": -self.W_ACTION_RATE * rate * pen,
            "arm_effort_penalty": -self._w_arm_effort * arm_effort * pen,
            "over_clip_penalty": -self.W_OVER_CLIP * float(np.square(
                np.maximum(np.abs(self.last_action) - self._act_clip, 0.0)
            ).mean()),
            # PUSHING UP THROUGH AN ARM: load carried x upward trunk speed.
            #
            # Measured on the working get-up, by height band (fraction of
            # standing), this is what the arms are FOR:
            #
            #   | height    | arm load | trunk dz/dt |
            #   |-----------|----------|-------------|
            #   | 0.00-0.20 |   0.556  |    +0.14    |
            #   | 0.20-0.35 |   0.526  |    +0.15    |
            #   | 0.35-0.50 |   0.594  |    +1.01    |  <- the launch
            #   | 0.50-0.70 |   0.037  |    +0.79    |
            #   | 0.70+     |   0.000  |    +0.81    |
            #
            # The arms carry over half the body weight from the floor to half
            # standing height, do the launch, and then let go entirely — the
            # legs finish. So the integral worth paying is load x UP-SPEED,
            # which is dense through the whole first half of the motion. The
            # first version of this term paid `gain` (new-best height) instead
            # and scored 0.20 a episode against a 356 return: correct in shape,
            # far too weak to steer anything, because most height is won on the
            # legs AFTER the arms release.
            "arm_push": (self._w_arm_push * min(self._arm_load(), 1.0)
                         * max(self._trunk_up_speed(), 0.0)),
            # HEAD OFF THE FLOOR — paid on a NEW BEST only, never per step.
            #
            # The per-step version was measured and it reproduced run be7733
            # exactly, which this class's own docstring warns about: it earned
            # **253.8 of a 396.6 return (64 %)** while standing 0 of 12,
            # because a head held high with the hips on the floor is a posture
            # you can hold for the whole episode. Sitting up is worth paying
            # for ONCE; it is not worth paying for 400 times.
            "head_up": self._w_head_up * self._head_gain() * max(upright, 0.0),
        }
        return float(sum(terms.values())), terms

    #: THE PENALTIES RAMP IN. Measured on `teach-asimov_getup-4c5c79`, 14.7 M
    #: steps at full weight from step one, reading the run's own reward log:
    #:
    #: | steps | task pays | penalties | |
    #: |---|---|---|---|
    #: | 43 k | 11.0 | **-62.4** | penalties are **5.7x** the reward |
    #: | 3.0 M | 23.8 | -36.4 | 1.5x |
    #: | 12.0 M | 41.9 | -13.8 | 0.3x |
    #:
    #: At initialisation the steepest gradient available is "stop moving and
    #: stop straining your arms", and that is what it learned: action-rate
    #: -30.0 -> -4.4, arm effort -32.4 -> -9.4, episode reward -29 -> +28 almost
    #: entirely by SHEDDING PENALTY rather than by rising. `standing` — the
    #: bonus for being up, upright and on both feet — paid **0.0 for all 14.7 M
    #: steps**: never sampled, never paid. The policy got up 0 of 24 times, the
    #: same as the donor it warm-started from.
    #:
    #: 0.05 is not a taste: at the measured init budget it puts the penalties at
    #: 11.0 x 0.05 x 62.4 / 11.0 = 28 % of what the task pays, so shedding them
    #: can no longer outrank finding the task. Full weight by 100 k steps PER
    #: ENV (3.2 M total at 32 envs), by which point the behaviour has to exist
    #: for the penalties to shape. The shape is `walk_env._ACTION_RATE_STAGES`,
    #: which this env never used — see AGENTS.md, "a ramped penalty can eat its
    #: task", measured before on the G1 idle (ep_rew -92 and a face-plant).
    _PENALTY_STAGES = (
        (0, 0.05), (20_000, 0.15), (40_000, 0.35),
        (70_000, 0.60), (100_000, 1.00),
    )

    def _penalty_scale(self) -> float:
        """How much of the shaping penalties apply yet, by per-env lifetime."""
        n = getattr(self, "_lifetime_steps", 0)
        w = self._PENALTY_STAGES[0][1]
        for step, wt in self._PENALTY_STAGES:
            if n >= step:
                w = wt
        return w

    # ------------------------------------------------------------- the cut

    def step(self, action):
        """`AsimovWalkEnv.step`, plus a fall cut that only fires once it HAS
        stood, and the two metrics that say whether it stayed.

        The inherited cut is off (see `__init__`) because the episode begins
        fallen. Leaving it at that was the fourth fault of the first four
        runs: an 8 s episode that toppled at 0.34 s spent 96 % of its steps
        lying on the floor still collecting `rise` and `stand` at whatever the
        floor pays, so NO return distinguished a fall at 0.34 s from one at
        7 s. Arming on `_stood` is the honest version — an episode that never
        stands cannot fall, and one that stands and then goes down loses the
        rest of its episode, which is the only thing that makes STAYING up
        worth more than reaching it.

        `held_s` is the longest unbroken time `up`, not the total: `standing`
        already sums the gate, and a sum cannot tell 40 scattered frames from
        one 0.8 s hold.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        if self._stood and not terminated:
            upright = float(-self._projected_gravity()[2])
            height = float(self.data.xpos[self.trunk_body_id][2]) / self._stand_z
            if upright < self.FELL_GRAVITY or height < self.FELL_HEIGHT_FRAC:
                terminated = True
        if terminated or truncated:
            self._write_episode_info(info)
        return obs, reward, terminated, truncated, info

    def _write_episode_info(self, info: dict) -> None:
        """The end-of-episode metrics, as a method a SUBCLASS can call.

        `AsimovGetupRefEnv` adds cuts of its own, which fire after this class
        has already decided whether the episode ended — so an episode it ended
        carried no `held_s` at all and the reader raised `KeyError` instead of
        reporting a number. Writing them here means every cut, inherited or
        added, leaves the same record.
        """
        from .. import contract as C
        dt = float(C.CTRL_DT)
        info["held_s"] = round(self._held_best * dt, 3)
        info["peak_rise"] = round(float(self._peak_rise), 4)
        info["stood"] = bool(self._stood)


class AsimovGetupRefEnv(AsimovGetupEnv):
    """Get up by TRACKING the authored reference — roadmap phase 3d.

    `AsimovGetupEnv`'s reward is built from pelvis height, uprightness and foot
    contact, and four runs found four ways to stop at the first phase. The
    measurement that explains it (`robots/asimov_getup_ref`): on this body both
    of those axes are **non-monotone along a correct get-up**. `u_z` runs +0.01
    at `prone` to **-0.37** at `pike` to +1.00 at `stand`, because Asimov has to
    turn FACE DOWN to get its forearms under it — it has no hands and the
    forearms are the only load-bearing surface on the arm. `height` dips too, at
    `plant_forearms`, because loading the elbows is the move. Any weighting of
    two non-monotone axes pays the robot to skip the phases that lift it.

    So the reward here measures agreement with a reference instead: where the
    joints should be at this phase, and which bodies should be touching the
    floor. `imitation-needs-not-tracking-termination` is the precedent — on the
    G1 kick the reward alone stood through every attempt and the DeepMimic
    not-tracking cut on a 5 -> 15 -> 30 cm ladder took it from 0.10 m to 0.62 m.
    **The ladder is the lever, not the pay.**

    **The phase is REWARD-SIDE ONLY.** `_get_obs` is untouched: the 78 floats
    stay exactly Menlo's layout and `obs[6:9]` stays zero, because the whole
    point of this task is a get-up their deployment can run. The phase index,
    the reference pose and the target contact set are privileged simulator
    state used to score the policy, never to inform it — the policy reads the
    phase off its own proprioception, which is nearly sufficient here since the
    joint configuration almost determines where in the motion it is. Putting a
    phase into the three free command slots is the fallback if that proves too
    hard, and it costs a deployment change.

    Three knobs, all recorded onto the run by `AsimovBody.train_env_kwargs`:

    `start_phase`  which waypoint the episode resets at. **This is the real
                   curriculum**, and unlike the retired blend ladder every rung
                   is a configuration measured to hold for 6 s. Start at `pike`
                   — the last pose before the one leg the servos cannot do —
                   and extend backwards.
    `track_tol`    the not-tracking cut, in radians of mean joint error. Loose
                   to tight is the ladder.
    `ref_speed`    how fast the reference plays. Slower is easier to track and
                   further from the 1.70 s median Menlo report.
    """

    #: Tracking, as `exp(-err / TRACK_STD)` on the mean absolute joint error —
    #: bounded in (0, 1] so it cannot be farmed by a pose that happens to score
    #: well on one joint, and dense everywhere, which is what the product
    #: reward of run 990c5d could not be.
    W_TRACK = 10.0
    TRACK_STD = 0.25
    #: Tracking is a PRODUCT of joint agreement and BODY agreement, and the
    #: body half is not optional on this robot.
    #:
    #: Measured 2026-10-04: the failure state and the target have **the same
    #: joint angles**. `fold` standing sits at pelvis 0.516 m, u +0.84; the same
    #: joint command, after the robot has fallen onto its forearms, sits at
    #: 0.197 m, u -0.35 — mean joint difference **0.0516 rad**. The joints do
    #: not determine the pose here, the CONTACT MODE does, and a joint-only
    #: tracking reward therefore paid `exp(-0.0516/0.25) = 0.81`, **81 % of full
    #: marks for lying face down**. Worse, pushing off the forearms means
    #: EXTENDING the elbows away from the target pose, so the reward actively
    #: charged for the one move that works. Six training chains against that.
    #:
    #: A product cannot be farmed by the joints alone: in the fallen state the
    #: height term is exp(-0.32/0.15) = 0.12 and the uprightness term
    #: exp(-1.18/0.40) = 0.05, so the whole thing collapses to ~0.005. Same
    #: shape, same reason, as the products that killed the sit and the shoulder
    #: stand in phase 3.
    TRACK_Z_STD = 0.15
    TRACK_U_STD = 0.40
    #: Agreement with the phase's measured contact set, as a Jaccard overlap.
    #: `quadruped` means ankles AND knees AND elbows; a robot in roughly the
    #: right shape with nothing touching is not in the phase.
    W_CONTACT = 2.0
    #: One-way, per waypoint, never paid twice — the progress-pay shape that
    #: solved the duck headstand. This is what makes getting FURTHER worth more
    #: than tracking the early phases nicely forever.
    W_ADVANCE = 30.0
    #: The terminal pay, once it is up and on its feet, per step.
    W_UP = 8.0

    #: Mean joint error that counts as having REACHED a waypoint. Looser than
    #: the cut, so a phase advances before the episode is at risk.
    REACH_TOL = 0.30
    #: And the body has to be there too — the measured pelvis height and
    #: uprightness of that waypoint, within these.
    #:
    #: **Not optional, and the first version of this env proved it.** With the
    #: clock advancing on JOINT error alone, a ZERO action — which commands
    #: `DEFAULT_POSE`, which IS the stand pose — walked the phase from `pike`
    #: all the way to `stand` with `reached_stand` true, lying on the floor the
    #: whole time at pelvis 0.10 m. The joints matched; the robot did not. It is
    #: the same family as the sit and the shoulder stand that beat the first two
    #: get-up rewards: a proxy that failure also satisfies.
    REACH_Z_TOL = 0.06
    REACH_U_TOL = 0.25
    #: Having ACHIEVED a waypoint, falling this far below its pelvis height
    #: ends the episode.
    #:
    #: **The cut on joint error alone was not enough, and the first chain shows
    #: why.** A policy reached `rock_back` at step 13 and then lay on the floor
    #: for the remaining 7.7 s at pelvis 0.099 m — with a joint error of 0.326
    #: against a tolerance of 0.60, so it counted as TRACKING the whole time and
    #: collected `track` and `contact` every step. Falling was nearly free,
    #: which is the identical structural fault the original get-up env had and
    #: that `AsimovGetupEnv.step` was written to fix. Measured from the ACHIEVED
    #: waypoint rather than from standing, because the reference's height is not
    #: monotone — `plant_forearms` DROPS the pelvis on purpose.
    CUT_BACK_Z = 0.15
    #: ...measured from an ANCHOR the episode observes, not from the authored
    #: number. Keying it to `settled_z` and then settling the reset separately
    #: made the two disagree: the `pike` rung reset to 0.264 m against an
    #: authored 0.300, so the floor sat 24 mm below the start and **every
    #: episode died in 2 steps** — 5 at the quadruped, 20 at `draw_knees_in`.
    #: A chain trained on it came out WORSE than the chain before, because no
    #: policy learns a get-up in two control steps. The anchor is where the
    #: robot actually was: the reset height, then the height at each waypoint
    #: it achieves.
    #:
    #: And the margin is 0.15 m rather than 0.06, because anchoring alone did
    #: not fix it — 0.06 m is three control steps of free fall, so the hard
    #: rungs still died before a policy could do anything. This cut is no
    #: longer the main defence against lying on the floor for free income: the
    #: product tracking pays that state 0.91 per step against a possible 10,
    #: and the stall deadline ends it at 2 s. Those two are worth ~91 for an
    #: episode spent lying down, against ~30 for one waypoint of `advance` PLUS
    #: a reset stall clock and the steps that buys. The cut only has to stop
    #: the 8-second version.
    #: A waypoint HIGHER than the last one achieved only counts if the pelvis
    #: is not falling through it.
    #:
    #: The gap waypoint is a state recorded off `fold` FALLING, so a robot on
    #: its way down passes through it by definition — measured: commanding the
    #: `pike` pose and doing nothing else "achieves" `rock_back` at step 18,
    #: while dropping from the reset placement at 0.527 m through 0.421 m on
    #: its way to settling at 0.300 m. Same family as the `up` gate brushed by
    #: a decaying reset pose, which is what `standing` 2.19 turned out to be.
    REACH_MIN_VZ = -0.10
    #: Seconds allowed on ONE target before the episode ends.
    #:
    #: The phase advances on ACHIEVEMENT and not on a clock, so that a slow
    #: get-up is fine — but achievement alone means the reference never
    #: PRESSES, and the measurement says that is an exploit at every rung:
    #: the mean joint distance between consecutive waypoints is 0.087 to 0.239
    #: rad, every one of them inside even the tightest cut. So STANDING STILL
    #: counts as tracking the next waypoint, and since `pike` holds for 6 s on
    #: its own, "hold the pike for the whole episode" collects `track` and
    #: `contact` every step and never falls. A deadline bounds "slow is fine"
    #: instead of removing it.
    PHASE_BUDGET_S = 2.0

    START_ENV = "MICRODUCK_ASIMOV_GETUP_START"
    TOL_ENV = "MICRODUCK_ASIMOV_GETUP_TRACK_TOL"
    SPEED_ENV = "MICRODUCK_ASIMOV_GETUP_REF_SPEED"
    #: Force the one-step command delay ON every episode, instead of the
    #: inherited 50/50 draw.
    #:
    #: The rung that stands learned a controller with NO delay margin. Measured
    #: on two independently trained chains: every episode WITHOUT the delay
    #: holds the full 7.76 s, and every episode WITH it falls at 1.3-2.0 s —
    #: 11/16 and 5/16, perfectly separated, no exceptions. One control step is
    #: 20 ms, which is the CAN-bus latency Menlo's own randomisation models, so
    #: a stand that needs zero delay is not a stand that ships.
    #:
    #: Training it 50 % of the time was not enough: from a single 78-float
    #: frame the policy cannot TELL whether it is lagged — `last_action` is what
    #: it asked for, not what was applied — so it has to use one controller for
    #: both, and it settled on the one that is optimal with no delay. Forcing
    #: the delay removes the choice. A controller with margin at 20 ms is stable
    #: at 0 ms; the reverse is what has been failing.
    LAG_ENV = "MICRODUCK_ASIMOV_GETUP_LAG"
    #: FREE mode: stop imitating the reference and just pay for getting up.
    #:
    #: Every rung below `fold` has been training the policy to TRACK the
    #: authored route `pike` -> `rock_back` -> `fold`, and that route has since
    #: been measured impossible: `com - foot` is 0.618 m at the pike against
    #: 0.114 at `fold`, it does not move while four contacts are planted, and
    #: the trunk cannot rotate up first (recovery boundary 33 deg). Five chains
    #: were spent asking the policy to copy a path that does not exist, and a
    #: tracking reward against an unreachable waypoint actively charges it for
    #: trying anything else.
    #:
    #: In free mode the reference is dropped for the intermediates and the pay
    #: is the PRODUCT `pelvis_z x uprightness`, counted only while a foot is on
    #: the floor — the shape that killed the sit, the shoulder stand and the
    #: somersault, because no single axis satisfies it. The waypoints still
    #: serve as the `advance` ladder and the phase clock, so the structure is
    #: kept and only the imitation is let go.
    FREE_ENV = "MICRODUCK_ASIMOV_GETUP_FREE"
    #: What the product rise is worth per step in free mode.
    W_FREE_RISE = 12.0

    #: How many past observations the policy sees, stacked. 1 is the contract.
    #:
    #: **Anything above 1 BREAKS the 78/23 contract and is a CONTROL ARM, not a
    #: deliverable.** It exists to answer the question this track wrote down as
    #: phase 2 and then skipped for six training chains: is a single 78-float
    #: frame SUFFICIENT to tell the phases of a get-up apart? A get-up is a
    #: sequence in which the right action depends on where in the motion you
    #: are, and from one frame two phases with similar joint angles are nearly
    #: indistinguishable — so no reward and no curriculum can fix it if the
    #: answer is no. The one community get-up that reportedly works uses a
    #: 5-step history (375 inputs), which is the clue that sat unread in this
    #: track's own notes while four reward runs and two reference chains ran.
    #:
    #: A policy trained with this > 1 cannot be hot-swapped onto the robot and
    #: will not load in Menlo's viewer. That is the POINT of the measurement: if
    #: history crosses the leg and 78-obs does not, the contract-preserving
    #: version is not possible as specified, and a negative result is the
    #: contribution.
    HISTORY_ENV = "MICRODUCK_ASIMOV_GETUP_HISTORY"

    def __init__(self, *args, start_phase: str | None = None,
                 track_tol: float | None = None,
                 ref_speed: float | None = None,
                 history: int | None = None,
                 force_lag: bool | None = None,
                 free: bool | None = None, **kwargs):
        from . import asimov_getup_ref as REF

        super().__init__(*args, **kwargs)
        self._history = max(1, int(history if history is not None
                                   else os.environ.get(self.HISTORY_ENV, "1") or 1))
        if self._history > 1:
            import gymnasium as gym
            w = int(self.robot.obs_dim) * self._history
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (w,), np.float32)
        self._obs_stack: list[np.ndarray] = []
        self._ref = REF
        # A COMMA-SEPARATED LIST, sampled per episode — not one rung.
        #
        # Training each rung only on its newest start state made the curriculum
        # destroy the skill it began with. Measured on `teach-asimov_getup_ref-
        # d17563`, every stage scored FROM `fold`:
        #
        #   s1 (fold)           median hold 7.76 s
        #   s2 (rock_back)                  3.30 s
        #   s3 (pike)                       0.44 s
        #   s5 (draw_knees_in)              0.32 s
        #   s8 (prone)                      0.24 s
        #
        # Every stage still REACHES standing 12/12 — it has not forgotten how
        # to get there — but the hold collapses 94 % by the third rung and
        # never comes back, because after s1 the policy never practises
        # standing again. The final policy of a 2.7 M-step chain holds a
        # quarter of a second. Catastrophic forgetting, in the plainest form.
        #
        # So a rung is now a SET: "fold,rock_back,pike" practises all three,
        # and the curriculum extends the set rather than replacing it.
        raw = str(start_phase or os.environ.get(self.START_ENV) or "pike")
        self._start_phases = [x.strip() for x in raw.split(",") if x.strip()]
        unknown = [x for x in self._start_phases if x not in REF.BY_NAME]
        if unknown or not self._start_phases:
            raise ValueError(
                f"{unknown or raw!r} is not a waypoint — have "
                f"{', '.join(w.name for w in REF.WAYPOINTS)}")
        self._start_phase = self._start_phases[-1]
        self._track_tol = float(track_tol if track_tol is not None
                                else os.environ.get(self.TOL_ENV, "0.60") or 0.60)
        self._ref_speed = float(ref_speed if ref_speed is not None
                                else os.environ.get(self.SPEED_ENV, "1.0") or 1.0)
        self._force_lag = bool(int(force_lag if force_lag is not None
                                   else os.environ.get(self.LAG_ENV, "0") or 0))
        self._free = bool(int(free if free is not None
                              else os.environ.get(self.FREE_ENV, "0") or 0))
        self._foot_geoms = [g for g in range(self.model.ngeom)
                            if int(self.model.geom_contype[g])
                            and self.model.geom_bodyid[g]
                            and "ankle_roll" in self.model.body(
                                int(self.model.geom_bodyid[g])).name]
        _names = [w.name for w in REF.WAYPOINTS]
        self._start_idxs = [_names.index(x) for x in self._start_phases]
        self._start_idx = self._start_idxs[-1]
        #: Running mean episode LENGTH per rung, and the length of the episode
        #: in progress — the two numbers the start sampler needs.
        #:
        #: Drawing rungs uniformly balances EPISODES, not STEPS, and on this
        #: task those are wildly different things. Measured on the first
        #: cumulative chain: `fold` episodes run to the 400-step cap while
        #: `rock_back` and `pike` die in 6-8, so a uniform draw gave
        #: `fold` **58.6 %** of every training step and the two rungs the
        #: curriculum exists for **0.9 % and 1.2 %**. The fix for forgetting
        #: starved the thing it was meant to enable. Sampling inversely to
        #: episode length equalises the STEP share instead, and it self-tunes:
        #: as a hard rung starts surviving longer it is automatically drawn
        #: less.
        self._rung_len = {i: 30.0 for i in self._start_idxs}
        self._ep_steps = 0
        #: The placed-and-settled start `qpos`, computed once. See `reset`.
        self._start_cache: dict = {}
        #: Every per-episode latch also gets a value HERE, not only in `reset`.
        #: `_on_target` lived only in `reset`, so any caller that placed a state
        #: and stepped without resetting first — a probe, a scripted rollout —
        #: died on `AttributeError` inside `_compute_reward`. An env should be
        #: steppable from construction.
        self._on_target = 0
        self._free_best = 0.0
        self._ep_steps = 0
        self._reached = self._start_idx
        self._anchor_z = 0.0
        self._stood = False
        self._held = 0
        self._held_best = 0
        self._peak_rise = 0.0
        self._obs_stack = []
        self._phase = self._start_idx
        #: The furthest waypoint ACHIEVED, which is NOT the furthest targeted.
        #: Reporting the target cost a false result: the first chain was read as
        #: "reached `fold` 16/16", crossing the one leg the servos cannot do,
        #: when what it had done was achieve `rock_back` — where the servos stop
        #: unaided — and advance the clock to aim at `fold`. An off-by-one in
        #: the metric, and it inverted the conclusion.
        self._reached = self._start_idx
        #: Joint addresses in JOINT_NAMES order, for the tracking error.
        self._q_adr = np.array([int(self.model.joint(n).qposadr[0])
                                for n in W.JOINT_NAMES])
        self._floor_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # ------------------------------------------------------------- reset

    def reset(self, *, seed=None, options=None):
        """Reset AT a waypoint, rather than at a blend of two poses.

        `prone` draws from the pose cache — the real distribution, many poses.
        Every other waypoint is a single authored configuration, which is the
        price of a rung that is measured to hold.
        """
        obs, info = super().reset(seed=seed, options=options)
        if self._force_lag:
            self._action_lag = 1
        # Draw THIS episode's rung from the set, so earlier ones keep being
        # practised instead of being forgotten.
        if len(self._start_idxs) > 1:
            if self._ep_steps > 0:          # book the episode that just ended
                prev = self._rung_len.get(self._start_idx, 30.0)
                self._rung_len[self._start_idx] = 0.9 * prev + 0.1 * self._ep_steps
            w = np.array([1.0 / max(self._rung_len[i], 1.0)
                          for i in self._start_idxs])
            w /= w.sum()
            self._start_idx = int(self._rng.choice(self._start_idxs, p=w))
            self._start_phase = self._ref.WAYPOINTS[self._start_idx].name
        self._ep_steps = 0
        w = self._ref.WAYPOINTS[self._start_idx]
        if self._start_phase != "prone":
            # CACHED. Placing and settling an authored waypoint is deterministic
            # — the same pose, the same floor, every episode — and it cost
            # **655 ms per reset**: 535 ms in the 0.6 s settle loop (120
            # `mj_step`) and 120 ms placing. Against episodes of 6 to 100 steps
            # at the hard rungs that is most of the compute, and it is what took
            # the lab from 788 to 99 steps/s at 37 % CPU. A trainer at 37 % CPU
            # is waiting, not throttling.
            #
            # The trade, stated: under domain randomisation the settled state
            # would otherwise differ slightly per episode, and this freezes one.
            # For a reverse-curriculum rung that is arguably the better choice —
            # every episode starts from the same measured configuration — but it
            # IS a choice, and a rung that wants start variety should draw from
            # a cache of settled states the way `prone` draws from the pose
            # cache, not pay for a fresh settle each time.
            if self._start_phase not in self._start_cache:
                self._place(w.joints, w.root_pitch)
                act = ((np.asarray(w.joints) - np.asarray(W.DEFAULT_POSE))
                       / W.ACTION_SCALE).astype(np.float32)
                act = act.clip(-self._act_clip, self._act_clip)
                if w.equilibrium:
                    for _ in range(int(0.6 * 50)):
                        self.data.xfrc_applied[:] = 0.0
                        self._write_ctrl(self.robot.scale_action(act))
                        for _ in range(C_DECIMATION):
                            mujoco.mj_step(self.model, self.data)
                self._start_cache[self._start_phase] = self.data.qpos.copy()
            else:
                self.data.qpos[:] = self._start_cache[self._start_phase]
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.trunk_body_id, 2] = self._assist_n
        mujoco.mj_forward(self.model, self.data)
        #: Where the fall-back cut is measured from — observed, not authored.
        self._anchor_z = float(self.data.xpos[self.trunk_body_id][2])
        self.twist_cmd[:] = 0.0
        # The clock aims at the waypoint AFTER the start, because the start is
        # where the robot already is. Scoring it against its own start pose
        # made phase 0 trivially achieved on step 1, which spent an `advance`
        # bonus for free and then raised the bar in the same instant. And the
        # `prone` rung cannot be scored against the authored `prone` pose at
        # all — it draws from the CACHE, and a cache pose sits 0.78 rad of mean
        # joint error from the authored one, past any usable cut, so every
        # episode terminated on its first step for being a different fallen
        # robot. Where it starts is the task, not an error.
        self._phase = min(self._start_idx + 1, len(self._ref.WAYPOINTS) - 1)
        self._reached = self._start_idx
        self._on_target = 0
        self._free_best = 0.0
        # Cleared so the first frame of an episode is not stacked with the last
        # frame of the previous one — a leak that would hand the policy the
        # outcome of the episode before it.
        self._obs_stack = []
        self._stood = False
        self._held = 0
        self._held_best = 0
        self._peak_rise = 0.0
        return self._get_obs(), info


    # -------------------------------------------------------- the phase clock

    def _joint_err(self, target) -> float:
        """Mean absolute joint error against a reference pose, radians."""
        return float(np.abs(self.data.qpos[self._q_adr] - np.asarray(target)).mean())

    def _get_obs(self) -> np.ndarray:
        """The contract's 78 floats, or `history` of them stacked oldest-first.

        At `history == 1` this is bit-identical to the parent — the contract arm
        must not be perturbed by the existence of the control arm.
        """
        obs = super()._get_obs()
        if self._history <= 1:
            return obs
        if not self._obs_stack:
            self._obs_stack = [obs.copy() for _ in range(self._history)]
        else:
            self._obs_stack.pop(0)
            self._obs_stack.append(obs.copy())
        return np.concatenate(self._obs_stack).astype(np.float32)

    def _target_of(self, waypoint):
        """`(joints, pelvis_z, u_z)` to score this waypoint against."""
        return waypoint.joints, waypoint.settled_z, waypoint.settled_u

    def _touching(self) -> set[str]:
        m, d = self.model, self.data
        out: set[str] = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if self._floor_geom >= 0 and self._floor_geom not in (c.geom1, c.geom2):
                continue
            for g in (c.geom1, c.geom2):
                b = int(m.geom_bodyid[g])
                if b:
                    out.add(m.body(b).name.replace("_link", "")
                            .replace("left_", "").replace("right_", ""))
        return out

    # ------------------------------------------------------------ reward

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        ref = self._ref
        target = ref.WAYPOINTS[self._phase]
        t_joints, t_z, t_u = self._target_of(target)
        err = self._joint_err(t_joints)

        pelvis_z = float(self.data.xpos[self.trunk_body_id][2])
        upright = float(-self._projected_gravity()[2])

        # ADVANCE the clock on ACHIEVEMENT, never on time: a slow get-up is
        # fine, and a skipped phase pays nothing because the next phase's error
        # is measured from where the robot actually is. Achievement is the
        # joints AND the body — see REACH_Z_TOL.
        advanced = 0
        # ACHIEVING a waypoint and ADVANCING the target are two things, and
        # conflating them made the last waypoint unreachable: the advance was
        # gated on `_phase < len - 1`, so when the target WAS `stand` the gate
        # was false and `stand` could never be recorded as reached. Every
        # stage of the first chain reported `reached_stand` 0/16, and it was
        # not achievable. From `crouch` a zero action sits at pelvis 0.608 m —
        # standing, by every number the table records — and scored nothing.
        at_target = (err < self.REACH_TOL
                     and abs(pelvis_z - t_z) < self.REACH_Z_TOL
                     and abs(upright - t_u) < self.REACH_U_TOL)
        if at_target and t_z > ref.WAYPOINTS[self._reached].settled_z + 1e-9:
            # climbing to it, not dropping through it
            at_target = float(self.data.cvel[self.trunk_body_id][5]) > self.REACH_MIN_VZ
        if at_target and self._phase > self._reached:
            advanced = self._phase - self._reached
            self._reached = self._phase
            self._anchor_z = max(self._anchor_z, pelvis_z)
        if at_target and self._phase < len(ref.WAYPOINTS) - 1:
            self._phase += 1
            target = ref.WAYPOINTS[self._phase]
            t_joints, t_z, t_u = self._target_of(target)
            err = self._joint_err(t_joints)

        # ONE-SIDED on the body terms: charged for being BELOW the waypoint's
        # height or less upright than it, never for being past it. Two-sided,
        # `fold` scored WORSE than `pike` while the target was the waypoint
        # between them — the reward punished overshooting toward the goal,
        # which on a rise is the one direction that cannot be wrong. The JOINT
        # term stays two-sided: a joint has a right value, a height does not.
        track = float(np.exp(-err / self.TRACK_STD)
                      * np.exp(-max(0.0, t_z - pelvis_z) / self.TRACK_Z_STD)
                      * np.exp(-max(0.0, t_u - upright) / self.TRACK_U_STD))
        want = set(target.contacts)
        got = self._touching()
        contact = (len(want & got) / len(want | got)) if (want | got) else 1.0

        height = min(max(pelvis_z / self._stand_z, 0.0), 1.0)
        feet = sum(bool(v) for v in self._foot_contacts().values()) / 2.0
        up = (height >= self.UP_HEIGHT_FRAC and upright >= self.UP_GRAVITY
              and feet >= 1.0)
        if up:
            self._stood = True
            self._held += 1
            self._held_best = max(self._held_best, self._held)
        else:
            self._held = 0
        self._peak_rise = max(self._peak_rise, height * max(upright, 0.0))

        rate = self._action_rate()
        # The arm penalty is GONE from the reward and that is deliberate. Run
        # 3b's version charged the robot for "leaning on the arms more than it
        # needs to", and on this body `quadruped` and `pike` cannot be reached
        # without the elbows at 58-66 % of rated torque. It was charging it for
        # the only phase that lifts it. The torque envelope is a phase-4
        # measurement, not a reward term.
        if self._free:
            # The PRODUCT, and only with a foot down — height alone bought a
            # somersault in a trajectory search, airborne and inverted.
            foot_down = any(
                self.data.contact[i].geom1 in self._foot_geoms
                or self.data.contact[i].geom2 in self._foot_geoms
                for i in range(self.data.ncon))
            # `(u + 1) / 2` and not `max(u, 0)`: with the raw clamp the term is
            # exactly 0.00 at `prone`, `plant_forearms`, `draw_knees_in`,
            # `quadruped` AND `pike` — every rung below the gap — because
            # uprightness is negative through all of them. A reward that is
            # flat at zero across the whole region the policy starts in has no
            # gradient, which is what run 990c5d already proved the expensive
            # way. Mapped to [0, 1] it rises monotonically from the floor:
            # pike 1.8, rock_back 3.7, fold 8.9, stand 11.4 per step.
            track = ((pelvis_z / self._stand_z) * (upright + 1.0) * 0.5
                     if foot_down else 0.0)
            w_track = self.W_FREE_RISE
        else:
            w_track = self.W_TRACK
        # THE DEADLINE MEASURES PROGRESS, not waypoint-hitting, in free mode.
        # Dropping the tracking reward but keeping a stall cut that fires
        # unless the next AUTHORED waypoint is reached within 2 s would still
        # force the policy down the route measured impossible — it would just
        # stop paying it for the trip. In free mode the clock resets whenever
        # the best rise improves, so the requirement is "keep getting higher
        # and more upright", which is the whole task and says nothing about HOW.
        if self._free:
            gained = track > self._free_best + 1e-4
            self._free_best = max(self._free_best, track)
            self._on_target = 0 if (at_target or gained) else self._on_target + 1
        else:
            self._on_target = 0 if at_target else self._on_target + 1

        terms = {
            "track": w_track * track,
            "contact": self.W_CONTACT * contact,
            "advance": self.W_ADVANCE * float(advanced),
            "up": self.W_UP * float(up),
            "action_rate_penalty": -self.W_ACTION_RATE * rate,
        }
        self._track_err = err
        return float(sum(terms.values())), terms

    # ------------------------------------------------------------- the cut

    def step(self, action):
        """The not-tracking cut — the ladder that is actually the curriculum.

        `AsimovGetupEnv.step`'s fall cut stays (it ends an episode that stood
        and went down). This adds the DeepMimic one: an episode whose tracking
        error leaves `track_tol` is over. On the G1 kick that cut, laddered, is
        what moved the result; the reward alone did nothing.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        self._ep_steps += 1
        if (not terminated and not self._free
                and getattr(self, "_track_err", 0.0) > self._track_tol):
            terminated = True
        if not terminated:
            floor = getattr(self, "_anchor_z", 0.0) - self.CUT_BACK_Z
            if float(self.data.xpos[self.trunk_body_id][2]) < floor:
                terminated = True
        # The stall deadline — except on the LAST waypoint, where staying put
        # is the task.
        if (not terminated
                and self._reached < len(self._ref.WAYPOINTS) - 1
                and self._on_target > self.PHASE_BUDGET_S * 50):
            terminated = True
        if terminated or truncated:
            self._write_episode_info(info)
            info["phase"] = int(self._reached)
            info["phase_name"] = self._ref.WAYPOINTS[self._reached].name
            info["targeting"] = self._ref.WAYPOINTS[self._phase].name
            info["track_err"] = round(float(getattr(self, "_track_err", 0.0)), 4)
            info["reached_stand"] = bool(
                self._reached == len(self._ref.WAYPOINTS) - 1)
        return obs, reward, terminated, truncated, info


class AsimovGetupClipEnv(AsimovGetupRefEnv):
    """Imitate a HARVESTED get-up: the reference is a demonstration, not a draft.

    Everything above this class tracked an AUTHORED reference, and the route it
    describes has been measured impassable: the `pike` reaches 98 deg of trunk
    tilt against the 50 deg the trained stand recovers from, uprightness-alone
    MPPI bought 14 degrees of the 62 needed, and six independent optimisers plus
    ~50 M training steps stopped in the same place. Three ladder designs failed
    for one reason — their rungs are not holdable (interpolated ones are
    one-footed and collapse in 1 s; fall-derived ones are the states the robot is
    falling THROUGH).

    What changed is that a working demonstration now exists IN OUR PHYSICS.
    `openhorizon-labs/asimov1-getup-safefall` (Apache-2.0, pinned sha
    f657abede...) rises **20/20, median 1.22 s** on our own fetched
    `asimov_1.xml`, which retires every "this body cannot" conclusion in the
    ledger: those described our route and our method, not the robot.
    `scripts/harvest_getup_clips.py` records its joint trajectories, and this env
    tracks them.

    Why TIME-indexed and not phase-indexed. The parent advances a phase when a
    waypoint is reached, which is right for a draft whose tempo is a guess. A
    demonstration already HAS the tempo — their get-ups run 2.74-3.44 s with a
    median 1.00 s to standing — so the reference is `clip[t]` and the policy is
    asked to be where the demonstration was at that moment.

    Why joint angles and not torques. Their runner drives `d.qfrc_applied` from
    its own DC-motor model (`hi = min(effort_limit, peak_torque * (1 -
    qd/velocity_limit))`, effort 40, peak 120) and bypasses MuJoCo's actuators;
    ours are position servos at a flat +/-45, stronger at low speed and never
    weaker. The joint path transfers, the torques do not.

    The cut is the curriculum, not the reward. On the G1 kick the imitation
    reward alone stood through every strike and a laddered not-tracking
    termination took it 0.10 -> 0.62 m, so `track_tol` tightening across stages
    is what does the work here too.
    """

    #: Where `harvest_getup_clips.py` writes, and the knob that points elsewhere.
    CLIP_ENV = "MICRODUCK_ASIMOV_GETUP_CLIPS"

    #: The action range the TEACHER actually uses, measured over its own
    #: rollouts in this env: p50 0.41, p90 **5.37**, p99 **7.20**, max **11.51**.
    #: At the body's inherited 4.0 — the DUCK's number — 15.8 % of the teacher's
    #: own steps are clipped and **92-98 % of its DAgger CORRECTIONS are**, so
    #: the student is asked to copy a teacher whose answers it cannot execute:
    #: iteration 0 cloned the teacher's trajectory to 0.0026 rad^2 and 60/60
    #: standing, and iteration 1 on the student's own states went 0/60.
    #:
    #: 13.0 is not a guess from either direction. Their policy spec says
    #: `"clip": null` and Menlo's `agent.yaml` says `clip_actions: null`, and
    #: 12.57 is max|jnt_range - DEFAULT_POSE| / ACTION_SCALE — the clip at which
    #: the whole PHYSICALLY reachable set becomes commandable. The teacher's
    #: 11.51 maximum sits just under that, which is two independent derivations
    #: agreeing.
    #:
    #: It is set HERE and not on the body because widening it globally has a
    #: measured cost: at 13.0 the authored `fold` pose stops holding (0.242 m at
    #: u -0.43 against its recorded 0.517 / +0.83), and `fold` is the rung our
    #: best stand depends on. This task does not command authored poses.
    TEACHER_ACTION_CLIP = 13.0

    def __init__(self, *args, clip_file: str | None = None, **kw):
        kw.setdefault("start_phase", "prone")      # the parent needs a valid rung
        kw.setdefault("free", False)               # we track; free mode drops it
        kw.setdefault("action_clip", self.TEACHER_ACTION_CLIP)
        super().__init__(*args, **kw)
        from . import asimov as _asimov
        path = (clip_file or os.environ.get(self.CLIP_ENV)
                or str(_asimov.CACHE_DIR / "getup_clips.npz"))
        d = np.load(path, allow_pickle=True)
        lens = d["lengths"].astype(int)
        flat, cuts = d["qpos"], np.cumsum(lens)[:-1]
        self._clips = np.split(flat, cuts)
        self._clip_starts = d["starts"]
        self._clip_src = str(d["source"]) if "source" in d else path
        if len(self._clips) != len(self._clip_starts):
            raise RuntimeError(
                f"{path}: {len(self._clips)} clips but "
                f"{len(self._clip_starts)} start states")
        self._clip = self._clips[0]
        self._clip_i = 0

    # ----------------------------------------------------------------- reset

    def reset(self, *, seed=None, options=None):
        """Start where a SUCCESSFUL demonstration started, and track it."""
        obs, info = super().reset(seed=seed, options=options)
        self._clip_i = int(self._rng.integers(len(self._clips)))
        self._clip = self._clips[self._clip_i]
        self.data.qpos[:] = self._clip_starts[self._clip_i]
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._ep_steps = 0
        self._anchor_z = float(self.data.xpos[self.trunk_body_id][2])
        return self._get_obs(), info

    # ------------------------------------------------------------- the target

    #: How far the ROOT may be from the demonstration's before `track` stops
    #: paying: metres of height, and of the uprightness cosine.
    #:
    #: THE BUG THESE EXIST FOR. `track` used to price JOINTS ONLY, and joints
    #: cannot tell a standing robot from a fallen one. Commanded `DEFAULT_POSE`
    #: — which is action 0, i.e. what an untrained policy emits — a robot lying
    #: on the floor straightens its legs, matches the standing reference in
    #: joint space, and collected **+7.409 per step against the riser's
    #: +6.067**. It never stood, so the `_stood`-armed fall cut never fired and
    #: it ran the full episode: measured, a do-nothing policy earned **2775 a
    #: episode against the clone's 788**, and a 2 M-step warm-started fine-tune
    #: duly converged to `ep_rew` 2942 while its rise fell from 24/24 to 15/24.
    #: PPO did exactly what it was paid for.
    #:
    #: The demonstration stores all 30 of `qpos`, root included, and the root is
    #: the part joints cannot express — so it is now priced, as a PRODUCT with
    #: the joint match, for the reason this class already gives for `rise`: a
    #: product cannot be farmed by one axis. Same fault as the authored
    #: reference's (`AGENTS.md`, "same joints, different body"), recurring here
    #: because the harvested reward was written from the joint half alone.
    ROOT_Z_STD = 0.10
    ROOT_U_STD = 0.25

    def _clip_target(self):
        """The demonstration's joints at THIS moment, held at the last frame."""
        return self._clip[self._clip_t()][self._q_adr]

    def _clip_t(self) -> int:
        return min(int(self._ep_steps), len(self._clip) - 1)

    def _root_err(self) -> tuple[float, float]:
        """(height error in m, uprightness error) against the demonstration.

        Root-to-root on purpose: the reference's uprightness is read off the
        same free-joint quaternion as the robot's, so the waist angle — itself
        one of the 23 tracked joints — cannot make the two incomparable.
        """
        ref = self._clip[self._clip_t()]
        dz = abs(float(self.data.qpos[2]) - float(ref[2]))
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.ascontiguousarray(ref[3:7], dtype=float))
        ref_u = float(mat[8])
        mujoco.mju_quat2Mat(mat, np.ascontiguousarray(
            self.data.qpos[3:7], dtype=float))
        return dz, abs(float(mat[8]) - ref_u)

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        err = self._joint_err(self._clip_target())
        pelvis_z = float(self.data.xpos[self.trunk_body_id][2])
        upright = float(-self._projected_gravity()[2])
        height = min(max(pelvis_z / self._stand_z, 0.0), 1.0)
        feet = sum(bool(v) for v in self._foot_contacts().values()) / 2.0
        up = (height >= self.UP_HEIGHT_FRAC and upright >= self.UP_GRAVITY
              and feet >= 1.0)
        if up:
            self._stood = True
            self._held += 1
            self._held_best = max(self._held_best, self._held)
        else:
            self._held = 0
        rate = self._action_rate()
        self._track_err = err
        dz, du = self._root_err()
        terms = {
            # A PRODUCT of the joint match and the ROOT match — see
            # ROOT_Z_STD. Either factor alone is farmable by lying still.
            "track": self.W_TRACK * float(np.exp(-err / self.TRACK_STD))
                     * float(np.exp(-dz / self.ROOT_Z_STD
                                    - du / self.ROOT_U_STD)),
            "up": self.W_UP * float(up),
            "action_rate_penalty": -self.W_ACTION_RATE * rate
                                   * self._penalty_scale(),
        }
        return float(sum(terms.values())), terms
