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
from . import asimov_motor as _motor
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

    #: Asimov's vendors DO publish a speed-torque curve, so this body can
    #: have `actuator='dc'` — see `robots/asimov_motor`.
    MOTOR_MODEL_CLS = _motor.DcMotorClip

    #: Which actuator the env runs: "xml" (the MJCF position servos, clamped
    #: at the URDF's continuous effort with NO speed dependence) or "dc" (the
    #: same PD output clamped inside the published speed-torque envelope).
    #: Read HERE and not from `MICRODUCK_ACTUATOR`, because this body's
    #: `setdefault` below would swallow that one — and recorded onto the run
    #: by `AsimovBody.train_env_kwargs`, since a policy trained unlimited and
    #: one trained inside the envelope are not two seeds of one experiment.
    MOTOR_ENV = "MICRODUCK_ASIMOV_MOTOR"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot", W.spec())
        act = kwargs.get("actuator_force") or kwargs.get("actuator") or ""
        if not act:
            act = os.environ.get(self.MOTOR_ENV, "") or "xml"
        act = str(act).strip().lower()
        if act == "bam":
            raise ValueError(
                "actuator='bam' is the duck's XL330 servo model — Asimov runs "
                "the Synapticon servos injected by robots/asimov_walk.walk_spec "
                "('xml'), or those servos inside their published speed-torque "
                "envelope ('dc', robots/asimov_motor)")
        kwargs["actuator_force"] = act
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
    #: Env var for `max_span` below.
    SPAN_ENV = "MICRODUCK_ASIMOV_GETUP_MAX_SPAN"
    #: Env var for `launch_span` — the span allowed INSIDE the launch window
    #: only. The hybrid: keep the narrow stance everywhere the robot is
    #: slow and supported, and permit the sprawl only in the one moment that
    #: demonstrably needs it.
    LAUNCH_SPAN_ENV = "MICRODUCK_ASIMOV_GETUP_LAUNCH_SPAN"
    #: Env var for `fall_depth` — how far down the fall `start_rung="falling"`
    #: may reset. 0 = only states still at `fold`'s height, 1 = all the way to
    #: `pike`. THE curriculum axis for the one leg nothing can learn from a
    #: static start.
    FALL_DEPTH_ENV = "MICRODUCK_ASIMOV_GETUP_FALL_DEPTH"
    #: Env var for `span_cost` — a CONTINUOUS charge per metre of foot span
    #: above `span_target`, as opposed to `max_span`, which TERMINATES.
    #:
    #: The distinction is the whole lesson of 2026-10-08. A termination is a
    #: cliff: `teach-asimov_getup_rise-126f42-s2` cut at 0.85 m while the
    #: policy's own rise spans 0.99, so every attempt to stand ended the
    #: episode and standing became strictly worse than lying still. Measured,
    #: the `up` term went to 0.0 at 98 k steps and never returned, `ep_len`
    #: climbed 28 -> 400, and the return settled at **35.5 against a
    #: do-nothing floor of 39.8**. The policy had not failed to narrow; it had
    #: correctly learned not to get up. A cost has a gradient at 0.99, which
    #: is the one place a cut at 0.85 has none.
    SPAN_COST_ENV = "MICRODUCK_ASIMOV_GETUP_SPAN_COST"
    #: Env var for `span_target` — the width below which the charge is zero.
    SPAN_TARGET_ENV = "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET"

    #: WHICH AXIS the span charge prices: "both" (default, unchanged),
    #: "lateral", or "fore_aft".
    #:
    #: `_foot_span` is `hypot(dx, dy)` — lateral and fore-aft collapsed into
    #: one number — and every span charge on this body has been priced against
    #: that. Decomposed in the body frame across the crossing, the working
    #: get-up is **0.353 m lateral + 0.702 m fore-aft**: only 42 % of it is
    #: abduction, and the rest is a STRIDE. A human stride is 0.7-0.8 m and
    #: nobody calls it the splits, so a charge on the combined number is
    #: partly charging the robot for stepping — which is what the 0/12 arm
    #: (span 30 + limit 50, against 10/12 without) was really paying for.
    #:
    #: Split so a charge can target the axis its claim is about. Lateral is
    #: already 0.32-0.37 on EVERY route measured (prone, supine, both sides,
    #: sitting, kneeling) and under the 0.40 gate, so "narrow" has little left
    #: to buy there; the posture that could still change is the fore-aft one.
    SPAN_AXIS_ENV = "MICRODUCK_ASIMOV_GETUP_SPAN_AXIS"

    #: Charge the span ONLY while the pelvis is inside this band, metres.
    #: "" (default) charges every step, unchanged.
    #:
    #: The charge is about the CROSSING, and applying it per step prices two
    #: regions where the claim does not apply: lying down, where foot
    #: positions mean nothing and the robot is paid to keep its legs together
    #: while it has no business moving them, and the long standing tail, where
    #: the stance is already 0.23 m and the charge is pure noise on a solved
    #: problem. An 8 s episode is ~1 s of crossing and ~7 s of standing, so
    #: the ungated charge spends 7/8 of its weight on the wrong frames.
    SPAN_BAND_ENV = "MICRODUCK_ASIMOV_GETUP_SPAN_BAND"
    #: Env var for `ground_cost` — a charge per step with NO part of the
    #: robot touching the floor.
    #:
    #: Measured on `teach-asimov_getup_turn-6a1490-s7`, the first policy here
    #: that stands from the floor: of its 3.4 s rise, **1.04 s is spent with
    #: zero floor contacts**, in 8 separate flights, the lowest geom of a
    #: 32 kg robot reaching **105 mm** of clearance. It is not pushing up, it
    #: is HOPPING up, and then it stands — 0.00 s airborne after it arrives.
    #:
    #: Nothing in the economy forbade it. `stand`, `rise` and `progress` pay
    #: trunk height, `rotate` pays uprightness and `tuck` pays the feet coming
    #: under the mass, and a bounce buys all five at once for one impulse,
    #: where a push-up buys them against gravity for a second. The cheapest
    #: route through that reward is ballistic, so the policy found it. This is
    #: the charge that makes the supported route the cheap one, and it is the
    #: difference between a stand and a SAFE stand: a 32 kg body that lands
    #: from 105 mm lands on hardware too.
    #:
    #: A charge, not a termination, for the reason SPAN_COST_ENV records: a
    #: cut at the moment of flight would make every attempt to rise end the
    #: episode, and lying still would win.
    GROUND_ENV = "MICRODUCK_ASIMOV_GETUP_GROUND"
    #: Env var for `balance_cost` — a charge per step for the centre of mass
    #: being OUTSIDE the support polygon while the robot is on its feet.
    #:
    #: The clause a stand was missing. Measured on
    #: `teach-asimov_getup_turn-380ec2`: while "standing" by
    #: `standing_report`, the CoM is outside the polygon **74 % of the
    #: time** (median -22 mm, p10 -131 mm), and every lost stand has one
    #: trace — uprightness holds at +0.98 while the gap from the feet grows
    #: 200 -> 320 mm and the trunk slides down. It is not toppling over, it
    #: is translating off its feet while vertical.
    #:
    #: Nothing paid for staying over them. `tuck` pays the CoM-to-feet gap
    #: closing, but it is MONOTONE BEST-SO-FAR — it pays for touching a
    #: value once, never for holding it — and `standing` pays the same per
    #: step for a balanced robot and a toppling one. So the policy banked
    #: the rise and let go.
    #:
    #: Gated on being UP (upright past `STAND_UPRIGHT`, nothing but feet
    #: down), so it is exactly zero for the whole rise and cannot charge the
    #: get-up it exists to finish — the failure recorded in "A ramped
    #: penalty can eat its task".
    BALANCE_ENV = "MICRODUCK_ASIMOV_GETUP_BALANCE"
    #: Env var for `home_pay` — a per-step PAY for sitting in the robot's own
    #: default pose while upright. The get-up has to END somewhere usable.
    #:
    #: Measured 2026-10-09 on `teach-asimov_getup_turn-dd413e`. The pose it
    #: reaches is **19 deg from the home stand on average and 57 deg at the
    #: worst joint**: `waist_yaw` +56.8 (home 0, and it is the ONLY joint
    #: between pelvis and torso), the hips scissored -53.6/+51.4 against
    #: home's -8.6/+8.6, `left_ankle_pitch` +20.9 against home's -17.2 — the
    #: full width of a 20 deg range, on the wrong side.
    #:
    #: It is not balancing, it is JAMMING: that pose puts **eight joints
    #: outside the model's limits at once** (both ankle pitches, both ankle
    #: rolls, a knee, both shoulder rolls, an elbow) and rests on the
    #: constraint forces. `_refuse_illegal_joints` will not even load it. So
    #: it freezes — no range left in any direction — then slides off anyway,
    #: and nothing can follow it. On hardware it would not exist.
    #:
    #: The target is not invented: the home stand IS `robot.default_pose`
    #: (0.0 deg from the authored `stand` waypoint), and under ZERO action
    #: the servos hold it **8.00 s of 8**, trunk 0.608, uprightness +1.00,
    #: CoM **+47 mm inside** the support polygon. The place to hand a walk
    #: policy the robot already holds itself up.
    HOME_ENV = "MICRODUCK_ASIMOV_GETUP_HOME"
    #: Env var for `joint_vel_cost` — the charge on LIMB SPEED while up,
    #: overriding `W_JOINT_VEL`.
    #:
    #: `W_JOINT_VEL` is **0.002**, and on `teach-asimov_getup_whole-c4bdba`
    #: that bought `joint_vel_penalty` **-0.22 per episode** against 14,690
    #: of positives — 0.0015 %, which `scripts/term_audit.py` prints as DEAD.
    #: The term was not missing and not mis-gated; it was present, correct and
    #: worth nothing, which is the hardest of the three to notice because
    #: every other check passes. The robot duly jiggled at a median **326
    #: deg/s** while standing, against the shipped walker's **3 deg/s** in the
    #: same posture, and no weight on `home` or `calm` reaches that: `home`
    #: pays joint POSITION, `_calm_charge` prices TRUNK speed, so a limb
    #: twitching at punch speed under a still torso is invisible to both.
    JOINT_VEL_ENV = "MICRODUCK_ASIMOV_GETUP_JOINT_VEL"
    #: Env var for `over_clip_cost`, overriding `W_OVER_CLIP` (0.2). Same
    #: story, same run: raw |a| left the 4.0 clip on **37 % of steps**,
    #: reaching **16.0**, while the charge that exists to make exactly that
    #: cost something was worth **-42 per episode**. It is EXACTLY ZERO
    #: inside the clip, so raising it cannot eat the task — every unit it
    #: charges is a command the robot could never have executed.
    OVER_CLIP_ENV = "MICRODUCK_ASIMOV_GETUP_OVER_CLIP"
    #: Joint error, radians, at which the home pay has decayed to 1/e. Sized
    #: against the measured gap: the policy sits at 0.33 rad mean error, so
    #: 0.20 leaves real gradient where it actually is rather than a flat
    #: floor it can never climb off.
    HOME_TOL = 0.20
    #: Env var for `limit_cost` — a charge per radian a joint sits OUTSIDE
    #: its own range. MuJoCo's limits are soft, so a policy can push through
    #: a stop and stand on the constraint force; nothing here charged for it
    #: and the first policy to stand did exactly that, on eight joints.
    LIMIT_ENV = "MICRODUCK_ASIMOV_GETUP_LIMIT"
    #: Env var for `push_bw` — random SHOVES, in bodyweights, applied to the
    #: trunk once the robot is standing.
    #:
    #: THE STATE THAT WAS NEVER SAMPLED. Across 38 M steps of this curriculum
    #: the robot was never disturbed: `xfrc_applied` carried the vertical
    #: ASSIST and nothing else. So "it stands but does not correct" was never
    #: a reward problem — a recovery is a state no rollout ever contained,
    #: and an unsampled state's value is never learned. Jonathan watched it
    #: and said *it doesn't look like it's doing any correction*; he was
    #: right, and nothing in the economy could have fixed it.
    #:
    #: A push lands only once `is_standing` holds, because a shove during the
    #: rise is just noise on a motion that is already hard. Direction is
    #: uniform in yaw and magnitude uniform in [0.5, 1.5] x `push_bw`, so the
    #: policy cannot learn one canned answer; they repeat through the episode
    #: so a single long stand is many recovery samples rather than one.
    PUSH_ENV = "MICRODUCK_ASIMOV_GETUP_PUSH"
    #: Env var for `calm_cost` — a charge per m/s of HORIZONTAL trunk speed
    #: above what a stance can absorb, while the robot is on its feet.
    #:
    #: THE RISE IS A LUNGE, and this is the measurement that says so. At the
    #: instant it first satisfies `is_standing`, the trunk is moving
    #: **0.89 m/s** sideways (p90 1.06) with **5.15 rad/s** of angular speed
    #: — against a servo-held home stand that absorbs **0.22 m/s** and falls
    #: at 0.29. It arrives four times too fast. Inside the same 24 episodes
    #: the split is clean: the ones that stayed up arrived at 0.60-0.87 m/s,
    #: the ones that fell at 0.88-0.98.
    #:
    #: This is why three stages of pose work bought so little. `balance`
    #: pays the CoM being over the feet, `home` pays the joints being in the
    #: default pose — and neither can be held by a body that arrives with a
    #: metre per second of momentum and a 20 deg ankle to stop it with. It
    #: also explains the jam (bracing), the saturated ankles (absorbing),
    #: and why handing off to the servos at the standing instant makes
    #: things WORSE, 13/24 -> 3/24: no passive stance absorbs 0.89 m/s.
    #:
    #: One-sided above `CALM_FREE`, like `_span_charge`: a quiet arrival
    #: pays nothing, so this is not a target to be optimised toward and
    #: cannot pay for refusing to move during the rise. Gated on being on
    #: its feet, so the rise itself is never charged.
    #: Env var for `swing_pay` — a pay for SWINGING THE ARMS UP during the
    #: rise, both the same way, which is momentum the legs cannot make.
    #:
    #: Jonathan's suggestion, 2026-10-10: *I wonder if it would also help if
    #: it swung its arms up as it's trying to stand up.* Measured on
    #: `teach-asimov_getup_turn-540bc4` over 2120 samples of the rise, it
    #: ALREADY does — and nothing pays for it:
    #:
    #:   symmetric swing (both arms one way)  2.75 rad/s   corr with trunk
    #:                                                     rising **+0.287**
    #:   anti-symmetric flail (one up one down) 2.57 rad/s  corr **+0.162**
    #:
    #: The symmetric component is both the larger motion AND nearly twice
    #: the correlate of the trunk going up. `arm_push` pays load x upward
    #: speed — PUSHING OFF THE FLOOR — and says nothing about swinging; so
    #: the one arm motion most associated with rising has been unpriced for
    #: this whole curriculum.
    #:
    #: Why it should matter here specifically: this robot arrives at
    #: standing with 0.89 m/s it cannot absorb through a 20 deg ankle. Arm
    #: swing is angular momentum generated ABOVE the ankle, which is the one
    #: source of rising impulse that does not have to be paid for later by
    #: the feet.
    #:
    #: Gated to the RISE (below the stand bar, not yet standing) and
    #: one-sided, so holding still costs nothing and a stand is never paid
    #: for waving.
    #: Env var for `handoff_pay` — train the get-up to DELIVER the robot to
    #: Menlo's walker, instead of to stand alone for eight seconds.
    #:
    #: THE JOB CHANGED. Everything before this trained the policy to be its
    #: own balance controller for a whole episode, and that is why it jams:
    #: measured, its standing pose puts EIGHT joints outside their limits and
    #: sits 49 deg from the default pose, because holding itself up is the
    #: only tool it has. It does not need that tool. Menlo ship a walker that
    #: holds the idle stance, and handing to it beats our policy on every
    #: axis (20/24 vs 13/24 standing, 3.7 deg from idle vs 49.1, 100 % shove
    #: recovery vs 75 %).
    #:
    #: So the real deliverable is a CATCHABLE HANDOFF STATE, and the catch
    #: basin says which one. Over 128 handoffs, 97 caught (76 %):
    #:
    #:   at handoff      caught   dropped   separation
    #:   trunk speed      0.409     0.793     1.19 sd   <- decides it
    #:   foot span        0.422     0.345     0.87 sd   (WIDER is caught more)
    #:   trunk z          0.609     0.545     0.63 sd
    #:   uprightness      0.988     0.889     0.48 sd
    #:
    #: Speed is the only strong separator, which is also why `calm` failed
    #: as a retrofit: it charged the HOLD, and the hold was never the task.
    #:
    #: The episode ENDS at the handoff and pays
    #: `handoff_pay x steps_remaining x score`, so arriving catchable is
    #: worth exactly what standing perfectly for the rest of the episode
    #: would have been — and arriving fast is worth a fraction of it. That
    #: self-balances: a terminal bonus that does not replace the forgone
    #: income just teaches the policy to avoid the terminal.
    HANDOFF_ENV = "MICRODUCK_ASIMOV_GETUP_HANDOFF"
    #: Trunk speed, m/s, at which the walker stops catching — the median of
    #: the handoffs it DROPPED. Score is 0 here and 1 at `HANDOFF_GOOD`.
    HANDOFF_DROP = 0.79
    HANDOFF_GOOD = 0.20
    #: Standing, slower than this, for this many steps — the same condition
    #: `behaviors.asimov_tasks` hands the lab, defined ONCE here so the
    #: reward and the deployment cannot drift apart.
    HANDOFF_SPEED = 0.60
    HANDOFF_DWELL = 4
    SWING_ENV = "MICRODUCK_ASIMOV_GETUP_SWING"
    #: A kip happens at the START of a get-up or not at all, so the swing is
    #: only paid inside this window, in seconds.
    SWING_WINDOW_S = 2.0
    CALM_ENV = "MICRODUCK_ASIMOV_GETUP_CALM"
    #: Horizontal trunk speed, m/s, that costs nothing — measured, not
    #: chosen: it is what a servo-held home stand demonstrably absorbs
    #: (survives 0.15 BW for 0.15 s = 0.22 m/s, falls at 0.20 BW = 0.29).
    CALM_FREE = 0.22
    #: How long after it first stands the arrival charge applies, seconds.
    #: Beyond this the robot is holding, not arriving, and a charge there
    #: prices the corrections — measured, that cost 13/24 -> 6/24.
    CALM_WINDOW_S = 1.0
    #: Seconds a shove lasts, and the gap between them.
    PUSH_HOLD_S = 0.15
    PUSH_EVERY_S = (0.8, 1.8)
    #: Seconds of STANDING between shoves, overriding `PUSH_EVERY_S`. 0 keeps
    #: the training cadence, so every earlier run reproduces exactly.
    #:
    #: Two different things, and conflating them is what made the lab demo
    #: unreadable. The training cadence counts WALL TIME, so with a rise
    #: taking ~3.4 s against a 0.8-1.8 s gap the next shove was always overdue
    #: by the moment the robot stood — it got hit again the instant it was up,
    #: every time, which reads as flailing rather than as recovery. With this
    #: set, the clock is held while the robot is DOWN and the gap is measured
    #: from the moment it is standing again, so there is a real settle window
    #: after every recovery.
    #:
    #: The DIRECTION stays random either way: that is what makes it a test
    #: rather than a rehearsed stunt. Only the timing is made legible.
    PUSH_EVERY_ENV = "MICRODUCK_ASIMOV_GETUP_PUSH_EVERY"
    #: +-30 %, so a demo is not metronomic and cannot be anticipated.
    PUSH_EVERY_JITTER = 0.3
    #: How far outside the polygon still counts, metres. Past this the robot
    #: is gone and a bigger number is not more information — it would just
    #: make one bad step dominate an episode.
    BALANCE_CAP = 0.15
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
    #: Env var for `prologue`. "pushup" makes the first ~1.3 s of EVERY
    #: episode play the authored push-up — prone, plant_forearms,
    #: draw_knees_in, quadruped, pike — through the position servos, ignoring
    #: the policy, and hand over at `pike`.
    #:
    #: Two reasons, and the second is why it is in the env rather than a
    #: script. (1) That half of the get-up needs no policy: it plays open loop
    #: at a 0.215 m foot span, so training it is wasted compute and letting a
    #: policy near it is wasted risk. (2) It happens in STEPPED time, so the
    #: lab shows it — a `reset` that placed the robot at `pike` was invisible,
    #: and every run for a week reset above the push-up, which is why nobody
    #: watching the stage ever saw one.
    #:
    #: It also changes the handover: the policy inherits the pike the robot
    #: ARRIVED at, carrying its real velocity, instead of one placed at rest.
    #: Env var for `rotate_pay` — what a NEW BEST uprightness is worth.
    #:
    #: Measured 2026-10-09: the get-up's blocker is ROTATION, not height. At
    #: `pike` the trunk is 112 deg from vertical and the hips+ankles supply
    #: ~59; there is no back pitch joint, so lifting the pelvis only dangles
    #: the trunk (assist at 0.85 reaches fold's HEIGHT at uprightness -0.86).
    #: And yet `progress` pays monotone trunk HEIGHT and nothing pays
    #: uprightness at all — the reward was blind to the one axis the task
    #: turns on. Monotone like `_progress`, for the same reason: a per-step
    #: rate is farmable by rocking, a new best is not.
    #: Env var for `tuck_pay` — what bringing the FEET UNDER THE MASS is
    #: worth, as monotone reduction in the horizontal CoM-to-feet distance.
    #:
    #: Measured 2026-10-09 on the policy that learned to sit up. It ends
    #: upright (+0.95) on its feet at 0.11 m and then stops, and the reason
    #: is in the joints: the knee sits at **-1 deg, its 0 deg stop**, legs
    #: straight out, with the CoM **300 mm from the feet**. It is a long sit.
    #: 86 deg of knee travel is available and completely unused.
    #:
    #: Nothing paid for using it. Tucking the feet in LOWERS the trunk and
    #: costs uprightness for a moment, so `rise`, `stand`, `progress` and
    #: `rotate` all charge for the one move that makes standing possible.
    #: This pays it: monotone, like `_progress` and `_upright_gain`, because
    #: a per-step rate would be farmable by shuffling the feet back and forth.
    TUCK_ENV = "MICRODUCK_ASIMOV_GETUP_TUCK"
    ROTATE_ENV = "MICRODUCK_ASIMOV_GETUP_ROTATE"
    #: ROLL ONTO THE BACK — pay the one transition the whole prone problem
    #: reduces to. Measured 2026-10-12: from a genuinely SUPINE pose the
    #: shipped get-up stands **40/40**; from prone it manages 8/40. A crude
    #: 180 deg root flip lands only 4/40 genuinely supine and already lifts it
    #: to 19/40. So "get onto the back" is worth 40/40 with nothing retrained,
    #: and hand-authoring it failed — a parameterised twist-and-push scanned
    #: over direction, waist, push, scissor, knee, TEMPO and a wind-up reached
    #: 1/6 at best (`scripts/search_prone_roll.py`).
    #:
    #: The signal is Menlo's own classifier's axis, not uprightness:
    #: `f_z = xmat[2,0]` of the torso, the world-z component of its FORWARD
    #: axis, which runs -1 (face down) to +1 (face up) and is exactly what
    #: `asimov_getup.label` takes the argmax of. Uprightness is useless here —
    #: it is ~0 for prone, supine AND side alike, and reading it as a roll
    #: indicator is how an earlier probe reported "rolled onto its side" for
    #: poses the classifier calls prone.
    #: CHARGE LIMB SPEED DURING THE RISE. `joint_vel_penalty` is multiplied
    #: by `float(up)`, and `up` requires the robot to be standing already —
    #: so the entire get-up, which is the FASTEST thing this robot does, has
    #: never been charged for how hard it flings its limbs.
    #:
    #: Measured 2026-10-13, peak joint rate: walking (the shipped walker in
    #: control) median **3 deg/s**, p99 566; RISING (this policy in control)
    #: median **532 deg/s**, p99 1171, max **1753**; falling 1580. A human
    #: elbow in a punch is roughly 500-900 deg/s, so the get-up swings limbs
    #: at or above punch speed as a matter of course, every time it stands —
    #: next to anyone kneeling beside a fallen robot, that is the hazard, far
    #: more than the fall itself (which topples AWAY from the pusher 19/19).
    #:
    #: A separate term rather than widening the gate, so it prices and reads
    #: independently in the budget. Off by default: every earlier run
    #: reproduces, and this one trades directly against the rise, whose top
    #: correlate is a fast symmetric arm swing (+0.287).
    #: HONOUR THE ACTUATOR'S RATED SPEED. Menlo's URDF publishes a
    #: `velocity` limit per joint (228-720 deg/s across the 23); their MJCF
    #: carries `forcerange` but NO velocity cap, so nothing in training has
    #: ever stopped a policy exploiting speeds the hardware cannot produce.
    #:
    #: Measured 2026-10-13 on `teach-asimov_getup_turn-967072`: during the
    #: RISE, **82 % of frames have at least one joint over its rating**,
    #: median 1.67x, p99 3.84x, max **5.29x** — worst is
    #: `left_shoulder_pitch` at 5.3x its rated 228 deg/s. Menlo's own walker,
    #: by contrast, exceeds the rating on 0 % of frames. So this is not a
    #: safety nicety: the get-up's defining arm swing is being performed at
    #: speeds that do not exist on the robot, and the policy may not run
    #: there at all.
    #:
    #: Modelled the way a motor actually behaves — available torque falls to
    #: zero at the rated speed — rather than by clamping `qvel`, which is not
    #: physics and would hide the cost instead of charging it. Applied
    #: through `qfrc_applied` so no shared `mjModel` is mutated.
    #:
    #: OFF by default: every run before this one trained against a robot with
    #: no speed limit, and turning it on silently would make those runs
    #: irreproducible.
    RISE_VEL_ENV = "MICRODUCK_ASIMOV_GETUP_RISE_VEL"
    ROLL_ENV = "MICRODUCK_ASIMOV_GETUP_ROLL"
    #: END THE EPISODE WHEN A ROBOT THAT STOOD GOES BACK DOWN. On by
    #: default, because it is right for TRAINING: a failed episode stops
    #: costing samples the moment it is lost.
    #:
    #: It has to be switchable for the recovery LOOP, in two places that each
    #: made the loop unmeasurable or dishonest. `probe_recovery_loop.py`
    #: first read 0 re-ups on BOTH arms because episodes ended at the shove —
    #: 12 episodes fired 13 shoves instead of 35. And in the lab,
    #: `Duck.step` calls `reset()` on `terminated`, so a knocked-down robot
    #: RESPAWNS: a viewer would see a fresh episode that looks exactly like a
    #: recovery, which is the one thing a demo must never do.
    #:
    #: A real robot that falls does not vanish. Off is the DEPLOYMENT's
    #: behaviour; on is the trainer's.
    FELL_CUT_ENV = "MICRODUCK_ASIMOV_GETUP_FELL_CUT"
    #: `f_z` at or above this counts as on the back. The classifier needs
    #: `f_z` to be the argmax of (f_z, -f_z, -l_z, l_z); 0.70 implies it.
    ROLL_DONE = 0.70
    #: Paid PER STEP while on the back, as `standing`'s sibling — the
    #: monotone gain below pays for TOUCHING supine and something must pay
    #: for being there, or the roll overshoots back onto its face for free.
    #:
    #: SMALL ON PURPOSE, and the arithmetic is the reason. Measured income on
    #: this parent: a prone SUCCESS earns ~9,170 an episode (`stand` 1,664 +
    #: `rise` 476 + ...), a prone FAILURE ~16. At `roll_pay` 100 this gives
    #: 2.0 a step, so ~400 over a long supine stretch — decisive against
    #: failing, and nowhere near the ~2,100 standing pays, so it cannot teach
    #: the robot to lie on its back instead of getting up. At the 0.5 this
    #: started as it would have been 50 a step and done exactly that.
    ROLL_HOLD_W = 0.02
    PROLOGUE_ENV = "MICRODUCK_ASIMOV_GETUP_PROLOGUE"
    #: The authored push-up, in order. `pike` is the handover.
    PROLOGUE_ROUTE = ("prone", "plant_forearms", "draw_knees_in",
                      "quadruped", "pike")
    #: Seconds per leg of the ramp, and the dwell after each. MEASURED: the
    #: route plays at 0.25 s a leg and falls at 0.50 (see "a reference has a
    #: tempo"), so this is not a free parameter.
    PROLOGUE_LEG_S = 0.25
    PROLOGUE_DWELL_S = 0.12
    ASSIST_ENV = "MICRODUCK_ASIMOV_GETUP_ASSIST"

    #: Env var for `getup_attitude` — an assist that holds the trunk's
    #: ATTITUDE instead of lifting it, OFF at 0.0 so no existing run moves.
    #:
    #: The lift assist cannot scaffold this get-up and the reason is
    #: mechanical, not a matter of tuning. Priced against ZERO ACTION from the
    #: `pushup` prologue, raising it buys pelvis height and LOSES attitude —
    #: uprightness at the peak goes -0.53, -0.36, -0.42, -0.48, -0.78, -1.00
    #: as the assist goes 0.00 -> 0.85 — because a vertical force at the
    #: pelvis tips the trunk over the planted forearms. At 0.85 the recipe's
    #: "crossing" is a HANDSTAND the assist produces with no policy at all:
    #: pelvis 0.643 m at uprightness -1.00, and on the trained policy 97 % of
    #: frames have nothing on the floor.
    #:
    #: A constant pitch TORQUE is the right kind of help and still not enough:
    #: -100 N.m reaches `fold`'s own attitude (+0.84 against +0.83) at 0 %
    #: airborne, but only at pelvis 0.111 m — it sits up without rising — and
    #: lift plus constant torque LAUNCHES the robot (58-69 % airborne above
    #: 80 N.m). A constant torque has no opinion about where it stops.
    #:
    #: So this is a PD on uprightness rather than a force: it vanishes as the
    #: trunk comes up, damps the rate so it cannot wind up, and is clamped, so
    #: the lift can raise the robot while this keeps it from going over the
    #: top. It is a scaffold to be faded exactly like the lift.
    #: Env var for `fold_pay` — per-step pay for being in a state the
    #: narrow stand policy can actually CATCH. OFF at 0.0.
    #:
    #: Every arrival term this repo has written, including `handoff_pay`,
    #: scores the TRUNK: height, uprightness, speed. Measured, that is the
    #: wrong target. A scaffolded state matching `fold`'s trunk to 30 mm at
    #: uprightness +0.73, FEET-ONLY and narrow at 0.215 m, is caught **0 of
    #: 78** times by `9702ba` — which stands 8/8 from `fold` placed through
    #: the identical replay path. The difference is JOINT SPACE: that state
    #: sits 0.290 rad mean and 1.214 rad max from `fold` (hips 0.186 against
    #: fold's 1.400), where the cache poses it handles 43/48 sit about
    #: 0.1 rad away. `fold` is a deep CROUCH and the state was straight-
    #: legged. So the catcher's basin is a neighbourhood of fold's
    #: CONFIGURATION, not of its trunk pose.
    #:
    #: `handoff_pay` cannot express this: `handoff_ready` requires
    #: `is_standing` before it fires, which is the thing that cannot be
    #: reached. So this is the missing term, and the three clauses are the
    #: measured spec rather than a guess:
    #:
    #: * joints within ~0.1 rad of the `fold` waypoint (the scale the
    #:   catcher's own start distribution sits at),
    #: * pelvis lift at or under `FOLD_LIFT` m/s. The basin figures this
    #:   clause was written against (21/24 at 0.0, 18/24 at 0.2, 7/24 at 0.6)
    #:   are VOID — `scripts/probe_catch_basin.py` scored `getup_ref` runs in
    #:   the plain `getup` env, where the same name `fold` is a SPRAWLED pose
    #:   (0.461 m span against the ref env's 0.215). Re-measured 2026-10-19 on
    #:   `c2bae8-s1`: 24/24 at 0.00 m/s, **6/24 at 0.20**, 0/24 at 0.40. So
    #:   `FOLD_LIFT = 0.20` admits arrivals that are caught a quarter of the
    #:   time; it is kept only so published numbers stay comparable, and a
    #:   gate written from it is three times too loose,
    #: * feet only on the floor, span at or under `FOLD_SPAN`.
    #:
    #: Paid PER STEP and not best-so-far, because the deliverable is dwelling
    #: in a catchable state long enough to be caught — a monotone term would
    #: pay for touching it once and let the robot leave for free, which is
    #: the failure `AGENTS.md` records for `progress`, `rotate` and `tuck`.
    #: Env var for `arm_unload` — a GATED charge on the weight carried
    #: through the arms. OFF at 0.0.
    #:
    #: `arm_effort` already charges `_arm_load`, but ONLY in the base getup
    #: env's terms dict: `AsimovGetupRefEnv` builds its own, and
    #: `arm_effort_penalty` is not in it. So on the `getup_ref` and
    #: `getup_gap` tasks that knob is DEAD — `ca350e` recorded
    #: `arm_effort: 0.02` and it was never charged once in 8.4 M steps.
    #:
    #: Why the charge is needed: `fold_pay` earned **0.00 per episode** across
    #: three stages of an 8.4 M-step chain because **no FEET-ONLY frame ever
    #: occurs** — the robot is supported on its forearms from the prologue to
    #: the end, so the arrival gate never opens. The blocker is not the
    #: arrival target, it is that the hands never come off.
    #:
    #: Why it is GATED and not flat: this robot has NO HANDS — its wrists
    #: carry no collision geometry — so the forearms are its only way up, and
    #: a flat charge removes the push-up along with the lingering, which is
    #: `AGENTS.md`'s "a ramped penalty can eat its task".
    #:
    #: Gated on NOT RISING rather than on height, and the first version had it
    #: wrong. A height gate at 0.35 m was measured UNREACHABLE on the policy it
    #: would warm-start from: over 1014 frames the trunk never exceeds **0.317
    #: m** (median 0.217, p90 0.282), so that charge would have fired on 0.0 %
    #: of frames and been a second `fold_pay`. The same rollout says the arms
    #: carry load on **98.2 %** of frames at a median **0.694** of body weight.
    #:
    #: So the reachable, correctly-targeted quantity is load carried while the
    #: trunk is NOT going up — which is exactly the complement of `arm_push`
    #: (load x upward trunk speed, the term that already describes what the
    #: arms are FOR). A genuine push pays nothing here; resting on the
    #: forearms does. `_arm_load` is a FRACTION of body weight from
    #: `mj_contactForce`, so a forearm brushing the floor weightless is
    #: already free.
    #: Env var for `arm_work` — SIGNED work done through the arms, paid per
    #: step. OFF at 0.0.
    #:
    #: Why `arm_push` is not enough. `arm_push` is
    #: `load x max(_trunk_up_speed(), 0)` and `_trunk_up_speed()` returns only
    #: the gain ABOVE the episode's best trunk height. That makes it
    #: unfarmable by oscillation — after the first bounce nothing exceeds the
    #: best again — and it also makes it **structurally incapable of starting
    #: a rise**. Measured 2026-10-19 on `140db6` from `pike`: `arm_push`
    #: earned **0.00 per episode** at w=150, because the robot only ever
    #: sinks and therefore never exceeds its start height. The one term that
    #: pays for pushing up on the arms pays nothing until the motion already
    #: works. The whole budget from `pike` was `track` 85.3 + `contact` 73.0
    #: with every other term exactly zero.
    #:
    #: Signed work fixes that without reintroducing the farm. `load x dz` is
    #: dense from the first step — negative while sinking on the forearms,
    #: positive while pushing up — and oscillation nets ZERO by construction,
    #: because the integral of signed `dz` round a closed cycle is zero. That
    #: is a different defence from monotone height and it does not have the
    #: can't-initiate defect. The rejected `up_speed x load` version was
    #: farmable precisely because it took `max(.., 0)` and so collected the
    #: up-halves of a bounce while the down-halves were free.
    #: Env var for `tuck_work` — SIGNED rate of closing the centre-of-mass to
    #: feet gap, paid per step. OFF at 0.0.
    #:
    #: The same defect as `arm_push`, in the term that now blocks the rise.
    #: `tuck` is `w * _tuck_gain()` and `_tuck_gain()` is monotone best-so-far
    #: — it pays only for a NEW smallest gap. Measured 2026-10-19 across
    #: `c130f7` (tuck INACTIVE), `54acc8` (w=450) and `405037` (w=450), the
    #: gap closed **0.055 m, 0.061 m, 0.060 m** — turning the term on and
    #: weighting it at 450 moved the quantity it pays for by 6 mm. It pays
    #: for an improvement that does not occur, so it cannot cause it.
    #:
    #: `arm_work` fixed exactly this shape for the push, and the effect was
    #: large: `arm_push` went 0.00 -> 227.73 per episode because signed work
    #: got the robot rising, which then satisfied the new-best gate the
    #: monotone term needed and could not reach itself. This is that
    #: treatment for the tuck — signed, so oscillation nets zero round a
    #: closed cycle, and dense from step one so it can START the motion.
    #: Env var for `tuck2_work` — SIGNED rate of closing the MEAN PER-FOOT
    #: horizontal distance to the centre of mass, paid per step. OFF at 0.0.
    #:
    #: `tuck_work` and `_tuck_gain` both measure the CoM against the feet
    #: MIDPOINT, and that quantity is not merely gameable, it is INVERTED.
    #: Measured 2026-10-19 -- midpoint / per-foot / span:
    #: `pike` 0.566 / 0.577 / 0.215; `fold` (the TARGET) 0.085 / **0.137** /
    #: 0.215; the splits a policy actually learns **0.033** / 0.446 / 0.890.
    #: The midpoint rates the SPLITS better than `fold`, so a reward built on
    #: it pays more for the degenerate state than for the goal -- which is
    #: what `7e0545` did, taking the midpoint 0.566 -> 0.021 by spreading the
    #: feet to 0.917 m into a state nothing recovers from (zero action, the
    #: fold policy and itself all 0/12). Spreading the feet AROUND the mass
    #: drives the midpoint to zero while moving both feet further away; the
    #: mean per-foot distance rises instead, and ranks `fold` best and `pike`
    #: worst. Signed and per-step for the reason in `ARM_WORK_ENV`.
    #: Env var for `rotate_work` — SIGNED rate of change of uprightness, paid
    #: per step. OFF at 0.0.
    #:
    #: THE THIRD TIME this env has needed the same fix. `arm_push`, `tuck` and
    #: `rotate` are all monotone best-so-far, and a monotone term cannot pay
    #: for a quantity the policy drives BACKWARDS -- it is identically zero
    #: there. Measured 2026-10-20 on the deep route bands, where the policy
    #: reverts to a face-down rise (uprightness at peak trunk height
    #: **-0.33** at 0.26-0.30 and **-0.46** at 0.20-0.26, climbing to 0.61 m
    #: INVERTED): `rotate` earns **19.2 of 3733 positives (0.5 %) at w=200**,
    #: because uprightness never sets a new best. `arm_push` earned 0.00 at
    #: w=150 for the same reason, and `tuck` moved its own quantity 6 mm at
    #: w=450.
    #:
    #: Signed work fixed both of those (`arm_work`, `tuck2`), and it is
    #: farm-resistant by cancellation rather than by monotonicity: the
    #: integral of a signed rate round a closed cycle is zero, so rocking
    #: upright and back nets nothing while a real rotation nets positive.
    #: Env var for `rise_work` — SIGNED rate of change of the env's own
    #: COUPLED objective `rise = height * max(upright, 0)`, paid per step.
    #: OFF at 0.0.
    #:
    #: THE SYNTHESIS OF TWO LESSONS, each learned four times.
    #:
    #: 1. A MONOTONE best-so-far term cannot START a motion: it is identically
    #:    zero wherever the policy drives its quantity backwards. `arm_push`
    #:    earned 0.00/episode at w=150, `tuck` moved its own quantity 6 mm at
    #:    w=450, `rotate` earned 0.5 % of positives at w=200.
    #: 2. An UNCOUPLED signed term gets spent on the axis it does not price.
    #:    `tuck_work` (CoM-to-feet midpoint) bought the 0.917 m splits;
    #:    `rotate_work` (uprightness alone) bought a roll onto the BACK —
    #:    uprightness +0.90 at trunk 0.078, paying 739 of an 873 episode for
    #:    one roll it then banked by freezing; `arm_work` (load x dz) buys a
    #:    LEAN — at w=35 the floor peak rises 0.449 -> 0.477 while uprightness
    #:    at that peak falls +0.34 -> +0.23, and raising `rotate_work` to 30
    #:    against it drives uprightness NEGATIVE (-0.30) at the same height.
    #:
    #: A stand needs trunk 0.62 m AND uprightness +0.85 TOGETHER, and the only
    #: quantity that prices the pair is the PRODUCT the env already computes.
    #: `stand` and `rise` pay its level (W_STAND 8, W_RISE 2) but nothing pays
    #: its RATE, so there is no dense signal for improving it from below —
    #: which is why the floor rise plateaus at 0.465 on a height-versus-
    #: uprightness frontier rather than converging on a stand.
    #:
    #: Signed, so it is dense from the first step and can initiate. Coupled,
    #: so leaning for height scores nothing (the product falls) and rolling
    #: supine scores nothing (height is ~0). Farm-resistant by cancellation:
    #: the integral round a closed cycle is zero.
    RISE_WORK_ENV = "MICRODUCK_ASIMOV_GETUP_RISE_WORK"
    ROTATE_WORK_ENV = "MICRODUCK_ASIMOV_GETUP_ROTATE_WORK"
    TUCK2_ENV = "MICRODUCK_ASIMOV_GETUP_TUCK2"
    TUCK_WORK_ENV = "MICRODUCK_ASIMOV_GETUP_TUCK_WORK"
    ARM_WORK_ENV = "MICRODUCK_ASIMOV_GETUP_ARM_WORK"
    ARM_UNLOAD_ENV = "MICRODUCK_ASIMOV_GETUP_ARM_UNLOAD"
    #: Upward trunk speed, m/s, at or below which arm load is charged.
    ARM_UNLOAD_RISE = 0.05

    FOLD_ENV = "MICRODUCK_ASIMOV_GETUP_FOLD_PAY"
    #: Joint-error scale, rad. The pay is `exp(-mean|err| / this)`.
    FOLD_TOL = 0.10
    #: Arrival lift ceiling, m/s. NOT the basin edge — see the clause above;
    #: the re-measured edge is ~0.0 and this value is 3x too loose.
    FOLD_LIFT = 0.20
    #: Span ceiling, m. `fold` itself is 0.215.
    FOLD_SPAN = 0.30
    #: Minimum trunk height, m — a little under `fold`'s own 0.511, because
    #: the joints alone are satisfied at 0.235 m on the forearms.
    FOLD_Z = 0.46
    #: Minimum uprightness. `fold` is +0.83; this leaves a little margin
    #: without admitting the +0.56 the trained chain was reaching.
    FOLD_UPRIGHT = 0.70

    ATTITUDE_ENV = "MICRODUCK_ASIMOV_GETUP_ATTITUDE"
    #: Uprightness the hold aims at — `fold`'s own (+0.83), not a stand's.
    ATTITUDE_TARGET = 0.83
    #: Hard clamp, N.m. Constant torques at 80 and above launched this robot
    #: when combined with a 0.55 lift, so the clamp is below that and the
    #: default gain reaches it only at full inversion.
    ATTITUDE_MAX = 60.0
    #: Rate damping, N.m per rad/s. Without it a PD on a 32 kg trunk with a
    #: 60 N.m authority oscillates at the step rate.
    ATTITUDE_DAMP = 8.0
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
                 history: int | None = None,
                 action_clip: float | None = None,
                 getup_blend: float | None = None,
                 arm_push: float | None = None,
                 arm_work: float | None = None,
                 tuck_work: float | None = None,
                 tuck2: float | None = None,
                 rotate_work: float | None = None,
                 rise_work: float | None = None,
                 arm_effort: float | None = None,
                 head_up: float | None = None,
                 start_rung: str | None = None,
                 max_span: float | None = None,
                 fall_depth: float | None = None,
                 launch_span: float | None = None,
                 span_cost: float | None = None,
                 span_target: float | None = None,
                 ground_cost: float | None = None,
                 balance_cost: float | None = None,
                 home_pay: float | None = None,
                 joint_vel_cost: float | None = None,
                 over_clip_cost: float | None = None,
                 limit_cost: float | None = None,
                 push_bw: float | None = None,
                 calm_cost: float | None = None,
                 swing_pay: float | None = None,
                 handoff_pay: float | None = None,
                 rotate_pay: float | None = None,
                 roll_pay: float | None = None,
                 fell_cut: bool | None = None,
                 push_every: float | None = None,
                 rise_vel_cost: float | None = None,
                 tuck_pay: float | None = None,
                 prologue: str | None = None,
                 getup_assist: float | None = None,
                 getup_attitude: float | None = None,
                 span_axis: str | None = None,
                 span_band: str | None = None,
                 fold_pay: float | None = None,
                 arm_unload: float | None = None,
                 **kwargs):
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
        # BEFORE anything reads `observation_space`: at N > 1 this widens it.
        self._set_history(history)

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
        self._attitude = float(
            getup_attitude if getup_attitude is not None
            else os.environ.get(self.ATTITUDE_ENV, "0.0") or 0.0)
        self._span_axis = str(
            span_axis if span_axis is not None
            else os.environ.get(self.SPAN_AXIS_ENV, "") or "both").lower()
        _b = (span_band if span_band is not None
              else os.environ.get(self.SPAN_BAND_ENV, "") or "")
        try:
            self._span_band = (tuple(float(x) for x in str(_b).split(","))
                               if _b else None)
        except ValueError:
            self._span_band = None
        self._fold_pay = float(
            fold_pay if fold_pay is not None
            else os.environ.get(self.FOLD_ENV, "0.0") or 0.0)
        self._arm_unload = float(
            arm_unload if arm_unload is not None
            else os.environ.get(self.ARM_UNLOAD_ENV, "0.0") or 0.0)
        self._roll_pay = float(roll_pay if roll_pay is not None
                               else os.environ.get(self.ROLL_ENV, "") or 0.0)
        self._rise_vel_cost = float(
            rise_vel_cost if rise_vel_cost is not None
            else os.environ.get(self.RISE_VEL_ENV, "") or 0.0)
        self._push_every = float(push_every if push_every is not None
                                 else os.environ.get(self.PUSH_EVERY_ENV, "")
                                 or 0.0)
        self._fell_cut = bool(int(fell_cut) if fell_cut is not None
                              else int(os.environ.get(self.FELL_CUT_ENV, "")
                                       or 1))
        self._rotate_pay = float(rotate_pay if rotate_pay is not None
                                 else os.environ.get(self.ROTATE_ENV, "")
                                 or 0.0)
        self._best_upright = None
        self._best_supine = None
        self._torso_bid = getattr(self, '_torso_bid', None)
        self._tuck_pay = float(tuck_pay if tuck_pay is not None
                               else os.environ.get(self.TUCK_ENV, "") or 0.0)
        self._best_tuck = None
        self._prologue = str(prologue if prologue is not None
                             else os.environ.get(self.PROLOGUE_ENV, "")
                             or "").strip().lower()
        self._prologue_plan = self._build_prologue() if self._prologue else []
        self._prologue_i = 0
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
        #: Foot-to-foot span, metres, past which the episode ENDS.
        #:
        #: Measured on teach-asimov_getup_clip-79ef4d, which rises 288/288:
        #: it sprawls its feet a MEDIAN of 0.989 m apart and up to 1.042 m, on
        #: a 1.2 m robot whose standing stance is 0.215 m. That is the splits,
        #: and it is the route Jonathan ruled out at the start of this track —
        #: "rather than doing the worm or some kind of break-dancing move,
        #: both which are dangerous".
        #:
        #: A CUT and not a penalty, because the lesson on this robot is that
        #: an unsampled state's value is never learned: a penalty leaves the
        #: sprawl in the rollouts and merely prices it, while a cut removes it
        #: from the distribution. 0.80 m by construction, not by taste — the
        #: splayed feet-only rungs the ladder has to pass through sit at
        #: 0.59-0.74 m, so this admits every pose in the cache and forbids the
        #: 0.99 m scramble. Set 0 or leave unset to disable.
        self._max_span = float(max_span if max_span is not None
                               else os.environ.get(self.SPAN_ENV, "") or 0.0)
        self._check_span_vs_starts = self._max_span > 0.0
        #: Span allowed while the trunk is between `pike` and `fold` — the
        #: BALLISTIC gap, the only stretch anything has ever crossed and the
        #: only stretch it crosses by sprawling. 0 (default) = the flat cut
        #: everywhere.
        #:
        #: Measured: every one of the teacher's 24 demonstrations reaches its
        #: maximum foot span (median 0.929 m) INSIDE this window. Training the
        #: handover under a flat 0.75 m cut therefore forbade the only known
        #: solution and then scored 0/24 — a ban mistaken for an impossibility.
        self._launch_span = float(launch_span if launch_span is not None
                                  else os.environ.get(self.LAUNCH_SPAN_ENV, "")
                                  or 0.0)
        #: The CONTINUOUS span charge, per metre above `span_target`, and the
        #: width it is measured from. 0 (the default) leaves the reward
        #: exactly as every earlier run had it.
        #:
        #: Sized against the economy rather than chosen: the rise earns about
        #: 13.5 a step (5422.8 over 400) and doing nothing earns 0.1, so a
        #: charge of 6.0 on the 0.59 m by which the donor's 0.99 m rise
        #: exceeds a 0.40 m stance costs 3.5 a step — enough that narrowing
        #: pays, and far too little to make lying down the better deal. A
        #: penalty that CAN eat its own task is the failure recorded in
        #: "A ramped penalty can eat its task".
        self._span_cost = float(span_cost if span_cost is not None
                                else os.environ.get(self.SPAN_COST_ENV, "")
                                or 0.0)
        self._span_target = float(span_target if span_target is not None
                                  else os.environ.get(self.SPAN_TARGET_ENV, "")
                                  or 0.40)
        #: The flight charge. Off by default (0.0), so every run before
        #: 2026-10-09 reproduces exactly.
        self._ground_cost = float(ground_cost if ground_cost is not None
                                  else os.environ.get(self.GROUND_ENV, "")
                                  or 0.0)
        #: The balance charge. Off by default (0.0).
        self._balance_cost = float(balance_cost if balance_cost is not None
                                   else os.environ.get(self.BALANCE_ENV, "")
                                   or 0.0)
        #: Come home, and stop jamming. Both off by default (0.0).
        self._home_pay = float(home_pay if home_pay is not None
                               else os.environ.get(self.HOME_ENV, "") or 0.0)
        #: Limb speed and out-of-clip commands. Default to the class
        #: constants, so every earlier run reproduces exactly.
        self._joint_vel_cost = float(
            joint_vel_cost if joint_vel_cost is not None
            else os.environ.get(self.JOINT_VEL_ENV, "") or self.W_JOINT_VEL)
        self._over_clip_cost = float(
            over_clip_cost if over_clip_cost is not None
            else os.environ.get(self.OVER_CLIP_ENV, "") or self.W_OVER_CLIP)
        self._limit_cost = float(limit_cost if limit_cost is not None
                                 else os.environ.get(self.LIMIT_ENV, "")
                                 or 0.0)
        #: Shoves, in bodyweights. Off by default (0.0).
        self._push_bw = float(push_bw if push_bw is not None
                              else os.environ.get(self.PUSH_ENV, "") or 0.0)
        #: Arrive quietly. Off by default (0.0).
        self._calm_cost = float(calm_cost if calm_cost is not None
                                else os.environ.get(self.CALM_ENV, "") or 0.0)
        #: Swing the arms up. Off by default (0.0).
        self._swing_pay = float(swing_pay if swing_pay is not None
                                else os.environ.get(self.SWING_ENV, "") or 0.0)
        #: Train for the handoff. Off by default (0.0).
        self._handoff_pay = float(handoff_pay if handoff_pay is not None
                                  else os.environ.get(self.HANDOFF_ENV, "")
                                  or 0.0)
        self._handoff_streak = 0
        self._handoff_prev_xy = None
        self._prev_sh = None
        self._prev_swing_z = None
        self._swing_lo = None
        self._swing_hi = None
        self._swing_best = 0.0
        self._push_until = -1
        self._push_next = -1
        self._push_vec = np.zeros(3)
        self._pushes = 0
        self._span_cuts = 0
        self._start_rung = str(start_rung if start_rung is not None
                               else os.environ.get(self.START_RUNG_ENV, "")
                               or "").strip().lower()
        self._rungs = None
        self._handover_q = None
        self._handover_v = None
        self._fall_q = None
        self._fall_v = None
        self._fall_groups = None
        self._fall_depth = float(fall_depth if fall_depth is not None
                                 else os.environ.get(self.FALL_DEPTH_ENV, "")
                                 or 1.0)
        if self._start_rung and self._start_rung != "floor":
            from . import asimov_getup_ref as _R

            z = _R.holdable_rungs()
            sup, zz = z["support"], z["trunk_z"]
            if self._start_rung == "arm_top":
                # An EXPLICIT height, not a quantile. As the top quartile this
                # band moved the moment the rung cache changed: it was 24 rungs
                # spanning 0.296-0.440 m when it took the floor rise rate from
                # 0/36 to 16/36, and after the cache was re-validated down to
                # 44 entries the same expression gave 10 rungs at 0.355-0.440 —
                # a different stage under the same name, which is how a
                # curriculum silently stops reproducing.
                #
                # RETUNED 2026-10-07, 0.29 -> 0.22, and the explicit height is
                # why this was caught loudly instead of silently: when
                # `_search_rungs` stopped hardcoding a 13.0 action clip and
                # built the cache at the clip training actually uses, the arm
                # plateau's top fell from 0.440 m to 0.280 m, `>= 0.29` matched
                # NOTHING, and the env raised. (A quantile would have carried
                # on, handing stage 2 the bottom of the plateau under the name
                # "arm_top".) 0.22 restores the band to 20 rungs spanning
                # 0.231-0.280 m — the size it had when it produced this
                # track's only positive result.
                keep = (sup == "arm") & (zz >= 0.16)
            elif self._start_rung.split(":")[0] == "route":
                # STATES A FEASIBLE GET-UP PASSES THROUGH, banded by trunk
                # height: `route` for all of them, `route:LO,HI` for one band.
                #
                # The authored waypoints do not sample the route that works —
                # per-foot CoM distance 0.577 at `pike`, 0.137 at `fold`,
                # nothing between, and uprightness -0.37 (FACE DOWN) at the
                # pike. Twelve reward economies and a 4x budget all failed to
                # cross it. The vendor's policy rises 12/12 from prone IN
                # THIS ENV under the real `dc` envelope, and its route keeps
                # uprightness positive throughout and closes the per-foot
                # distance gradually, so the continuum exists and the
                # reference misses it. See `asimov_getup_ref.route_states`
                # for exactly what is and is not taken (qpos/qvel only, from
                # episodes that ended standing; no actions, no distillation).
                self._rungs = None
                band = self._start_rung.split(":", 1)
                lo, hi = (0.0, 9.9)
                if len(band) == 2 and band[1]:
                    lo, hi = (float(x) for x in band[1].split(","))
                rs = _R.route_states(lo, hi)
                self._fall_q = np.asarray(rs["qpos"])
                self._fall_v = np.asarray(rs["qvel"])
                # STRATIFY BY HEIGHT, for the reason the `falling` rung does
                # and with a worse ratio: the robot STANDS for about 6 s of
                # each 8 s episode, so 25,296 of 29,000 harvested states sit
                # in 0.52-0.62 m and the whole rise is ~450 states. Sampled
                # uniformly, a `route:0.45,0.62` rung draws ~98 % standing
                # frames and teaches a stand the policy already has — measured
                # 14/16 on the band and **0/16** on every transition band
                # under it. Sampling a BAND first and then a state inside it
                # spends equal time at every height.
                rz = np.asarray(rs["trunk_z"])
                edges = np.linspace(float(rz.min()), float(rz.max()) + 1e-6, 9)
                bands = [np.flatnonzero((rz >= a) & (rz < b))
                         for a, b in zip(edges[:-1], edges[1:])]
                self._fall_groups = [b for b in bands if len(b)]
                keep = None
            elif self._start_rung == "pike":
                # the AUTHORED pike (0.215 m span), not an arm-plateau rung
                # (0.70 m) — the hybrid keeps the narrow stance outside the
                # launch window and those are different poses.
                self._rungs = None
                ps = _R.pike_states()
                self._fall_q = np.asarray(ps["qpos"])
                self._fall_v = np.asarray(ps["qvel"])
                self._fall_groups = None
                keep = None
            elif self._start_rung == "seat":
                # THE LONG SIT — the teacher's own signature and the one pose
                # the authored route never had. The push-up ends face DOWN
                # (uprightness -0.37 at `pike`) where hip flexion swings a leg
                # 751-1095 mm INTO THE AIR, so the feet can never come under
                # the mass; the teacher instead reaches +0.49 at 0.06-0.12 m
                # and rises already upright. Settled, these sit at trunk
                # 0.43 m, uprightness +0.93, on both feet at a 0.215 m span —
                # above `rock_back` and more upright than `fold`, and nothing
                # trained so far stands from them (s8: 0/24), which is exactly
                # why they are a start to train ON rather than a rung.
                self._rungs = None
                q, v = _R.seat_states()
                self._fall_q = np.asarray(q)
                self._fall_v = np.asarray(v)
                self._fall_groups = None
                keep = None
            elif self._start_rung == "falling":
                # A REVERSE CURRICULUM IN PERTURBATION SPACE, not pose space.
                # The policy stands 16/16 from `fold` and 0/16 from
                # `rock_back` even trained on it; the move is decided by the
                # contact mode and the contact mode by momentum. But the FALL
                # is free — shove a settled fold and it drops through the whole
                # gap in a second — so these states are reachable BY
                # CONSTRUCTION and carry the velocity they really have. Three
                # interpolated pose ladders failed because their rungs were
                # averages of two poses and fell on their own; a state the
                # robot demonstrably passes through cannot have that defect.
                self._rungs = None
                f = _R.fall_states()
                fz = _R.BY_NAME["fold"].settled_z
                pz = _R.BY_NAME["pike"].settled_z
                cut = fz - self._fall_depth * (fz - pz)
                sel = np.asarray(f["trunk_z"]) >= cut
                if not sel.any():
                    raise RuntimeError(
                        f"no fall states at depth {self._fall_depth} "
                        f"(cut {cut:.3f} m)")
                self._fall_q = np.asarray(f["qpos"])[sel]
                self._fall_v = np.asarray(f["qvel"])[sel]
                # STRATIFY BY HEIGHT, because the fall is FAST through the gap
                # and slow near the top: 89 of 134 cached states sit within
                # 4 cm of `fold`. Sampled uniformly over states, the deep end
                # is barely practised — measured, the first version of this
                # curriculum held 20/48 overall and **0 of the 14 starts below
                # 0.45 m**, i.e. it reproduced the fold-holding the policy
                # already had and learned nothing at depth. Sampling a BAND
                # first and then a state inside it spends equal time at every
                # height.
                fz_ = np.asarray(f["trunk_z"])[sel]
                edges = np.linspace(float(fz_.min()), float(fz_.max()) + 1e-6, 7)
                bands = [np.flatnonzero((fz_ >= lo) & (fz_ < hi))
                         for lo, hi in zip(edges[:-1], edges[1:])]
                self._fall_groups = [b for b in bands if len(b)]
                keep = None
            elif self._start_rung == "handover":
                # THE SCRIPTED RAMP'S OUTPUT, not a rung. The arm plateau
                # traverses open loop (phase 3i), so the bottom of the get-up
                # is PLAYED and the policy only has to learn the one
                # transition above it. These states carry their VELOCITY —
                # `_rung_qvel` below — because a settled rung hands the policy
                # a robot at rest and the gap above is ballistic.
                self._rungs = None
                h = _R.handover_states()
                self._handover_q = np.asarray(h["qpos"])
                self._handover_v = np.asarray(h["qvel"])
                keep = None
            elif self._start_rung in ("arm", "feet"):
                keep = sup == self._start_rung
            elif self._start_rung == "low":
                # The bottom of the ladder: every rung within 15 cm of the
                # floor. This is the band that was missing — a search filter
                # requiring trunk > 0.15 m had excluded it, so the first stage
                # of the curriculum had nothing to aim at between the fallen
                # pose (~0.11 m) and 0.21 m, and it stalled at 0.203 m.
                # <= 0.30 and not 0.26, read off the validated distribution:
                # the rungs sit at 0.146, 0.161, then nothing until 0.252, then
                # a dense cluster. A 0.26 threshold caught 5 rungs and missed
                # the cluster's bottom; 0.30 catches 19 and spans the 9 cm hole
                # from 0.161 to 0.252 — which is the gap the policy is actually
                # stuck in, since the old stage 1 reached 0.203 m, past both
                # low rungs and below the next foothold.
                keep = zz <= 0.30
            elif self._start_rung in ("mix_top", "mix_arm", "mix_low"):
                # A BAND, mixed with the floor. Measured, these two facts have
                # to hold at once and only this shape does both:
                #
                #  * the stage that teaches the rise is the one that starts
                #    HIGH. Across two chains, `arm_top` (0.30-0.44 m) took the
                #    floor rise rate 0/36 -> **16/36** in a single stage, while
                #    a `mix` stage and a `low` stage each left it at 0/36. A
                #    near-goal start is the lever; starting the policy low
                #    leaves the whole hard part still ahead of it.
                #  * the floor's own states must stay in the normalizer's
                #    distribution, because a warm start FREEZES it
                #    (`venv.training = False`): the floor reaches |z| 13.1
                #    under floor-trained stats while the rungs sit at 8.4 and
                #    3.9, so a chain that only ever starts high and then
                #    descends hands its last stage inputs that mean something
                #    else.
                #
                # So every stage draws half its episodes from where the robot
                # FELL and half from its band, and the band descends.
                lo, hi = {"mix_top": (0.30, 9.9), "mix_arm": (0.14, 0.46),
                          "mix_low": (0.14, 0.30)}[self._start_rung]
                keep = (zz >= lo) & (zz <= hi)
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
                    "arm_top, feet, low, mix, handover, falling or pike")
            if keep is None:
                # "handover" is not a rung selection — the scripted ramp's
                # output is cached separately and carries its own velocity.
                pass
            else:
                if not keep.any():
                    raise RuntimeError(
                        f"no {self._start_rung!r} rungs in the cache")
                self._rungs = (z["joints"][keep], z["root_pitch"][keep],
                               zz[keep])
        self._w_rise_work = float(
            rise_work if rise_work is not None
            else os.environ.get(self.RISE_WORK_ENV, "") or 0.0)
        self._w_rotate_work = float(
            rotate_work if rotate_work is not None
            else os.environ.get(self.ROTATE_WORK_ENV, "") or 0.0)
        self._w_tuck2 = float(tuck2 if tuck2 is not None
                              else os.environ.get(self.TUCK2_ENV, "") or 0.0)
        self._w_tuck_work = float(tuck_work if tuck_work is not None
                                  else os.environ.get(self.TUCK_WORK_ENV, "")
                                  or 0.0)
        self._prev_tuck_gap = None
        self._prev_tuck2 = None
        self._prev_upright = None
        self._prev_rise = None
        self._w_arm_work = float(arm_work if arm_work is not None
                                 else os.environ.get(self.ARM_WORK_ENV, "")
                                 or 0.0)
        #: Previous step's trunk height, or None at the start of an episode.
        #: Reset BESIDE `_best_trunk_z` on every reset path: a stale value
        #: makes step 1 read the PREVIOUS episode's final height, which from
        #: 0.009 m lying to 0.520 m at `fold` is a +25 x load spurious payment
        #: on the first step.
        self._prev_trunk_z = None
        self._prev_tuck_gap = None
        self._prev_tuck2 = None
        self._prev_upright = None
        self._prev_rise = None
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
        # The observation ring starts the episode empty and NOT live, so every
        # `_get_obs` during this reset rebuilds it from the current frame
        # instead of advancing it (see `_get_obs` — reset reads it more than
        # once). `_obs_live` is what makes that correct; the clear here is
        # belt-and-braces and the ORDER of the two against `super()` does not
        # matter, which was checked by planting the reverse and watching the
        # test still pass.
        self._obs_stack = []
        self._obs_live = False
        obs, info = super().reset(seed=seed, options=options)
        if getattr(self, "_prologue_plan", None):
            # The push-up starts where the push-up starts. Rewind the script
            # and put the robot face down; everything below it (rungs, fall
            # states, a start_rung) would be a different episode entirely, so
            # the prologue OWNS the start when it is on.
            from . import asimov_getup_ref as _R
            self._prologue_i = 0
            w = _R.BY_NAME[self.PROLOGUE_ROUTE[0]]
            self._place(np.asarray(w.joints, float), w.root_pitch)
            self.data.qvel[:] = 0.0
            self.data.xfrc_applied[:] = 0.0
            self.data.xfrc_applied[self.trunk_body_id, 2] = self._assist_n
            mujoco.mj_forward(self.model, self.data)
            return self._get_obs(), info
        if getattr(self, "_fall_q", None) is not None:
            groups = getattr(self, "_fall_groups", None)
            if groups:
                g = groups[int(self._rng.integers(len(groups)))]
                i = int(g[int(self._rng.integers(len(g)))])
            else:
                i = int(self._rng.integers(len(self._fall_q)))
            self.data.qpos[:] = self._fall_q[i]
            self.data.qvel[:] = self._fall_v[i]      # the velocity is the point
            self.data.xfrc_applied[:] = 0.0
            self.data.xfrc_applied[self.trunk_body_id, 2] = self._assist_n
            mujoco.mj_forward(self.model, self.data)
            self.twist_cmd[:] = 0.0
            h0 = min(max(self.data.xpos[self.trunk_body_id][2]
                         / self._stand_z, 0.0), 1.0)
            self._best_rise = h0 * max(float(-self._projected_gravity()[2]), 0.0)
            self._reset_episode()
            return self._get_obs(), {}

        if getattr(self, "_handover_q", None) is not None:
            # THE SCRIPTED RAMP HANDS OVER HERE. qvel is written too, and that
            # is the whole difference from a rung: a settled rung starts the
            # policy at rest, and `asimov_getup_floor` stage 2 does exactly
            # that and rises 0/24 across a gap that is ballistic.
            i = int(self._rng.integers(len(self._handover_q)))
            self.data.qpos[:] = self._handover_q[i]
            self.data.qvel[:] = self._handover_v[i]
            self.data.xfrc_applied[:] = 0.0
            self.data.xfrc_applied[self.trunk_body_id, 2] = self._assist_n
            mujoco.mj_forward(self.model, self.data)
            self.twist_cmd[:] = 0.0
            h0 = min(max(self.data.xpos[self.trunk_body_id][2]
                         / self._stand_z, 0.0), 1.0)
            self._best_rise = h0 * max(float(-self._projected_gravity()[2]), 0.0)
            self._reset_episode()
            return self._get_obs(), {}

        use_rung = self._rungs is not None
        if use_rung and self._start_rung.startswith("mix"):
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
            self._reset_episode()
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
        self._prev_trunk_z = None
        self._prev_tuck_gap = None
        self._prev_tuck2 = None
        self._prev_upright = None
        self._prev_rise = None
        self._best_upright = None
        self._best_supine = None
        self._torso_bid = getattr(self, '_torso_bid', None)
        self._best_tuck = None
        self._reset_push()
        self._best_head = None
        return self._get_obs(), info


    def _build_prologue(self) -> list:
        """The push-up as a list of ACTIONS, precomputed once per env.

        Actions, not poses, because `step` has to substitute them for the
        policy's output and the action is what the servos see. A smoothstep
        ramp between consecutive waypoints at the measured tempo, with a short
        dwell so each new contact set settles before the next leg.
        """
        from . import asimov_getup_ref as _R

        if self._prologue not in ("pushup", "push_up", "1", "true"):
            raise ValueError(
                f"prologue={self._prologue!r} is not a route — "
                f"use 'pushup' or leave it unset")
        scale = float(np.atleast_1d(self.robot.action_scale)[0])
        default = np.asarray(self.robot.default_pose, float)
        clip = float(self._act_clip)
        route = [_R.BY_NAME[n] for n in self.PROLOGUE_ROUTE]
        plan = []
        cur = np.asarray(route[0].joints, float)
        for _ in range(int(self.PROLOGUE_DWELL_S * 50)):
            plan.append(np.clip((cur - default) / scale,
                                -clip, clip).astype(np.float32))
        for w in route[1:]:
            tgt = np.asarray(w.joints, float)
            n = max(2, int(self.PROLOGUE_LEG_S * 50))
            for k in range(n):
                f = (k + 1) / n
                f = f * f * (3.0 - 2.0 * f)
                q = cur + (tgt - cur) * f
                plan.append(np.clip((q - default) / scale,
                                    -clip, clip).astype(np.float32))
            cur = tgt
            for _ in range(int(self.PROLOGUE_DWELL_S * 50)):
                plan.append(np.clip((cur - default) / scale,
                                    -clip, clip).astype(np.float32))
        return plan

    def in_prologue(self) -> bool:
        """True while the scripted push-up is driving, not the policy."""
        return bool(self._prologue_plan) and self._prologue_i < len(
            self._prologue_plan)

    def _refuse_illegal_joints(self, q, adr) -> None:
        """A pose outside `jnt_range` is not a pose — refuse it here.

        `qpos` is NOT clamped on assignment, so a placement past a stop is
        accepted silently and the constraint solver then fights it for the
        rest of the episode, which looks exactly like an unstable pose. On
        2026-10-09 a search for a kneel ran ~26 000 samples with the knee at
        -1.2 to -2.0 rad while that joint's range is **+0.00 to +1.50 and it
        bends one way only**; every one of those poses was illegal, and the
        toppling they showed was the solver ejecting them. `add-a-robot`
        already says to check the joint limits BEFORE authoring a pose; this
        makes skipping it impossible.
        """
        m = self.model
        bad = []
        for i, a in enumerate(np.atleast_1d(adr)):
            j = int(np.argmax(m.jnt_qposadr == a))
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            if hi > lo and not (lo - 1e-6 <= float(q[i]) <= hi + 1e-6):
                bad.append(f"{m.joint(j).name}={float(q[i]):+.3f} "
                           f"outside [{lo:+.3f}, {hi:+.3f}]")
        if bad:
            raise ValueError("pose is outside the model's joint limits: "
                             + "; ".join(bad))

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
        adr = getattr(self, "_q_adr", self.joint_qpos_adr)
        self._refuse_illegal_joints(np.asarray(q), adr)
        d.qpos[adr] = np.asarray(q)
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

    def _com_foot_gap(self) -> float:
        """Horizontal distance from the centre of mass to the feet midpoint."""
        import mujoco as _mj

        _mj.mj_comPos(self.model, self.data)
        com = np.asarray(self.data.subtree_com[0], float)
        ids = getattr(self, "_span_body_ids", None)
        if ids is None:
            ids = tuple(int(self.model.geom_bodyid[self.foot_geoms[s]])
                        for s in ("left", "right"))
            self._span_body_ids = ids
        mid = (np.asarray(self.data.xpos[ids[0]], float)
               + np.asarray(self.data.xpos[ids[1]], float)) / 2.0
        return float(np.hypot(com[0] - mid[0], com[1] - mid[1]))

    def _com_foot_per(self) -> float:
        """Mean horizontal distance from the CoM to EACH foot.

        Unlike the midpoint, spreading the feet around the mass makes this
        WORSE, which is the whole point -- see `TUCK2_ENV`.
        """
        import mujoco as _mj

        _mj.mj_comPos(self.model, self.data)
        com = np.asarray(self.data.subtree_com[0], float)
        ids = getattr(self, "_span_body_ids", None)
        if ids is None:
            ids = tuple(int(self.model.geom_bodyid[self.foot_geoms[s]])
                        for s in ("left", "right"))
            self._span_body_ids = ids
        d = [float(np.hypot(com[0] - self.data.xpos[i][0],
                            com[1] - self.data.xpos[i][1])) for i in ids]
        return float(np.mean(d))

    def _rise_work_gain(self) -> float:
        """Signed rate of change of `height * max(upright, 0)`, per second.

        The SAME product the env scores with `rise`/`stand`, differenced
        instead of levelled. `_prev_rise` is checked with `is None` rather
        than read through a `getattr` default, for the reason in
        `_arm_work_gain`.
        """
        if self._w_rise_work <= 0.0:
            return 0.0
        z = float(self.data.xpos[self.trunk_body_id][2])
        h = min(max(z / self._stand_z, 0.0), 1.0)
        u = max(float(-self._projected_gravity()[2]), 0.0)
        r = h * u
        prev = self._prev_rise
        self._prev_rise = r
        if prev is None:
            return 0.0
        return (r - prev) * 50.0

    def _rotate_work_gain(self) -> float:
        """Signed rate of change of uprightness, per second.

        Positive while the trunk rotates toward vertical, negative while it
        falls away. `_prev_upright` is checked with `is None` rather than read
        through a `getattr` default, for the reason in `_arm_work_gain`.
        """
        if self._w_rotate_work <= 0.0:
            return 0.0
        u = float(-self._projected_gravity()[2])
        prev = self._prev_upright
        self._prev_upright = u
        if prev is None:
            return 0.0
        return (u - prev) * 50.0

    def _tuck2_gain(self) -> float:
        """Signed rate of closing the mean per-foot CoM distance, m/s."""
        if self._w_tuck2 <= 0.0:
            return 0.0
        g = self._com_foot_per()
        prev = self._prev_tuck2
        self._prev_tuck2 = g
        if prev is None:
            return 0.0
        return (prev - g) * 50.0

    def _tuck_work_gain(self) -> float:
        """Signed rate of closing the CoM-to-feet gap, m/s.

        Positive while the feet come under the mass, negative while the mass
        runs away from them. `_prev_tuck_gap` is checked with `is None` rather
        than read through a `getattr` default, for the reason spelled out in
        `_arm_work_gain`.
        """
        if self._w_tuck_work <= 0.0:
            return 0.0
        gap = self._com_foot_gap()
        prev = self._prev_tuck_gap
        self._prev_tuck_gap = gap
        if prev is None:
            return 0.0
        return (prev - gap) * 50.0

    def _arm_work_gain(self) -> float:
        """Signed work through the arms this step, load-weighted, per second.

        `_prev_trunk_z` is initialised to `None` and checked with `is None`,
        NOT read through `getattr(self, "_prev_trunk_z", 0.0)` — that default
        returns **None** when the attribute exists and IS None, which made
        `float - None` the first arithmetic of every rollout and sat a trainer
        at 0 of 3,000,000 steps for 16 minutes while the lab reported
        `running: True` (AGENTS.md, "Running is not progressing").
        """
        if self._w_arm_work <= 0.0:
            return 0.0
        z = float(self.data.xpos[self.trunk_body_id][2])
        prev = self._prev_trunk_z
        self._prev_trunk_z = z
        if prev is None:
            return 0.0
        return min(self._arm_load(), 1.0) * (z - prev) * 50.0

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

    def _tuck_gain(self) -> float:
        """Monotone gain from bringing the FEET UNDER THE MASS.

        The horizontal distance from the centre of mass to the midpoint of
        the feet, best-so-far. Reducing it is the whole of "get your feet
        under you", and it is the one move every other term charges for:
        tucking lowers the trunk and costs uprightness on the way.
        """
        import mujoco as _mj

        _mj.mj_comPos(self.model, self.data)
        com = np.asarray(self.data.subtree_com[0], float)
        ids = getattr(self, "_span_body_ids", None)
        if ids is None:
            ids = tuple(int(self.model.geom_bodyid[self.foot_geoms[s]])
                        for s in ("left", "right"))
            self._span_body_ids = ids
        mid = (np.asarray(self.data.xpos[ids[0]], float)
               + np.asarray(self.data.xpos[ids[1]], float)) / 2.0
        gap = float(np.hypot(com[0] - mid[0], com[1] - mid[1]))
        best = self._best_tuck
        if best is None:
            self._best_tuck = gap
            return 0.0
        gain = max(best - gap, 0.0)
        self._best_tuck = min(best, gap)
        return gain

    def _upright_gain(self) -> float:
        """Monotone gain in UPRIGHTNESS, the mirror of `_progress`.

        Only motion that takes the trunk more upright than its own best this
        episode. A per-step rate would be farmable by rocking, exactly as
        plain up-speed was — bouncing scored 32.5 and random actions 35.4
        against a real get-up's 57.2.
        """
        u = float(-self._projected_gravity()[2])
        best = self._best_upright
        if best is None:
            self._best_upright = u
            return 0.0
        gain = max(u - best, 0.0)
        self._best_upright = max(best, u)
        return gain

    def _supine_axis(self) -> float:
        """`f_z`: the world-z component of the TORSO's forward axis.

        -1 face down, +1 face up, ~0 on a side — Menlo's own
        `label_fallen_states` takes the argmax of (f_z, -f_z, -l_z, l_z), so
        this is the axis "supine" is decided on. Read off the torso, not the
        trunk/pelvis body, for the same reason the classifier does.
        """
        import mujoco as _mj
        if getattr(self, "_torso_bid", None) is None:
            from . import asimov_getup as _G
            bid = _mj.mj_name2id(
                self.model, _mj.mjtObj.mjOBJ_BODY, _G.TORSO_BODY)
            # `mj_name2id` returns -1 for a name this model does not have,
            # and -1 is not None — so an unguarded cache would accept it and
            # `xmat[-1]` would silently read the LAST body, paying `roll_pay`
            # on an unrelated link with no error. The same shape as this
            # repo's vendor-link-name trap.
            if bid < 0:
                raise RuntimeError(
                    f"this model has no {_G.TORSO_BODY!r} body, so the supine "
                    "axis cannot be read — `roll_pay` must stay 0 on it")
            self._torso_bid = int(bid)
        return float(self.data.xmat[self._torso_bid].reshape(3, 3)[2, 0])

    def _roll_gain(self) -> float:
        """Monotone gain in `f_z` — the mirror of `_upright_gain`.

        Only motion that turns the chest further skyward than its own best
        this episode. A per-step rate would be farmable by rocking, which is
        how plain up-speed was farmed (bouncing scored 32.5 and random
        actions 35.4 against a real get-up's 57.2), and `swing` was farmed at
        1807 an episode before it was made monotone.
        """
        f = self._supine_axis()
        best = self._best_supine
        if best is None:
            self._best_supine = f
            return 0.0
        gain = max(f - best, 0.0)
        self._best_supine = max(best, f)
        return gain

    def _regulariser_terms(self, up: bool) -> dict[str, float]:
        """The two charges that CANNOT eat a get-up, on every get-up task.

        `joint_vel_penalty` is gated on `up`, so it is identically zero for
        the whole rise; `over_clip_penalty` is exactly zero inside the action
        clip, so every unit it charges is a command the robot could never
        have executed. Neither can price the lifting phase, which is what
        makes them safe to sum on a reference-tracking task that has its own
        objective.

        **The other two inline charges are NOT here, and that is deliberate.**
        `arm_effort_penalty` is excluded from the tracking tasks by
        measurement — `quadruped` and `pike` are unreachable on this body
        without the elbows at **58-66 % of rated torque**, so the penalty
        charged the robot for the only phase that lifts it, and the torque
        envelope is a phase-4 measurement rather than a reward term
        (`test_the_arm_penalty_is_gone_from_the_tracking_reward`).
        `rise_vel_penalty` is the same objection by construction: it is gated
        on `not up`, which is precisely the rise. A first version of this
        helper returned all four and re-introduced both; the ref env's two
        exact-set tests caught it.

        They were written inline in `AsimovGetupEnv._compute_reward` and
        nowhere else, so the reference and clip tasks — **209 of 449**
        get-up runs — summed neither. Both subclass dicts already carry a
        comment about this happening a fourth time (`span_penalty`,
        `arm_effort`, `arm_push`, `tuck`); these were the fifth. Measured on
        `teach-asimov_getup_whole-c4bdba`:

          * raw |a| exceeded the 4.0 clip on **37 % of steps**, reaching
            **16.0** — four times a command the robot could ever execute —
            because `over_clip_penalty` is what makes unbounded growth cost
            anything, and it was not summed.
          * while STANDING, limb rate ran a median **326 deg/s** against the
            shipped walker's **3 deg/s** in the same posture. That is the
            jiggle Jonathan saw at the end of the render, and no amount of
            `home` or `calm` reaches it: `home` pays joint POSITION and
            `_calm_charge` prices TRUNK speed, so both are blind to a limb
            twitching at punch speed under a still torso.

        Returning them from one place is the fix: a subclass that writes
        `**self._regulariser_terms(up)` cannot omit three of the four, and
        `test_every_getup_task_charges_the_ROBOTS_own_costs` fails the build
        if a dict drops the call.
        """
        pen = self._penalty_scale()
        joint_vel = float(np.square(self._joint_vel()).mean())
        return {
            "joint_vel_penalty": (-self._joint_vel_cost * joint_vel
                                  * float(up) * pen),
            "over_clip_penalty": -self._over_clip_cost * float(np.square(
                np.maximum(np.abs(self.last_action) - self._act_clip, 0.0)
            ).mean()),
        }

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
            # The foot-span charge. It belongs on EVERY get-up reward, and
            # adding it to the two subclasses and not this base is how the
            # descend curriculum was first priced with a charge of exactly
            # 0.0 while `span_cost=80` sat in its kwargs.
            "span_penalty": -self._span_charge(),
            "flight_penalty": -self._flight_charge(),
            "fold_arrival": self._fold_arrival_gain(),
            "arm_unload_penalty": -self._arm_unload_charge(),
            # NOTE: `arm_push` is defined ONCE in this dict, below. A second
            # copy was added here on 2026-10-19 and silently ZEROED it: a
            # dict literal evaluates BOTH values for a duplicate key, and
            # `_trunk_up_speed()` is MONOTONE — the first call banks the
            # new-best gain and the second returns 0.0, so the surviving
            # entry read zero. The subclasses genuinely were missing it.
            "balance_penalty": -self._balance_charge(),
            "home": self._home_gain(),
            "calm_penalty": -self._calm_charge(),
            "swing": self._swing_gain(),
            "limit_penalty": -self._limit_charge(),
            "progress": self.W_PROGRESS * gain,
            # ROTATION, the axis the task actually turns on. Off by default
            # (0.0), so every earlier run reproduces exactly.
            "rotate": self._rotate_pay * self._upright_gain(),
            # GET THE FEET UNDER THE MASS. Off by default (0.0).
            "tuck": self._tuck_pay * self._tuck_gain(),
            # ROLL ONTO THE BACK. Off by default (0.0). See `ROLL_ENV`.
            "roll": self._roll_pay * self._roll_gain(),
            "on_back": (self._roll_pay * self.ROLL_HOLD_W
                        * float(self._supine_axis() >= self.ROLL_DONE)),
            "standing": self.W_STANDING * float(up),
            "action_rate_penalty": -self.W_ACTION_RATE * rate * pen,
            # The same limb speed while NOT up, which is where all the speed
            # actually is (see `RISE_VEL_ENV`), and the arm-torque charge.
            # Both stay on THIS task only: they price the lifting phase, so
            # `_regulariser_terms` deliberately leaves them out of the
            # reference tasks. See its docstring for the measurement.
            "rise_vel_penalty": (-self._rise_vel_cost * joint_vel
                                 * float(not up) * pen),
            "arm_effort_penalty": -self._w_arm_effort * arm_effort * pen,
            **self._regulariser_terms(up),
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
            "arm_work": self._w_arm_work * self._arm_work_gain(),
            "tuck_work": self._w_tuck_work * self._tuck_work_gain(),
            "tuck2": self._w_tuck2 * self._tuck2_gain(),
            "rotate_work": self._w_rotate_work * self._rotate_work_gain(),
            "rise_work": self._w_rise_work * self._rise_work_gain(),
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

    def _span_charge(self) -> float:
        """What this step's stance costs, in reward, above `span_target`.

        One-sided on purpose — AGENTS.md's rule for body terms — so a stance
        NARROWER than the target is not paid for being narrow and the robot
        has no reason to bring its feet together past a human stance.
        """
        if self._span_cost <= 0.0:
            return 0.0
        if self._span_band is not None:
            lo, hi = self._span_band[0], self._span_band[-1]
            z = float(self.data.xpos[self.trunk_body_id][2])
            if not (lo <= z < hi):
                return 0.0
        return self._span_cost * max(
            0.0, self._foot_span_axis(self._span_axis) - self._span_target)

    def _foot_span_axis(self, axis: str = "both") -> float:
        """Foot separation along one axis of the BODY frame, metres.

        "both" is `_foot_span` exactly, so every run trained before this
        existed is unchanged. "lateral" is abduction — the hip loaded
        sideways, which is what "the splits" means and what a narrow
        requirement is actually about. "fore_aft" is the STRIDE.

        The body frame, not the world's: a robot that has turned reads its
        stride as abduction in world coordinates, which is precisely the
        distinction this exists to make.
        """
        import numpy as _np

        if axis not in ("lateral", "fore_aft"):
            return self._foot_span()
        ids = getattr(self, "_span_body_ids", None)
        if ids is None:
            ids = tuple(int(self.model.geom_bodyid[self.foot_geoms[s]])
                        for s in ("left", "right"))
            self._span_body_ids = ids
        v = _np.asarray(self.data.xpos[ids[0]], float) - _np.asarray(
            self.data.xpos[ids[1]], float)
        w, x, y, z = (float(c) for c in self.data.xquat[self.trunk_body_id])
        if axis == "lateral":                      # body +y
            ax = _np.array([2 * (x * y - z * w),
                            1 - 2 * (x * x + z * z),
                            2 * (y * z + x * w)])
        else:                                      # body +x, forward
            ax = _np.array([1 - 2 * (y * y + z * z),
                            2 * (x * y + z * w),
                            2 * (x * z - y * w)])
        return abs(float(v @ ax))

    def _arm_unload_charge(self) -> float:
        """Charge for weight still carried through the arms, ABOVE a height.

        Zero and a no-op when the knob is off, which is the default.

        `_arm_load` is a FRACTION of body weight read from
        `mj_contactForce`, not a boolean, so this charges a forearm bearing
        load and ignores one brushing the floor — which is the distinction
        that matters here, since a hand resting weightless is already off as
        far as the arrival gate is concerned.

        Gated on NOT RISING, not on height: a height gate at 0.35 m fired on
        0.0 % of a real rollout's 1014 frames (the trunk never exceeds 0.317
        m), which is the `fold_pay` failure again. Charging only while the
        trunk is not going up leaves a genuine push free and is reachable on
        98.2 % of frames.
        """
        if self._arm_unload <= 0.0:
            return 0.0
        if float(self.data.qvel[2]) > self.ARM_UNLOAD_RISE:
            return 0.0       # it is PUSHING; that is what the arms are for
        return self._arm_unload * float(self._arm_load())

    def _fold_arrival_gain(self) -> float:
        """Per-step pay for sitting in a state `9702ba` can catch.

        Returns 0.0 when the knob is off, which is the default, so no run
        trained before this existed changes.

        All three clauses are hard gates rather than a weighted sum: a state
        that is close in joint space but airborne, or close and fast, is not
        catchable at all, and a soft sum would pay for two out of three. That
        is how this track ended up with a 97 %-airborne anchor.
        """
        if self._fold_pay <= 0.0:
            return 0.0
        from . import asimov_getup_ref as _R

        parts = [str(x) for x in _R.floor_parts(self)]
        feet = sum(1 for x in parts if "foot" in x)
        if feet < 2 or len(parts) != feet:
            return 0.0
        if self._foot_span() > self.FOLD_SPAN:
            return 0.0
        if abs(float(self.data.qvel[2])) > self.FOLD_LIFT:
            return 0.0
        # THE ROOT CLAUSES, and leaving them out was a real defect: `fold`'s
        # JOINT configuration is reachable at pelvis 0.235 m as well as at its
        # own 0.511 m — a trajectory search hit **0.040 rad** of mean joint
        # error while propped on its forearms at 0.235 m, and the route's
        # arrival at `pike` is already only 0.127 rad away. So the joints do
        # NOT determine the pose; the root does, and a joints-only gate pays
        # nearly full value for a crouch at half the height.
        #
        # The mirror-image error was the one this term was built to fix: a
        # state matching fold's ROOT (pelvis 0.518 m, upright +0.73) with the
        # wrong joints is caught 0/78. Both axes are required, which is
        # `two-maxima-are-not-a-pose` with the two maxima being joints and
        # root instead of height and uprightness.
        if float(self.data.xpos[self.trunk_body_id][2]) < self.FOLD_Z:
            return 0.0
        if float(-self._projected_gravity()[2]) < self.FOLD_UPRIGHT:
            return 0.0
        # Built lazily rather than read off `self._q_adr`, which exists only
        # on the REF subclass: this term lives on the parent so every getup
        # env can use it, and depending on a subclass attribute here is the
        # shape of bug that makes a knob silently dead for one caller.
        if getattr(self, "_fold_q", None) is None:
            from . import asimov_walk as _W
            self._fold_qadr = np.array(
                [int(self.model.joint(n).qposadr[0])
                 for n in _W.JOINT_NAMES], int)
            self._fold_q = np.asarray(_R.BY_NAME["fold"].joints, float)
        q = np.asarray(self.data.qpos[self._fold_qadr], float)
        err = float(np.abs(q - self._fold_q).mean())
        return self._fold_pay * float(np.exp(-err / self.FOLD_TOL))

    def _apply_attitude_assist(self) -> float:
        """Scaffold the trunk's ATTITUDE, as a PD, written into `xfrc_applied`.

        Returns the torque applied, in N.m, so a probe can read it without
        re-deriving it. Zero and a no-op when the knob is off, which is the
        default — `xfrc_applied[trunk, 4]` is not touched at all in that case,
        so every run trained before this existed is bit-identical.

        Index 4 is the torque about y; index 2 is the LIFT and is left alone,
        so the two assists compose. MuJoCo does not clear `xfrc_applied`, and
        the lift is written once at reset, so writing only the torque slot
        here preserves it.
        """
        if self._attitude <= 0.0:
            return 0.0
        up = float(-self._projected_gravity()[2])
        # Proportional to the ERROR, so it fades out as the trunk arrives and
        # never pushes past the target — which is what a constant torque did,
        # sitting the robot up and then carrying on over.
        err = self.ATTITUDE_TARGET - up
        rate = float(self.data.qvel[4]) if self.data.qvel.shape[0] > 4 else 0.0
        tau = -self._attitude * err + self.ATTITUDE_DAMP * rate
        tau = float(np.clip(tau, -self.ATTITUDE_MAX, self.ATTITUDE_MAX))
        self.data.xfrc_applied[self.trunk_body_id, 4] = tau
        return tau

    def _flight_charge(self) -> float:
        """What this step costs if the robot is in the AIR, in reward.

        Zero floor contacts = flight. Asked of MuJoCo's own contact list
        rather than of any height threshold, because the question is whether
        the robot is SUPPORTED, and a 1.2 m body can have every geom above
        50 mm while still resting on a heel.

        Binary per step on purpose. A charge proportional to clearance would
        pay for a LOW hop, and a low hop is the same move.
        """
        if self._ground_cost <= 0.0:
            return 0.0
        return 0.0 if self._on_floor_contacts() else self._ground_cost

    def _home_gain(self) -> float:
        """Per-step pay for being in the robot's OWN default pose, while up.

        Gated on uprightness AND on the feet being the only thing down —
        not on a full stand, so it still pulls through the last part of the
        rise (the feet-only condition holds from ~0.8 s, well before the
        trunk clears the bar), but a KNEEL cannot collect it. Kneeling in
        the default pose is vertical, scores uprightness ~+1, and under an
        uprightness-only gate would have paid full price for never standing
        up — the "a proxy that failure also satisfies" trap, which this
        repo has paid for five times.
        """
        if self._home_pay <= 0.0:
            return 0.0
        from . import asimov_getup_ref as _R
        if float(-self._projected_gravity()[2]) <= _R.STAND_UPRIGHT:
            return 0.0
        if not self._on_feet_only():
            return 0.0
        adr = getattr(self, "_q_adr", self.joint_qpos_adr)
        q = np.asarray(self.data.qpos[adr], float)
        err = np.abs(q - np.asarray(self.robot.default_pose, float))
        # NOTE (2026-10-10): scoring this PER JOINT was tried and reverted.
        # Mean-of-exponentials makes a single 50 deg joint cost LESS than the
        # same total error spread over 23 (9.57 vs 8.27 at weight 10), which
        # is the opposite of the intent. And the premise was wrong anyway:
        # forcing the arms to neutral on a standing policy takes it from
        # 13/24 to **0/24**, so the raised arm is LOAD-BEARING, not sloppy.
        # No pose reward can fix it while the robot arrives at 0.89 m/s.
        # The old note on the mean hiding a joint:
        # which is what this was and which let one joint hide completely.
        # Measured on s11: mean error 13.2 deg while `right_shoulder_pitch`
        # sat **49.4 deg** out (+63.8 against a neutral +14.3) and
        # `right_shoulder_yaw` 37 deg. Averaging first, a 50 deg arm moves a
        # 23-joint mean by 2 deg and costs almost nothing, so the policy was
        # never paid to put the arm down. Jonathan saw it immediately next
        # to the shipped walk policy's neutral stand: *it's not really
        # standing straight with both of the arms down*.
        #
        # Scored this way a joint 50 deg out contributes exp(-4.3) = 0.014
        # of its share whatever the others do, so the gradient points at the
        # WORST joint instead of away from it.
        return self._home_pay * float(np.exp(-err.mean() / self.HOME_TOL))

    def _limit_charge(self) -> float:
        """What this step costs for standing ON its joint stops, in reward.

        Summed over every actuated hinge, in radians past its own range.
        """
        if self._limit_cost <= 0.0:
            return 0.0
        m = self.model
        adr = np.atleast_1d(getattr(self, "_q_adr", self.joint_qpos_adr))
        ids = getattr(self, "_limit_jnt", None)
        if ids is None:
            ids = [int(np.argmax(m.jnt_qposadr == a)) for a in adr]
            self._limit_jnt = ids
        over = 0.0
        for a, j in zip(adr, ids):
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            if hi <= lo:
                continue
            v = float(self.data.qpos[a])
            over += max(0.0, v - hi) + max(0.0, lo - v)
        return self._limit_cost * over

    def handoff_ready(self) -> bool:
        """Is the robot in a state Menlo's walker can take over?

        THE ONE DEFINITION. `behaviors.asimov_tasks._getup_handoff_due`
        delegates here, so the condition the reward trains against and the
        condition the lab deploys cannot drift — the failure this repo has
        under "One definition of success, in the library".
        """
        from . import asimov_getup_ref as _R

        xy = np.asarray(self.data.xpos[self.trunk_body_id][:2], float)
        prev = self._handoff_prev_xy
        self._handoff_prev_xy = xy.copy()
        speed = 0.0 if prev is None else float(np.linalg.norm(xy - prev) * 50.0)
        self._handoff_speed = speed
        ok = _R.is_standing(self) and speed <= self.HANDOFF_SPEED
        self._handoff_streak = (self._handoff_streak + 1) if ok else 0
        return self._handoff_streak >= self.HANDOFF_DWELL

    def _handoff_score(self) -> float:
        """How CATCHABLE this handoff is, 0..1, from the measured basin."""
        v = float(getattr(self, "_handoff_speed", 0.0))
        lo, hi = self.HANDOFF_GOOD, self.HANDOFF_DROP
        return float(np.clip((hi - v) / (hi - lo), 0.0, 1.0))

    def _shoulder_pitch(self) -> np.ndarray:
        """(left, right) shoulder pitch, radians, resolved once."""
        idx = getattr(self, "_sh_adr", None)
        if idx is None:
            m = self.model
            adr = np.atleast_1d(getattr(self, "_q_adr", self.joint_qpos_adr))
            want = {}
            for a in adr:
                j = int(np.argmax(m.jnt_qposadr == a))
                nm = m.joint(j).name
                if nm == "left_shoulder_pitch_joint":
                    want["l"] = int(a)
                elif nm == "right_shoulder_pitch_joint":
                    want["r"] = int(a)
            idx = (want.get("l"), want.get("r"))
            self._sh_adr = idx
        if idx[0] is None or idx[1] is None:
            return np.zeros(2)
        return np.array([float(self.data.qpos[idx[0]]),
                         float(self.data.qpos[idx[1]])])

    def _swing_gain(self) -> float:
        """Pay for swinging BOTH arms the same way, while still rising.

        The shoulders are mirrored in the model, so the same world direction
        is `-left + right`; the ANTI-symmetric part (one up, one down) is a
        balance flail and is deliberately not paid.
        """
        if self._swing_pay <= 0.0:
            return 0.0
        cur = self._shoulder_pitch()
        from . import asimov_getup_ref as _R
        if _R.is_standing(self):
            return 0.0                      # a STAND is never paid for waving
        z = float(self.data.xpos[self.trunk_body_id][2])
        if z >= _R.STAND_FRAC * _R.BY_NAME["stand"].settled_z:
            return 0.0
        # NO RATE BASELINE. An amplitude term does not need one, and keeping
        # it cost two wasted calls at the start of every episode: the first
        # returned early on `prev is None` before the min/max were seeded,
        # so the sweep could not register until the third step.
        # MONOTONE GAIN IN THE SWEEP, the same shape as `progress` and
        # `rotate`, and the third design for this term.
        #
        #   * paid on the RATE alone it was farmed instantly — 1807 an
        #     episode for lying on the floor waving;
        #   * multiplied by upward trunk speed it stopped being farmable and
        #     stopped being LEARNABLE, because a policy that cannot yet rise
        #     has `up` about zero, so the term meant to CAUSE the rise only
        #     paid once the rise existed (arm sweep moved 90.9 -> 96.6 deg,
        #     i.e. nothing);
        #   * paid per step against a per-episode budget, random thrashing
        #     hit the cap in under two seconds, so there was no gradient
        #     toward a BIGGER swing either.
        #
        # What the kip actually needs is AMPLITUDE: 200 deg of sweep lifts
        # the trunk 16 mm and rotates it -0.13 -> -0.05 with the legs
        # passive, while holding the arms overhead without sweeping does
        # nothing at all. So pay the peak-to-peak sweep, best-so-far: a
        # wider sweep earns, the same sweep repeated earns nothing, and the
        # total over an episode is simply `swing_pay x final sweep`.
        if self._ep_k > int(self.SWING_WINDOW_S * 50):
            return 0.0
        pos = float(-cur[0] + cur[1]) / 2.0
        lo = self._swing_lo = (pos if self._swing_lo is None
                               else min(self._swing_lo, pos))
        hi = self._swing_hi = (pos if self._swing_hi is None
                               else max(self._swing_hi, pos))
        sweep = hi - lo
        gain = max(0.0, sweep - self._swing_best)
        self._swing_best = max(self._swing_best, sweep)
        return self._swing_pay * gain

    def _trunk_speed(self) -> float:
        """Horizontal speed of the trunk, m/s, in the world frame.

        `cvel` is in the body's own frame and its linear half is rows 3:6 —
        the velocity-frame trap this repo has a memory about. Taken from
        `xpos` across one control step instead, which is unambiguous.
        """
        tb = self.trunk_body_id
        now = np.asarray(self.data.xpos[tb][:2], float)
        prev = getattr(self, "_prev_trunk_xy", None)
        self._prev_trunk_xy = now.copy()
        if prev is None:
            return 0.0
        return float(np.linalg.norm(now - prev) * 50.0)

    def _calm_charge(self) -> float:
        """What this step costs for arriving (or standing) FAST, in reward."""
        if self._calm_cost <= 0.0:
            return 0.0
        if not self._on_feet_only():
            return 0.0
        from . import asimov_getup_ref as _R
        if float(-self._projected_gravity()[2]) <= _R.STAND_UPRIGHT:
            return 0.0
        # THE ARRIVAL, NOT THE HOLD. The first version charged speed for as
        # long as the robot was up, which prices the CORRECTIONS that keep it
        # there — and measured, that stage went `stays` 13/24 -> 6/24. The
        # lunge is a property of how it gets to standing, so the charge ends
        # once it has been up for `CALM_WINDOW_S`; after that it may move as
        # much as it needs to.
        k = int(getattr(self, "_ep_k", 0))
        if self._stood_at is None:
            self._stood_at = k
        if k - self._stood_at > int(self.CALM_WINDOW_S * 50):
            return 0.0
        return self._calm_cost * max(0.0, self._trunk_speed() - self.CALM_FREE)

    def _reset_episode(self) -> None:
        """EVERY per-episode baseline, in ONE place.

        The `_fall_q` and `_handover_q` reset branches returned early having
        cleared only `_best_rise`, `_stood` and `_held` — so every run using
        `start_rung` in (`falling`, `pike`, `seat`, `handover`, `route`)
        carried `_best_trunk_z`, `_prev_trunk_z`, `_prev_tuck_gap`,
        `_prev_tuck2`, `_best_upright`, `_best_supine`, `_best_tuck`,
        `_best_head`, `_peak_rise`, `_held_best` AND the whole push clock
        across episode boundaries. That is the fault this file already
        records twice — "a rate carried across a reset is a rate between two
        different episodes", and `rotate` paying 0.00 at weight 40 — and it
        silently applied to the `catch`, `descend`, `hybrid` and `seat`
        recipes. It surfaced as a crash only when `calm_cost` was first
        enabled on such a run: `_calm_charge` reads `_stood_at`, which
        `_reset_push` is what creates, so the trainer raised on the first
        step of every episode and sat at 0 of 4,000,000 steps reporting
        `running: True`.
        """
        self._stood = False
        self._held = 0
        self._held_best = 0
        self._peak_rise = self._best_rise
        self._best_trunk_z = None
        self._prev_trunk_z = None
        self._prev_tuck_gap = None
        self._prev_tuck2 = None
        self._prev_upright = None
        self._prev_rise = None
        self._best_upright = None
        self._best_supine = None
        self._torso_bid = getattr(self, '_torso_bid', None)
        self._best_tuck = None
        self._reset_push()
        self._best_head = None

    def _reset_push(self) -> None:
        """Disarm the shove clock. PER EPISODE, beside the best-so-far
        baselines — `rotate` paid 0.00 at a weight of 40 for exactly one
        reason, which was a per-episode quantity initialised once in the
        constructor."""
        self._push_until = -1
        self._push_next = -1
        self._push_k = 0
        self._ep_k = 0
        self._stood_at = None
        # The SWING baselines, per episode and for the same reason every
        # best-so-far baseline is reset here: a rate carried across a reset
        # is a rate between two different episodes. `rotate` paid 0.00 at a
        # weight of 40 because one of these lived only in the constructor.
        self._prev_sh = None
        self._prev_swing_z = None
        self._prev_trunk_xy = None
        self._handoff_streak = 0
        self._handoff_prev_xy = None
        self._swing_lo = None
        self._swing_hi = None
        self._swing_best = 0.0
        self._pushes = 0
        self._push_vec = np.zeros(3)
        if self._push_bw > 0.0:
            self.data.xfrc_applied[self.trunk_body_id, 0:2] = 0.0

    #: **ANSWERED 2026-10-13, and the answer is no: one frame is NOT
    #: sufficient.** Freeze openhorizon's 5-frame ring to the present — same
    #: graph, same weights, the exact input their own spec says step 1
    #: receives — and their get-up goes **24/24 -> 0/24** prone and 23/24 ->
    #: 0/24 supine (`scripts/probe_history_ablation.py`). The history is
    #: load-bearing, not spare capacity.
    #:
    #: It lives on THIS class and not on `AsimovGetupRefEnv`, where it was
    #: added, because a stacked observation is a property of what the POLICY
    #: consumes and not of the tracking task. While it sat on the subclass, a
    #: 78*N policy could be TRAINED (the clip task is a subclass) and then
    #: could not be SCORED at all: the deliverable is a get-up in the plain
    #: `getup` env, that env emitted 78 floats, and the scorer died on
    #: `(78,) vs (390,)`. "Score at the horizon the deliverable is defined
    #: at" is impossible if the deployment env cannot speak the policy's own
    #: observation.
    #:
    #: What `history > 1` costs, precisely. It **breaks the drop-in ONNX
    #: contract** — 78*N inputs, so it cannot replace a 78-input policy in
    #: place and will not load in Menlo's viewer. It is NOT undeployable: the
    #: robot senses the same 78 floats and buffering N of them on-device is a
    #: ring buffer, which is exactly what openhorizon ship. So the honest
    #: finding for Menlo is "a get-up on this body needs N frames of your
    #: observation, and your hot-swap contract carries one", not "this robot
    #: cannot get up".
    #:
    #: Stacked FRAME-major (each frame's 78 in order, oldest first) where
    #: theirs is TERM-major (each term's N frames). Irrelevant to a policy
    #: trained here — a first layer can permute its own inputs — but their
    #: weights cannot be loaded into this layout directly. At N = 5 our 390
    #: floats are a strict SUPERSET of their 375 (ours carries the 3
    #: velocity-command slots per frame, which a get-up has no use for), so
    #: anything their policy computes, a policy on this observation can.
    HISTORY_ENV = "MICRODUCK_ASIMOV_GETUP_HISTORY"

    #: Class-level defaults so `_get_obs` is safe even if the BASE
    #: constructor reaches it before `_set_history` runs — an AttributeError
    #: there would be raised inside env construction, where this repo has
    #: twice mistaken a build failure for a policy that does nothing.
    _history = 1
    _obs_stack: list = []
    #: False until this episode has actually STEPPED. While it is False every
    #: `_get_obs` REBUILDS the ring instead of advancing it — see `_get_obs`.
    #:
    #: **Set by `step`. A caller that drives the physics ITSELF must set it,
    #: or every observation it collects will have N identical frames** — a
    #: history-less observation wearing a 78*N shape, with no error anywhere.
    #: That is not hypothetical: `scripts/distill_getup.py` hand-rolls
    #: `mj_step` for the teacher-driven iteration, and the first clone fitted
    #: this way reported action MSE **0.000359** on its own training set and
    #: **0.273526** on the identical states reached through `env.step` — a
    #: 760x gap that read exactly like covariate shift and was an input bug.
    _obs_live = False

    def _set_history(self, history: int | None) -> None:
        """Adopt an N-frame observation, widening `observation_space` with it.

        Called from `__init__` and by any subclass that takes `history` as a
        kwarg. At N == 1 it is a no-op and `_get_obs` is bit-identical to the
        parent's — the contract arm must not be perturbed by the existence of
        the control arm.
        """
        self._history = max(1, int(
            history if history is not None
            else os.environ.get(self.HISTORY_ENV, "1") or 1))
        if self._history > 1:
            import gymnasium as gym
            w = int(self.robot.obs_dim) * self._history
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, (w,), np.float32)
        self._obs_stack: list[np.ndarray] = []
        self._obs_live = False

    def _get_obs(self) -> np.ndarray:
        """The contract's 78 floats, or `history` of them stacked oldest-first.

        At `history > 1` this is STATEFUL, and `reset` calls it MORE THAN
        ONCE: the base constructor-side reset builds an observation, then this
        class's own reset repositions the robot (a prologue rung, a fall
        state, a pose cache draw) and builds another. Advancing on each of
        those left the ring holding four copies of a PRE-reposition frame and
        one real one — a 5-frame velocity transient, at the start of every
        episode, that no episode contained. Measured: after a reset the five
        frames were not equal when they must be.

        So the ring only advances once the episode has STEPPED (`_obs_live`).
        Before that every read REBUILDS it from the current frame, which is
        also what openhorizon's spec prescribes on a (re)start ("every history
        slot is filled with the first value") — zeros would be a different
        lie. That makes the method idempotent during reset and stateful only
        where it should be, which is the `rotate`-paid-0.000 lesson applied to
        an observation instead of a reward.
        """
        obs = super()._get_obs()
        if self._history <= 1:
            return obs
        if not self._obs_live or not self._obs_stack:
            self._obs_stack = [obs.copy() for _ in range(self._history)]
        else:
            self._obs_stack.pop(0)
            self._obs_stack.append(obs.copy())
        return np.concatenate(self._obs_stack).astype(np.float32)

    def rated_speeds(self):
        """Per-joint rated speed (rad/s) from Menlo's URDF, `JOINT_NAMES` order.

        Delegates to `robots/asimov_motor`, which is the one place this repo
        parses the vendor's ratings — "one definition of success, in the
        library, called by every scorer" applies to the robot's numbers too.
        """
        import numpy as _np

        from . import asimov_walk as _W
        lim = _motor.urdf_limits()
        return _np.array([lim[n][1] if n in lim else _np.nan
                          for n in _W.JOINT_NAMES], float)

    def _apply_push(self) -> None:
        """Shove the trunk, once the robot is up. Called before every step.

        Writes the HORIZONTAL row only and leaves `xfrc_applied[trunk, 2]`
        alone, because that is the assist's channel and clobbering it would
        silently switch the assist off mid-episode.
        """
        if self._push_bw <= 0.0:
            return
        from . import asimov_getup_ref as _R

        tb = self.trunk_body_id
        k = int(getattr(self, "_ep_steps", 0) or self._step_count())

        def _gap() -> int:
            if self._push_every > 0.0:
                j = self.PUSH_EVERY_JITTER
                return int(50 * self._push_every
                           * float(self._rng.uniform(1 - j, 1 + j)))
            lo, hi = self.PUSH_EVERY_S
            return int(50 * float(self._rng.uniform(lo, hi)))

        if k <= self._push_until:
            self.data.xfrc_applied[tb, 0:2] = self._push_vec[:2]
            return
        self.data.xfrc_applied[tb, 0:2] = 0.0
        if not _R.is_standing(self):
            # HOLD THE CLOCK while it is down, so the gap is time spent
            # STANDING rather than wall time. Without this the next shove is
            # already overdue by the time a 3.4 s rise finishes and the robot
            # is hit the instant it is up.
            if self._push_every > 0.0 and self._push_next >= 0:
                self._push_next = k + _gap()
            return
        if self._push_next < 0:                    # first stand: arm the clock
            self._push_next = k + _gap()
            return
        if k < self._push_next:
            return
        mass = float(self.model.body_subtreemass[tb])
        mag = float(self._rng.uniform(0.5, 1.5)) * self._push_bw * mass * 9.81
        yaw = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self._push_vec = np.array([mag * np.cos(yaw), mag * np.sin(yaw), 0.0])
        self.data.xfrc_applied[tb, 0:2] = self._push_vec[:2]
        self._push_until = k + int(50 * self.PUSH_HOLD_S)
        self._push_next = self._push_until + _gap()
        self._pushes += 1

    def _step_count(self) -> int:
        """Steps this episode, for bodies that do not keep `_ep_steps`."""
        n = getattr(self, "_push_k", 0) + 1
        self._push_k = n
        return n

    def _on_feet_only(self) -> bool:
        """Is the robot touching the floor with its FEET and nothing else?

        The cheap half of `standing_report`'s test, without the part naming
        — this runs every step of every env and `part_of` walks the chain.
        """
        m, d = self.model, self.data
        feet = {int(m.geom_bodyid[self.foot_geoms[s]])
                for s in ("left", "right")}
        seen = False
        for c in range(int(d.ncon)):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if m.geom_bodyid[g1] == 0 or m.geom_bodyid[g2] == 0:
                b = int(m.geom_bodyid[g2] if m.geom_bodyid[g1] == 0
                        else m.geom_bodyid[g1])
                if b not in feet:
                    return False
                seen = True
        return seen

    def _balance_charge(self) -> float:
        """What this step costs for standing off its feet, in reward.

        Zero unless the robot is UP — so the rise is never charged — and
        zero while the CoM is inside the polygon, so this is a one-sided
        body term like `_span_charge`, not a target to be optimised toward.
        """
        if self._balance_cost <= 0.0:
            return 0.0
        from . import asimov_getup_ref as _R
        if float(-self._projected_gravity()[2]) <= _R.STAND_UPRIGHT:
            return 0.0
        if not self._on_feet_only():
            return 0.0
        margin = _R.support_margin(self)
        if margin is None or margin >= 0.0:
            return 0.0
        return self._balance_cost * min(-margin, self.BALANCE_CAP)

    def _on_floor_contacts(self) -> int:
        """How many contacts this step are with the floor (world body 0)."""
        ids = getattr(self, "_floor_geom_ids", None)
        if ids is None:
            ids = frozenset(g for g in range(self.model.ngeom)
                            if self.model.geom_bodyid[g] == 0)
            self._floor_geom_ids = ids
        return sum(1 for i in range(self.data.ncon)
                   if self.data.contact[i].geom1 in ids
                   or self.data.contact[i].geom2 in ids)

    def _span_allowed(self) -> float:
        """The span cut AT THIS HEIGHT — wide only inside the launch window.

        Outside it the robot is either supported on its arms or standing on
        its feet, and in both the narrow stance is what the authored route
        uses (every waypoint sits at 0.215 m). Inside it the robot is crossing
        136 mm of nothing, and a wide base is the only mechanism measured to
        work. This bounds the sprawl to that moment instead of banning it.
        """
        if self._launch_span <= 0.0:
            return self._max_span
        from . import asimov_getup_ref as _R
        z = float(self.data.xpos[self.trunk_body_id][2])
        lo = _R.BY_NAME["pike"].settled_z - 0.02
        hi = _R.BY_NAME["fold"].settled_z + 0.02
        return self._launch_span if lo <= z <= hi else self._max_span

    def _foot_span(self) -> float:
        """Distance between the two feet, metres — lateral AND fore-aft.

        Not the lateral width alone: the sprawl this exists to forbid opens
        both, and a robot in a front-to-back splits is in the splits.
        """
        import numpy as _np
        ids = getattr(self, "_span_body_ids", None)
        if ids is None:
            # resolved from the SPEC's own foot geoms, so a body that renames
            # its ankle link cannot silently measure the wrong pair
            ids = tuple(int(self.model.geom_bodyid[self.foot_geoms[side]])
                        for side in ("left", "right"))
            self._span_body_ids = ids
        a = self.data.xpos[ids[0]]
        b = self.data.xpos[ids[1]]
        return float(_np.hypot(a[0] - b[0], a[1] - b[1]))

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
        if self.in_prologue():
            # THE SERVOS DRIVE, not the policy. Substituting here rather than
            # in `reset` is the whole point: this is stepped time, so the lab
            # shows the push-up instead of jumping to its last frame.
            action = self._prologue_plan[self._prologue_i]
            self._prologue_i += 1
        self._ep_k = getattr(self, "_ep_k", 0) + 1
        # From here the observation ring ADVANCES rather than rebuilding —
        # see `_get_obs`. Set before `super().step`, which builds the obs.
        self._obs_live = True
        self._apply_push()
        self._apply_attitude_assist()
        obs, reward, terminated, truncated, info = super().step(action)
        info["prologue"] = self.in_prologue()
        info["pushes"] = self._pushes
        if self._max_span > 0.0 and not terminated and self._foot_span() > self._span_allowed():
            terminated = True
            self._span_cuts += 1
        # THE EPISODE ENDS AT THE HANDOFF, when trained for it. The bonus
        # replaces the income the rest of the episode would have paid, so
        # arriving catchable is worth standing perfectly and arriving fast
        # is worth a fraction of it.
        if self._handoff_pay > 0.0 and not terminated and self.handoff_ready():
            left = max(0, int(self.max_steps) - int(self._ep_k))
            reward += self._handoff_pay * left * self._handoff_score()
            info["handoff_score"] = self._handoff_score()
            info["handoff_step"] = int(self._ep_k)
            terminated = True
        if self._stood and not terminated and self._fell_cut:
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
    #: Env var for `cut_back` — how far the trunk may sink BELOW the height
    #: the episode started at before the episode ends, metres. Defaults to
    #: `CUT_BACK_Z`, so omitting it reproduces every run in the repo.
    #:
    #: Why it had to become a knob. At 0.15 m this constant is what makes
    #: every rung below `fold` unteachable, and the cause is not the reward.
    #: `rock_back` (0.377 m) and `pike` (0.263 m) are not equilibria, so they
    #: sink on step 1 and the cut fires in **7-9 steps (0.14-0.18 s)** every
    #: episode, forever — measured 2026-10-19 under zero action with
    #: `fell_cut=False`. Two chains spent 2.37 M and 2.96 M steps scoring
    #: **0/24** on those rungs. Seven reward designs were aimed at a reward
    #: that was never the problem.
    #:
    #: And it selects FOR the sprawl. The cut is survivable only by going up
    #: IMMEDIATELY, which on this body the narrow route cannot do — no back
    #: pitch joint, CoM 560 mm ahead of the feet at the pike — while the
    #: 1.029 m lunge can. So the one motion the constant permits from `pike`
    #: is the one the whole narrow track exists to avoid: under zero action
    #: the span at `rock_back` is a narrow 0.214 m, and the policy trained
    #: under the cut reaches 0.530 there and 0.731 at `pike`.
    #:
    #: Raising it asks the question nobody could ask: is there a narrow
    #: pike->fold route that DIPS first and then rises? The unholdable band
    #: (no supported hold between pelvis 0.30 and 0.511 m) forbids a static
    #: rung there, but says nothing about a dynamic dip-and-rise. A generous
    #: value is a SCAFFOLD and has to be faded back to 0.15 before any claim,
    #: exactly like `getup_assist` — a policy that only stands with a 0.5 m
    #: back-slide allowance has not learned the deliverable.
    CUT_BACK_ENV = "MICRODUCK_ASIMOV_GETUP_CUT_BACK"
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

    #: Env var for `arrival` — the MAXIMUM arrival velocity, m/s, injected at
    #: reset on top of the placed waypoint. 0.0 (OFF) reproduces the placed,
    #: motionless start this env has always used.
    #:
    #: Why this knob exists. There is no static rung between `pike` (0.300 m)
    #: and `fold` (0.511 m) — searched with the hold inside the objective the
    #: best candidate collapses — so that leg has to be ONE dynamic transition
    #: that arrives at `fold` slow enough to be caught. That makes the
    #: catcher's ARRIVAL BASIN the spec, and this env could not train it:
    #: `reset` places the waypoint from `_start_cache` and then sets
    #: `qvel[:] = 0.0`, so every episode this task has ever run began at rest.
    #: The base env's `start_rung="falling"` machinery cannot supply the
    #: states either — the ref reset overwrites both the pose and the
    #: velocity, which is why a `fall_depth` sweep over 0.1/0.25/0.5/1.0 came
    #: back BYTE-IDENTICAL (24/24, peak 0.623, span 0.269 at every depth) on
    #: 2026-10-19. A dead hook, not a null result.
    #:
    #: Measured on `teach-asimov_getup_ref-c2bae8-s1`, which stands 48/48 from
    #: a placed `fold`: it holds 24/24 at a 0.00 m/s arrival, **6/24 at 0.20**
    #: and 0/24 at 0.40 and above. So the basin is "arrive at rest", and a
    #: launch out of the pike cannot hit that. Training ON the arrivals is the
    #: direct attack: the deliverable is the largest arrival still caught.
    #:
    #: The injection is COUPLED the way a pike->fold rise actually arrives —
    #: the pelvis rising while the trunk is still rotating up — because
    #: vertical speed with no pitch rate is not an arrival, it is a hop. Over
    #: that move the pelvis gains 0.211 m while the trunk rotates about a
    #: radian, hence `ARRIVAL_PITCH` rad/s per m/s. Drawn uniformly from
    #: [0, arrival] per episode rather than fixed, so the easy arrivals keep
    #: being practised — `a-curriculum-must-rehearse`: training each rung only
    #: on its newest, hardest start is what made this robot forget.
    ARRIVAL_ENV = "MICRODUCK_ASIMOV_GETUP_ARRIVAL"
    #: Pitch rate per m/s of lift, measured over the pike->fold rise.
    ARRIVAL_PITCH = 4.7

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

    def __init__(self, *args, start_phase: str | None = None,
                 track_tol: float | None = None,
                 arrival: float | None = None,
                 cut_back: float | None = None,
                 ref_speed: float | None = None,
                 history: int | None = None,
                 force_lag: bool | None = None,
                 free: bool | None = None, **kwargs):
        from . import asimov_getup_ref as REF

        super().__init__(*args, **kwargs)
        if history is not None:
            self._set_history(history)
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
        self._arrival = float(arrival if arrival is not None
                              else os.environ.get(self.ARRIVAL_ENV, "")
                              or 0.0)
        # None when nothing was set, and resolved against `CUT_BACK_Z` at USE
        # time rather than here. Baking the default into the instance made
        # `env.CUT_BACK_Z = 99.0` a no-op — the idiom
        # `test_the_planted_regression_for_the_fall_back_cut` uses to disarm
        # the cut, and the thing that proves the cut test passes for the right
        # reason. Shadowing a class constant with an instance attribute kills
        # every existing override of it.
        _cb = (cut_back if cut_back is not None
               else (os.environ.get(self.CUT_BACK_ENV, "") or None))
        self._cut_back = None if _cb is None else float(_cb)
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
        # THE ARRIVAL. Injected after the placement and the qvel zeroing, so
        # the pose is the cache's own waypoint and only the root twist differs
        # — which is exactly what separates "placed at fold" from "arriving at
        # fold". See `ARRIVAL_ENV` for why this is the spec for the one leg
        # that cannot be laddered.
        self._arrival_v = 0.0
        if self._arrival > 0.0:
            self._arrival_v = float(self._rng.uniform(0.0, self._arrival))
            self.data.qvel[2] += self._arrival_v
            self.data.qvel[4] += self._arrival_v * self.ARRIVAL_PITCH
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
            "span_penalty": -self._span_charge(),
            "flight_penalty": -self._flight_charge(),
            "fold_arrival": self._fold_arrival_gain(),
            "arm_unload_penalty": -self._arm_unload_charge(),
            # PUSHING UP THROUGH AN ARM — load carried x upward trunk speed.
            # Present in the base getup terms and absent here, so `arm_push`
            # READ correctly on this task and summed nowhere: a run at 150
            # and one at the 4.0 default produced byte-identical results
            # (4/12 rises, span 0.742/0.408/0.517 to three decimals), which is
            # the dead-hook signature. Third knob found this way on this body
            # after `arm_effort` and the un-recorded ones.
            #
            # It matters for the narrow route specifically: the stride exists
            # to buy support across a band with nothing to lean on, and the
            # arms are the other way to buy it. Default 4.0 is unchanged, so
            # no earlier run moves.
            "arm_push": (self._w_arm_push * min(self._arm_load(), 1.0)
                         * max(self._trunk_up_speed(), 0.0)),
            "arm_work": self._w_arm_work * self._arm_work_gain(),
            "tuck_work": self._w_tuck_work * self._tuck_work_gain(),
            "tuck2": self._w_tuck2 * self._tuck2_gain(),
            "rotate_work": self._w_rotate_work * self._rotate_work_gain(),
            "rise_work": self._w_rise_work * self._rise_work_gain(),
            # GET THE FEET UNDER THE MASS, and ROTATE UPRIGHT. Both off by
            # default (0.0), and both were MISSING from this dict while being
            # read, recorded onto the run and reported as set — the fourth
            # time in this file (after `span_penalty`, `arm_effort` and
            # `arm_push`) that a knob was live on the base task and summed
            # nowhere here. `tuck` is the one that matters for the narrow
            # route: on 2026-10-19 `c130f7` climbed out of `pike` to 0.364 m
            # and stopped, with the budget showing `arm_push` 561 and the
            # blocker being the centre of mass still ahead of the feet —
            # which is exactly what `tuck` pays for, and it could not fire.
            "tuck": self._tuck_pay * self._tuck_gain(),
            "rotate": self._rotate_pay * self._upright_gain(),
            "balance_penalty": -self._balance_charge(),
            "home": self._home_gain(),
            "calm_penalty": -self._calm_charge(),
            "swing": self._swing_gain(),
            "limit_penalty": -self._limit_charge(),
            "action_rate_penalty": -self.W_ACTION_RATE * rate,
            **self._regulariser_terms(up),
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
        if (not terminated and not self._free and not self.in_prologue()
                and getattr(self, "_track_err", 0.0) > self._track_tol):
            # NOT WHILE THE SERVOS DRIVE. During the scripted push-up the
            # policy's action is discarded, so ending the episode for drifting
            # off the reference would charge it for a motion it did not make —
            # and would kill most episodes in their first second.
            terminated = True
        if not terminated:
            cb = getattr(self, "_cut_back", None)
            floor = (getattr(self, "_anchor_z", 0.0)
                     - (self.CUT_BACK_Z if cb is None else cb))
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
    its own DC-motor model (mjlab's `dc_motor_clip`) and bypasses MuJoCo's
    actuators. The joint path transfers, the torques do not -- and NEITHER
    actuator dominates the other, which an earlier version of this docstring
    got backwards in the direction that flattered us.

    Both compute the same PD output from the same gains: our `GAIN_GROUPS` and
    their `pd_gains` carry identical kp, kd, armature and friction on all 12
    groups, and our per-group `effort` IS Menlo's URDF `<limit effort=>`. The
    one difference is what clamps it. Ours clamps at that effort with NO speed
    dependence. Theirs clamps inside an envelope that starts at `peak_torque`
    (3x effort; 36-145.4 N*m) at zero speed and falls linearly to zero at the
    URDF's rated speed. So theirs is the STRONGER actuator near rest -- 120
    N*m against our 45 at the hip -- and ours is the only one with any torque
    left above the rating. Reading it the other way is what made a crude
    overspeed brake look like a faithful limit and produced a "0/24 under
    limits" that over-penalised our own policy: it removed the high-speed
    torque without ever granting the low-speed peak. `robots/asimov_motor`
    implements the real envelope, and `actuator_force="dc"` selects it.

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
        """Start where a SUCCESSFUL demonstration started, and track it.

        THIS TASK CHOOSES ITS OWN START, so `start_rung` cannot apply — the
        line below overwrites whatever the base class placed. It used to do
        that in SILENCE: three chains were launched with
        `start_rung="pike"`, recorded it in `run.json`, trained entirely from
        clip frames at a median trunk height of **0.153 m**, and were then
        evaluated in an env built the same way, so "rises 16/16 from the
        settled pike" was measuring clip starts at every step. `getup` is the
        task that honours the rung (pike -> 0.299 m, falling at depth 0 ->
        0.523 m, both measured). The sixth knob on this body to train while
        something else was recorded, and the first that a RECORDING test
        could not catch — recorded is not applied.
        """
        if self._start_rung and self._start_rung != "floor":
            raise ValueError(
                f"getup_clip starts on a demonstration frame, so "
                f"start_rung={self._start_rung!r} would be silently "
                f"discarded — use --task getup, which honours it")
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
    #: Tightened 0.10 -> 0.05 on 2026-10-07. At 0.10 a 2 cm sag cost `track`
    #: only 18 %, so the reward was nearly FLAT across the exact drift that
    #: loses the episode: measured on teach-asimov_getup_clip-0e31e5, `track`
    #: paid 6.75 -> 6.43 -> 6.49 -> 5.55 while the trunk fell 0.613 -> 0.597 m,
    #: and `up` contributes nothing at all until the robot is already below the
    #: standing threshold — a cliff, not a slope. 59 of 72 episodes stand and
    #: then go down, and the servos are not the limit: during the sink the
    #: commanded knee climbs 0.383 -> 0.645 rad with max torque 12.8 Nm against
    #: 45 rated. The policy COMMANDS the sag, and nothing charged it.
    ROOT_Z_STD = 0.05
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
        # ONE-SIDED, and that is AGENTS.md's rule rather than a preference:
        # "Make the BODY terms one-sided (max(0, target - actual)). Two-sided,
        # the reward scored a pose past the target WORSE than one short of it
        # — it punished overshooting toward the goal, which on a rise cannot be
        # wrong." This was `abs()`, so a robot AHEAD of the demonstration was
        # charged exactly as much as one sagging below it, and the measured
        # failure is entirely on the low side: the policy stands at 0.613 m
        # and then drifts DOWN to 0.452 while the reference holds 0.619.
        dz = max(0.0, float(ref[2]) - float(self.data.qpos[2]))
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
            "span_penalty": -self._span_charge(),
            "flight_penalty": -self._flight_charge(),
            "fold_arrival": self._fold_arrival_gain(),
            "arm_unload_penalty": -self._arm_unload_charge(),
            # PUSHING UP THROUGH AN ARM — load carried x upward trunk speed.
            # Present in the base getup terms and absent here, so `arm_push`
            # READ correctly on this task and summed nowhere: a run at 150
            # and one at the 4.0 default produced byte-identical results
            # (4/12 rises, span 0.742/0.408/0.517 to three decimals), which is
            # the dead-hook signature. Third knob found this way on this body
            # after `arm_effort` and the un-recorded ones.
            #
            # It matters for the narrow route specifically: the stride exists
            # to buy support across a band with nothing to lean on, and the
            # arms are the other way to buy it. Default 4.0 is unchanged, so
            # no earlier run moves.
            "arm_push": (self._w_arm_push * min(self._arm_load(), 1.0)
                         * max(self._trunk_up_speed(), 0.0)),
            "arm_work": self._w_arm_work * self._arm_work_gain(),
            "tuck_work": self._w_tuck_work * self._tuck_work_gain(),
            "tuck2": self._w_tuck2 * self._tuck2_gain(),
            "rotate_work": self._w_rotate_work * self._rotate_work_gain(),
            "rise_work": self._w_rise_work * self._rise_work_gain(),
            # GET THE FEET UNDER THE MASS, and ROTATE UPRIGHT. Both off by
            # default (0.0), and both were MISSING from this dict while being
            # read, recorded onto the run and reported as set — the fourth
            # time in this file (after `span_penalty`, `arm_effort` and
            # `arm_push`) that a knob was live on the base task and summed
            # nowhere here. `tuck` is the one that matters for the narrow
            # route: on 2026-10-19 `c130f7` climbed out of `pike` to 0.364 m
            # and stopped, with the budget showing `arm_push` 561 and the
            # blocker being the centre of mass still ahead of the feet —
            # which is exactly what `tuck` pays for, and it could not fire.
            "tuck": self._tuck_pay * self._tuck_gain(),
            "rotate": self._rotate_pay * self._upright_gain(),
            "balance_penalty": -self._balance_charge(),
            "home": self._home_gain(),
            "calm_penalty": -self._calm_charge(),
            "swing": self._swing_gain(),
            "limit_penalty": -self._limit_charge(),
            "action_rate_penalty": -self.W_ACTION_RATE * rate
                                   * self._penalty_scale(),
            **self._regulariser_terms(up),
        }
        return float(sum(terms.values())), terms
