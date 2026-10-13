"""The authored get-up reference — `robots/asimov_getup_ref`, roadmap phase 3c.

Four get-up runs failed on a reward built from `height`, `u_z` and `feet`. The
reference exists because those three are **non-monotone along a correct get-up
on this body**, and the tests below pin the measurements that say so, because
each one is a thing a future change could silently undo:

* `u_z` goes +0.01 (prone) -> -0.37 (pike) -> +1.00 (stand). A reward paying
  uprightness per step pays the robot to skip the only phase that lifts it.
* `height` is non-monotone too: `plant_forearms` DROPS the pelvis, because
  loading the elbows is the move.
* The ONE unverified leg is `rock_back` -> `fold`. Everything before it the
  position servos execute with no policy at all, so the learning problem is a
  single dynamic transition — if that stops being true, a test should say so.

The heavy checks (place, command, hold 6 s) are marked `slow`: there are six
equilibria at 6 s each plus an 8 s chain. `scripts/probe_asimov_getup_ref.py
--verify` is the same measurement as a script.

Skipped, not failed, when Asimov's model is not downloaded.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from microduck_local.robots import asimov
from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import asimov_walk as W

pytestmark = pytest.mark.skipif(
    not asimov.ready(),
    reason="Asimov's model is not fetched (uv run fetch-robot asimov1)")


# ------------------------------------------------------------- the helper

def test_joints_with_no_arguments_is_the_default_pose():
    assert np.allclose(R.joints(), W.DEFAULT_POSE)


def test_a_mirrored_kwarg_is_the_left_sides_value():
    """Passing a joint its own default must be a NO-OP.

    The trap: the sign DEFAULT_POSE uses is not uniform — `left_knee` is +0.45
    against right -0.45, but `left_hip_pitch` is -0.15 against right +0.15, and
    `left_shoulder_pitch` is -0.25 while `left_elbow` is +0.40. A helper
    applying one sign to all of them swung both arms opposite to the default
    while the legs obeyed, and a crouch that holds read as collapsing to
    0.139 m. Without this property a pose draft cannot be read.
    """
    assert np.allclose(R.joints(hip_pitch=-0.15, knee=0.45, ankle_pitch=-0.30,
                                shoulder_pitch=-0.25, elbow=0.40),
                       W.DEFAULT_POSE)


def test_a_literal_joint_name_sets_one_side_only():
    q = R.joints(left_knee_joint=1.2)
    i = list(W.JOINT_NAMES).index("left_knee_joint")
    j = list(W.JOINT_NAMES).index("right_knee_joint")
    assert q[i] == pytest.approx(1.2)
    assert q[j] == pytest.approx(W.DEFAULT_POSE[j])


def test_an_unknown_joint_is_a_KeyError_not_a_silent_no_op():
    with pytest.raises(KeyError):
        R.joints(wrist_pitch=0.3)


# -------------------------------------------------- the shape of the reference

def test_every_waypoint_is_inside_the_joint_limits():
    """An authored pose outside a limit is silently clamped, and the reference
    then asks for something the robot is not doing."""
    spec = W.walk_spec()
    model = spec.compile() if hasattr(spec, "compile") else spec
    lo = np.array([model.joint(n).range[0] for n in W.JOINT_NAMES])
    hi = np.array([model.joint(n).range[1] for n in W.JOINT_NAMES])
    for w in R.WAYPOINTS:
        q = w.joints
        over = [(n, float(v), float(a), float(b))
                for n, v, a, b in zip(W.JOINT_NAMES, q, lo, hi)
                if v < a - 1e-9 or v > b + 1e-9]
        assert not over, f"{w.name} is outside its limits: {over}"


def test_the_reference_is_continuous_at_every_waypoint():
    """`pose_at` is differentiated by an imitation reward; a corner in it reads
    as an acceleration the policy is charged for not producing."""
    total = R.duration()
    ts = np.arange(0.0, total, 0.02)
    q = np.array([R.pose_at(float(t))[0] for t in ts])
    step = np.abs(np.diff(q, axis=0)).max()
    # 0.14 rad per control step is 7 rad/s, inside what these servos follow.
    # The bound was 0.12 while the legs were 0.6-1.0 s; the tail is now a
    # MEASURED 0.25/0.30 s, because slower than that the robot settles at each
    # step instead of carrying through and sits down. Faster legs mean bigger
    # per-step commands, and the honest limit is what a servo tracks.
    assert step < 0.14, f"a {step:.3f} rad jump between control steps"


def test_the_reference_starts_prone_and_ends_standing():
    q0, p0, _ = R.pose_at(0.0)
    q1, p1, _ = R.pose_at(R.duration())
    assert R.WAYPOINTS[0].name == "prone"
    assert p0 == pytest.approx(1.57, abs=0.01)      # face DOWN
    assert p1 == pytest.approx(0.0, abs=0.01)
    assert np.allclose(q1, W.DEFAULT_POSE)


def test_uprightness_is_non_monotone_along_the_reference():
    """THE reason the first four runs failed, as an assertion.

    If this ever becomes monotone, either the route changed or the measurements
    did — and a reward paying `u_z` per step would stop being wrong, which is a
    big enough change to be told about.
    """
    u = [w.settled_u for w in R.WAYPOINTS]
    assert min(u) < -0.3, f"nothing in the reference turns face down: {u}"
    assert u[-1] > 0.95
    assert any(u[i + 1] < u[i] - 0.05 for i in range(len(u) - 1)), u


def test_height_rises_monotonically_through_the_table():
    """CORRECTED. This asserted the opposite until 2026-10-04, and it passed for
    the wrong reason: `rock_back` was an INVENTED waypoint recording pelvis
    0.170 m — below the `pike` it follows — because its numbers came from where
    a training chain happened to collapse. Replacing it with a state recorded
    off the real trajectory (0.377 m) made the table monotone, which is what a
    reference for a rise should be. A dip here now means a waypoint is being
    taken from a collapse again.

    The height non-monotonicity that IS real lives in the TRAJECTORY, not the
    table — `plant_forearms` drops the pelvis from the prone placement as the
    elbows come under the body — and
    `test_the_servos_alone_follow_the_reference_up_to_the_rock_back` measures it.
    """
    z = [w.settled_z for w in R.WAYPOINTS]
    bad = [(R.WAYPOINTS[i].name, z[i], z[i + 1])
           for i in range(len(z) - 1) if z[i + 1] < z[i] - 1e-9]
    assert not bad, f"the reference descends at {bad}"


def test_exactly_one_leg_of_the_reference_is_unverified():
    """The phase-3c result: the learning problem is ONE dynamic transition."""
    names = [w.name for w in R.WAYPOINTS]
    assert R.VERIFIED_THROUGH in names
    cut = names.index(R.VERIFIED_THROUGH)
    assert all(w.verified for w in R.WAYPOINTS[:cut + 1]), \
        "a leg before VERIFIED_THROUGH claims to be unverified"
    assert not any(w.verified for w in R.WAYPOINTS[cut + 1:]), \
        "a leg after VERIFIED_THROUGH claims to be verified"


def test_no_waypoint_exceeds_the_torque_envelope():
    """`fits in within the limits of the hardware` — the standing requirement
    on everything trained for this robot. The elbows are the bottleneck."""
    worst = max(R.WAYPOINTS, key=lambda w: w.torque_frac)
    assert worst.torque_frac < 0.80, (
        f"{worst.name} holds at {worst.torque_frac:.0%} of rated torque, "
        "leaving nothing to move with")


# --------------------------------------------------------------- the physics

def _env():
    from microduck_local.robots.asimov_env import AsimovGetupEnv
    return AsimovGetupEnv(domain_rand=False, obs_noise=False,
                          getup_blend=0.0, getup_assist=0.0)


@pytest.mark.slow
@pytest.mark.parametrize("name", [w.name for w in R.WAYPOINTS if w.equilibrium])
def test_each_equilibrium_holds_for_six_seconds(name):
    """A keyframe is only a PHASE if the robot stays in it — the check the
    retired blend ladder would have failed in 30 seconds.

    6 s, not 2: a knee-1.00 crouch sits at 0.511 m for two full seconds and
    collapses at three, so a short window reports a slow collapse as a hold.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from probe_asimov_getup_ref import hold

    env = _env()
    env.reset(seed=0)
    w = R.BY_NAME[name]
    z, u, _, peak = hold(env, w, secs=6.0)
    assert z == pytest.approx(w.settled_z, abs=0.02), \
        f"{name}: pelvis {z:.3f}, recorded {w.settled_z:.3f}"
    assert u == pytest.approx(w.settled_u, abs=0.06), \
        f"{name}: u_z {u:+.3f}, recorded {w.settled_u:+.3f}"
    assert peak < 0.80, f"{name} needs {peak:.0%} of rated torque to be held"


@pytest.mark.slow
def test_the_servos_alone_follow_the_reference_to_the_verified_boundary():
    """The verified half, and its boundary — both directions of the claim.

    This is the planted-regression shape: it is not enough that the chain gets
    far, the leg AFTER `VERIFIED_THROUGH` must still fail. If it starts
    succeeding, the reference got easier and `VERIFIED_THROUGH` is stale; if it
    starts failing earlier, something in the model or the gains moved.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from probe_asimov_getup_ref import chain

    env = _env()
    env.reset(seed=0)
    rows = chain(env)
    on = {name: abs(z - want) < 0.04 for name, z, _u, _c, want in rows}
    cut = [w.name for w in R.WAYPOINTS].index(R.VERIFIED_THROUGH)
    for w in R.WAYPOINTS[1:cut + 1]:
        assert on[w.name], f"the servos no longer reach {w.name} unaided"
    after = R.WAYPOINTS[cut + 1]
    assert not on[after.name], (
        f"the servos now reach {after.name} unaided — the reference got "
        "easier, so VERIFIED_THROUGH is stale and the whole phase-3c result "
        "(one learned transition) needs restating")


# -------------------------------------------- the tracking env (roadmap 3d)

def _ref_env(**kw):
    from microduck_local.robots.asimov_env import AsimovGetupRefEnv
    kw.setdefault("domain_rand", False)
    kw.setdefault("obs_noise", False)
    kw.setdefault("start_phase", "pike")
    return AsimovGetupRefEnv(**kw)


def _run(env, action=None, cap=400):
    env.reset(seed=0)
    a = np.zeros(len(W.JOINT_NAMES), dtype=np.float32) if action is None else action
    info: dict = {}
    for n in range(1, cap + 1):
        *_, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            return n, info
    return cap, info


def test_the_observation_is_untouched_by_the_phase():
    """The whole point of this task is a get-up Menlo's deployment can run.

    The phase, the reference pose and the target contact set are privileged
    simulator state used to SCORE the policy, never to inform it — so the 78
    floats stay their layout and `obs[6:9]`, the command slots, stay zero.
    Putting a phase in those slots is the fallback and it costs a deployment
    change; this test is what stops it happening by accident.
    """
    env = _ref_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (78,)
    assert np.allclose(obs[6:9], 0.0)
    for _ in range(40):
        obs, *_ = env.step(np.zeros(23, dtype=np.float32))
    assert np.allclose(obs[6:9], 0.0), "something published a phase into the command"


def test_a_zero_action_cannot_walk_the_phase_clock_to_standing():
    """The exploit the first version of this env shipped with.

    A zero action commands `DEFAULT_POSE`, which IS the stand pose. With the
    clock advancing on JOINT error alone it ran from `pike` all the way to
    `stand` with `reached_stand` true, lying on the floor at pelvis 0.10 m the
    whole time — the joints matched and the robot did not. Same family as the
    sit and the shoulder stand that beat the first two get-up rewards: a proxy
    that failure also satisfies.
    """
    _, info = _run(_ref_env(start_phase="pike"))
    assert info["reached_stand"] is False, (
        "a zero action reached `stand` — the phase clock is advancing on "
        "joint error without requiring the body to be there")
    # `pike` is its own start: from here a constant DEFAULT_POSE command
    # achieves nothing at all, which is the measured baseline.
    assert info["phase_name"] == "pike"


def test_the_planted_regression_for_that_exploit():
    """Widen the body tolerances until they cannot bind, and the exploit must
    come back — otherwise the test above is not testing them."""
    env = _ref_env(start_phase="pike")
    env.REACH_Z_TOL = env.REACH_U_TOL = 9.0       # can never fail
    # The fall-back cut has to come off too, or it ends the episode at ~22
    # steps before the clock can walk anywhere and masks the plant.
    env.CUT_BACK_Z = 99.0
    _, info = _run(env)
    assert info["reached_stand"] is True, (
        "widening the body tolerances did not restore the exploit, so "
        "test_a_zero_action_cannot_walk_the_phase_clock_to_standing is "
        "passing for some other reason")


def test_the_fold_rung_is_free_and_the_pike_rung_is_not():
    """Which rung the curriculum must start at, as an assertion.

    From `fold` the robot is already on its feet, so a constant DEFAULT_POSE
    command stands it up — zero action REACHES `stand`, though it holds it only
    0.56 s before falling. From `pike` it achieves nothing at all. That is the
    same boundary the open-loop servo chain found, measured a second way, and it
    is why the first curriculum stage starts at `pike`.
    """
    _, from_fold = _run(_ref_env(start_phase="fold"))
    assert from_fold["reached_stand"] is True
    assert from_fold["held_s"] < 2.0, "zero action now HOLDS the stand from fold"
    _, from_pike = _run(_ref_env(start_phase="pike"))
    assert from_pike["reached_stand"] is False
    assert from_pike["phase_name"] == "pike"


def test_the_last_waypoint_is_reachable_at_all():
    """It was not, and every number in the first chain's report was wrong for it.

    Achieving a waypoint and ADVANCING the target were one branch, gated on
    `_phase < len - 1`, so when the target WAS `stand` the gate was false and
    `stand` could never be recorded. All four stages reported `reached_stand`
    0/16 and it was structurally impossible — from `crouch` a zero action sits
    at pelvis 0.608 m, standing by every number the table records, and scored
    nothing.
    """
    _, info = _run(_ref_env(start_phase="crouch"))
    assert info["reached_stand"] is True
    assert info["phase_name"] == "stand"
    assert info["held_s"] > 5.0, info["held_s"]


def test_every_cut_leaves_the_same_episode_record():
    """A cut added by this subclass fires after the base class has decided, so
    an episode it ended used to carry no `held_s` and the reader raised
    KeyError instead of reporting a number."""
    for start, tol in (("pike", 0.6), ("fold", 0.6), ("prone", 0.9)):
        _, info = _run(_ref_env(start_phase=start, track_tol=tol))
        for k in ("held_s", "peak_rise", "stood", "phase", "phase_name",
                  "targeting", "track_err", "reached_stand"):
            assert k in info, f"{start}: the record is missing {k!r}"


def test_the_prone_rung_needs_a_looser_cut_than_the_default():
    """Check a knob's reachable set before believing in it.

    A cache pose sits ~0.78 rad of mean joint error from the first waypoint, so
    at the default 0.60 cut every `prone` episode dies on its FIRST step for
    being a different fallen robot. The curriculum sets 0.90 for that rung and
    this is the measurement behind it.
    """
    tight, _ = _run(_ref_env(start_phase="prone", track_tol=0.60))
    loose, _ = _run(_ref_env(start_phase="prone", track_tol=0.90))
    assert tight <= 2, f"the tight cut no longer bites on prone ({tight} steps)"
    assert loose > 100, f"the loose cut still bites on prone ({loose} steps)"


def test_an_unknown_start_phase_fails_loudly():
    with pytest.raises(ValueError, match="not a waypoint"):
        _ref_env(start_phase="halfway")


def test_the_curriculum_starts_at_the_one_unverified_leg():
    """The behaviour's first rung must be the leg the servos cannot do."""
    import microduck_local.behaviors  # noqa: F401 — registers the recipes
    from microduck_local.behaviors.core import for_robot

    ref = next(b for b in for_robot("asimov1") if b.id == "asimov_getup_ref")
    names = [w.name for w in R.WAYPOINTS]
    cut = names.index(R.VERIFIED_THROUGH)

    def newest(stage):
        """A rung is a SET now — the newest start is the last one listed."""
        return stage.env["MICRODUCK_ASIMOV_GETUP_START"].split(",")[-1].strip()

    first = names.index(newest(ref.curriculum[0]))
    assert first >= cut, (
        "the curriculum starts before the leg the servos can already do, so "
        "its first rung is not the learning problem")
    # it must WORK BACKWARDS, and end at the real thing: the pose cache
    starts = [names.index(newest(st)) for st in ref.curriculum]
    assert all(b <= a for a, b in zip(starts, starts[1:])), starts
    assert newest(ref.curriculum[-1]) == "prone"

    # CUMULATIVE: every rung must keep the ones before it, or the policy stops
    # practising what it already learned. Measured: training each stage only on
    # its newest start collapsed the hold from 7.76 s to 0.24 s across a chain.
    sets = [set(st.env["MICRODUCK_ASIMOV_GETUP_START"].split(","))
            for st in ref.curriculum]
    for earlier, later in zip(sets, sets[1:]):
        assert earlier <= later, (
            f"a rung dropped a start it had already learned: "
            f"{sorted(earlier - later)}")


def test_the_arm_penalty_is_gone_from_the_tracking_reward():
    """Deliberate, and the reason is measured.

    Phase 3b's reward charged the robot for "straining its arm motors". On this
    body `quadruped` and `pike` cannot be reached without the elbows at 58-66 %
    of rated torque — the penalty was charging it for the only phase that lifts
    it. The torque envelope is a phase-4 MEASUREMENT, not a reward term.
    """
    env = _ref_env()
    env.reset(seed=0)
    _, terms = env._compute_reward()
    assert "arm_effort_penalty" not in terms, terms.keys()
    assert set(terms) == {"track", "contact", "advance", "up",
                          # the foot-span charge, added 2026-10-08. ZERO here,
                          # because `span_cost` defaults to 0 — it is a term
                          # the curriculum turns on, not a change to this
                          # reward. The arithmetic below is unaffected.
                          "span_penalty",
                          # the FLIGHT charge, added 2026-10-09, and zero
                          # here for the same reason: `ground_cost` defaults
                          # to 0. It exists because the first policy that
                          # stood from the floor got there by hopping.
                          "flight_penalty",
                          # the BALANCE charge, added 2026-10-09 — zero here
                          # (`balance_cost` defaults to 0), and gated on
                          # being on its feet, so it never touches a rise.
                          "balance_penalty",
                          # COME HOME and STOP JAMMING, added 2026-10-09.
                          # Both zero here (weights default to 0). The first
                          # policy to stand did it with EIGHT joints outside
                          # the model's limits, resting on the stops.
                          "home", "calm_penalty", "swing", "limit_penalty",
                          "action_rate_penalty"}, terms.keys()


def test_the_reported_waypoint_is_the_one_ACHIEVED_not_the_one_TARGETED():
    """An off-by-one here inverted a whole result, so it is pinned.

    The first chain was read as "reached `fold` 16/16" — crossing the one leg
    the position servos cannot do, which would have been the headline. What it
    had actually done was ACHIEVE `rock_back`, where the servos stop unaided,
    and advance the clock to AIM at `fold`. The metric was reporting the target.
    """
    env = _ref_env(start_phase="pike")
    n, info = _run(env)
    assert info["phase_name"] != info["targeting"], (
        "achieved and targeted are the same name, so one of them is not being "
        "reported — this is exactly the off-by-one")
    names = [w.name for w in R.WAYPOINTS]
    assert names.index(info["phase_name"]) < names.index(info["targeting"])


def test_a_fall_back_ends_the_episode():
    """The second fault of the first chain, and the same one as the original
    get-up env in a new place.

    A policy reached `rock_back` at step 13 and then lay on the floor for the
    remaining 7.7 s at pelvis 0.099 m, with a joint error of 0.326 against a
    0.60 tolerance — so it counted as TRACKING and collected `track` and
    `contact` every step. 400-step episodes of free reward for having fallen.
    """
    env = _ref_env(start_phase="pike")
    n, _ = _run(env)
    assert n < 120, (
        f"a zero-action episode from `pike` ran {n} steps — it falls almost "
        "at once, so the fall-back cut is not biting and falling is free")


def test_the_planted_regression_for_the_fall_back_cut():
    env = _ref_env(start_phase="pike")
    env.CUT_BACK_Z = 99.0                      # can never be crossed
    env.PHASE_BUDGET_S = 999.0                 # and nor can the deadline
    n, _ = _run(env)
    assert n == 400, (
        "disarming the fall-back cut did not restore the long episode, so "
        "test_a_fall_back_ends_the_episode is passing for another reason")


def test_the_cut_is_measured_from_the_achieved_waypoint_not_from_standing():
    """Because the reference's height is NOT monotone. `plant_forearms` drops
    the pelvis to 0.020 m on purpose — a cut measured from standing, or from the
    previous height, would end every episode for doing the move correctly."""
    zs = [w.settled_z for w in R.WAYPOINTS]
    assert min(zs[1:4]) < zs[0] + 0.02, \
        "plant_forearms no longer dips; re-check what the cut is measured from"
    env = _ref_env(start_phase="prone", track_tol=0.9)
    n, info = _run(env)
    assert n > 2, "the cut bites at the prone rung, where there is nowhere to fall"


def _stall_action(name):
    """Command the pose you are already in — the do-nothing strategy."""
    q = R.BY_NAME[name].joints
    a = (np.asarray(q) - np.asarray(W.DEFAULT_POSE)) / W.ACTION_SCALE
    return a.astype(np.float32).clip(-4, 4)


@pytest.mark.parametrize("rung", ["pike", "quadruped"])
def test_holding_a_waypoint_forever_is_cut(rung):
    """The reference has to PRESS, and achievement alone does not.

    Measured: the mean joint distance between consecutive waypoints is 0.087 to
    0.239 rad — every pair inside even the tightest not-tracking cut. So
    standing still counts as tracking the NEXT waypoint, and `pike` holds for
    6 s unaided, which makes "hold the pike for the whole episode" a safe
    400-step income of `track` and `contact` with no risk at all.
    """
    env = _ref_env(start_phase=rung)
    n, info = _run(env, action=_stall_action(rung))
    assert n < 160, (
        f"holding `{rung}` ran {n} steps — the stall deadline is not biting, "
        "so doing nothing is a better strategy than attempting the move")
    assert info["phase_name"] == rung, "it moved on, so this is not a stall"


def test_the_planted_regression_for_the_stall_deadline():
    env = _ref_env(start_phase="pike")
    env.PHASE_BUDGET_S = 999.0
    n, _ = _run(env, action=_stall_action("pike"))
    assert n == 400, (
        "removing the deadline did not restore the free episode, so "
        "test_holding_a_waypoint_forever_is_cut passes for another reason")


def test_the_deadline_does_not_apply_once_standing():
    """On the last waypoint, staying put IS the task — a deadline there would
    cut the robot for succeeding."""
    _, info = _run(_ref_env(start_phase="crouch"))
    assert info["reached_stand"] is True
    assert info["held_s"] > 5.0, (
        f"held only {info['held_s']} s — the deadline is cutting a robot that "
        "has already reached `stand`")


# ------------------------------------- the history CONTROL arm (roadmap 3d/2)

def test_the_contract_arm_is_untouched_by_the_control_arm():
    """`history=1` must be bit-identical to having no history code at all."""
    a = _ref_env(start_phase="pike")
    b = _ref_env(start_phase="pike", history=1)
    oa, _ = a.reset(seed=0)
    ob, _ = b.reset(seed=0)
    assert oa.shape == ob.shape == (78,)
    assert np.array_equal(oa, ob)


def test_the_history_arm_stacks_and_declares_its_width():
    env = _ref_env(start_phase="pike", history=5)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (390,)
    assert env.observation_space.shape == (390,)
    # the newest frame is the contract's own 78
    contract, _ = _ref_env(start_phase="pike").reset(seed=0)
    assert np.allclose(obs[:78], contract)


def test_the_stack_does_not_leak_across_episodes():
    """Stacking the last frame of the previous episode with the first of the
    next hands the policy the outcome of a run it is not in."""
    env = _ref_env(start_phase="pike", history=5)
    env.reset(seed=0)
    for _ in range(6):
        env.step(np.zeros(23, dtype=np.float32))
    obs, _ = env.reset(seed=1)
    assert np.allclose(obs[:78], obs[-78:]), "the stack carried frames across a reset"


def test_the_history_is_recorded_on_the_run(no_knob_env_ref, monkeypatch):
    """A 390-input policy is indistinguishable from a 78-input one by anything
    except the ONNX, so the run has to say which it is."""
    import types
    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_HISTORY", "5")
    kw = W.spec().train_env_kwargs(types.SimpleNamespace(task="getup_ref"))
    assert kw["history"] == 5


@pytest.fixture
def no_knob_env_ref(monkeypatch):
    for k in ("MICRODUCK_ASIMOV_GETUP_START", "MICRODUCK_ASIMOV_GETUP_TRACK_TOL",
              "MICRODUCK_ASIMOV_GETUP_REF_SPEED", "MICRODUCK_ASIMOV_GETUP_HISTORY",
              "MICRODUCK_ASIMOV_GETUP_POSE"):
        monkeypatch.delenv(k, raising=False)


# ------------------------------------------- FREE mode (roadmap phase 3e)

def test_free_mode_pays_the_product_and_no_known_exploit_beats_standing():
    """Free mode drops the imitation for the intermediates — the authored route
    through the gap was measured impossible, and a tracking reward against an
    unreachable waypoint CHARGES the policy for trying anything else.

    What replaces it has to resist the three rewards this task has already been
    beaten by, so they are pinned as numbers.
    """
    env = _ref_env(start_phase="pike", free=True)
    env.reset(seed=0)
    _, terms = env._compute_reward()
    assert set(terms) == {"track", "contact", "advance", "up",
                          # the foot-span charge, added 2026-10-08. ZERO here,
                          # because `span_cost` defaults to 0 — it is a term
                          # the curriculum turns on, not a change to this
                          # reward. The arithmetic below is unaffected.
                          "span_penalty",
                          # the FLIGHT charge, added 2026-10-09 — zero here
                          # (`ground_cost` defaults to 0), and it cannot be
                          # an exploit route: it only ever SUBTRACTS.
                          "flight_penalty",
                          # the BALANCE charge, added 2026-10-09 — zero here
                          # (`balance_cost` defaults to 0), and gated on
                          # being on its feet, so it never touches a rise.
                          "balance_penalty",
                          # COME HOME and STOP JAMMING, added 2026-10-09.
                          # Both zero here (weights default to 0). The first
                          # policy to stand did it with EIGHT joints outside
                          # the model's limits, resting on the stops.
                          "home", "calm_penalty", "swing", "limit_penalty",
                          "action_rate_penalty"}, terms

    w = env.W_FREE_RISE

    def pay(z, u):
        return w * (z / 0.639) * (u + 1.0) * 0.5

    standing = pay(0.608, 1.00)
    for name, z, u in (("the sit (run be7733)", 0.20, 0.90),
                       ("the shoulder stand (08bdf5)", 0.52, -0.57),
                       ("a somersault peak", 0.53, -0.92),
                       ("lying face down", 0.197, -0.35)):
        assert pay(z, u) < 0.5 * standing, f"{name} pays too well"


def test_free_mode_is_never_flat_where_the_policy_starts():
    """`max(u, 0)` made the term exactly 0.00 at EVERY rung below the gap,
    because uprightness is negative through all of them — the dead-gradient
    failure run 990c5d already paid for. Mapped to [0, 1] it rises from the
    floor."""
    env = _ref_env(start_phase="pike", free=True)
    w = env.W_FREE_RISE
    pays = [w * (x.settled_z / 0.639) * (x.settled_u + 1.0) * 0.5
            for x in R.WAYPOINTS]
    assert min(pays) > 0.0, pays
    assert pays[-1] > 10.0
    # and it must RISE along the reference, or there is nothing to climb
    assert pays[-1] > pays[0] * 5


def test_free_mode_deadline_measures_progress_not_waypoints():
    """Keeping a stall cut that fires unless the next AUTHORED waypoint is hit
    would still force the policy down the impossible route; it would only stop
    paying for the trip. In free mode the clock resets on a better rise."""
    # HOLD the pose rather than going limp: under a zero action the robot sinks
    # out of the pike and the FALL-BACK cut ends it in 6 steps, so the deadline
    # never gets a turn and the test would measure the wrong guard.
    hold = _stall_action("pike")
    env = _ref_env(start_phase="pike", free=True, track_tol=0.6)
    n, _ = _run(env, action=hold)
    assert n < 200, f"the deadline stopped biting ({n} steps)"
    env2 = _ref_env(start_phase="pike", free=True)
    env2.PHASE_BUDGET_S = 999.0
    n2, _ = _run(env2, action=hold)
    assert n2 > n, (
        "disarming the deadline changed nothing, so something else is ending "
        "the episode and this test is not measuring the deadline")


def test_free_mode_does_not_apply_the_not_tracking_cut():
    """There is nothing to track, so a tracking cut would end episodes for the
    policy doing what free mode asks."""
    env = _ref_env(start_phase="pike", free=True, track_tol=0.01)
    n, _ = _run(env)
    assert n > 5, "the not-tracking cut is still firing in free mode"


def test_the_rung_sampler_balances_STEPS_and_not_episodes():
    """Equal episodes is not equal data, and on this task they differ wildly.

    `fold` episodes run to the 400-step cap while `rock_back` and `pike` die in
    6-8 steps, so drawing rungs uniformly gave `fold` **58.6 %** of every
    training step and the two rungs the curriculum exists for **0.9 %** and
    **1.2 %**. The fix for catastrophic forgetting had starved the thing it was
    meant to enable. Sampling inversely to a running mean episode length
    equalises the step share and self-tunes as a hard rung survives longer.
    """
    env = _ref_env(start_phase="fold,rock_back,pike", free=True, force_lag=True)
    zero = np.zeros(len(W.JOINT_NAMES), dtype=np.float32)
    steps = {}
    for i in range(90):
        env.reset(seed=i)
        rung = env._start_phase
        n = 0
        for n in range(1, 401):
            *_, term, trunc, _info = env.step(zero)
            if term or trunc:
                break
        steps[rung] = steps.get(rung, 0) + n
    total = sum(steps.values())
    assert set(steps) == {"fold", "rock_back", "pike"}, steps
    for rung, n in steps.items():
        share = n / total
        assert share > 0.10, (
            f"{rung} gets {share:.1%} of the training steps — a rung starved "
            "this badly is not being trained on")


def test_the_sampler_adapts_when_a_rung_starts_surviving():
    """It is a running mean, so a rung that stops dying early is drawn less —
    otherwise balancing by steps would over-sample a rung that no longer needs
    it."""
    env = _ref_env(start_phase="fold,pike", free=True, force_lag=True)
    env.reset(seed=0)
    env._rung_len = {env._start_idxs[0]: 400.0, env._start_idxs[1]: 5.0}
    drawn = {}
    for i in range(80):
        env.reset(seed=i)
        drawn[env._start_phase] = drawn.get(env._start_phase, 0) + 1
        env._ep_steps = 0            # keep the seeded lengths
    assert drawn.get("pike", 0) > drawn.get("fold", 0) * 3, (
        f"the short-episode rung is not being favoured: {drawn}")


def test_every_waypoint_is_commandable_enough_to_track():
    """The policy commands `DEFAULT_POSE + 0.25 * action`, action clipped to
    +/-4.0, so it can only ask for +/-1.000 rad from standing — 76 % of this
    robot's joint range on average and 32 % on a wrist. Six of the nine
    waypoints need more than that on 2-6 joints each (`plant_forearms` 6.00 on
    its elbows, `quadruped` 5.20, `pike` and `draw_knees_in` 4.80, `fold` 5.00),
    which is the shape of `test_both_strikes_are_reachable_inside_the_action_clip`
    for the G1 — a rule this repo already had and never pointed at the get-up.

    It is a tax, not a wall, and the distinction is the whole finding.
    `_joint_err` is a MEAN absolute error over 23 joints, so truncating a few of
    them leaves a 0.02-0.04 rad floor and an 0.84-0.92 ceiling on the tracking
    term. MEASURED 2026-10-04, and the bound here is what fires if a pose is
    ever re-authored far enough out that a perfect policy could not be paid.

    The negative result that makes this a bound and not a fix: widening the clip
    to 13.0 (max|jnt_range - DEFAULT_POSE| / ACTION_SCALE, so the whole
    physically reachable set is commandable) made `fold` stop holding — 0.242 m
    at u -0.43 against its recorded 0.517 / +0.83 — because every verified hold
    and the one transition that works were measured on the CLIPPED pose. The
    authored pose is a different and unstable one. The open-loop chain dies at
    `rock_back` under BOTH clips, so the route is the problem, not the reach.
    """
    from microduck_local.robots.asimov_env import AsimovGetupRefEnv

    clip = 4.0                  # RobotSpec's inherited default, deliberately
    scale, std = W.ACTION_SCALE, AsimovGetupRefEnv.TRACK_STD
    ceilings = {}
    for wp in R.WAYPOINTS:
        pose = np.asarray(wp.joints, dtype=np.float64)
        need = (pose - W.DEFAULT_POSE) / scale
        reach = W.DEFAULT_POSE + scale * np.clip(need, -clip, clip)
        floor = float(np.abs(pose - reach).mean())
        ceilings[wp.name] = float(np.exp(-floor / std))
    worst = min(ceilings, key=ceilings.get)
    assert ceilings[worst] >= 0.80, (
        f"a perfect policy could only earn {ceilings[worst]:.2f} of the tracking "
        f"term at {worst!r} — the pose is outside what the clip can command; "
        f"re-author it inside +/-{scale * clip:.3f} rad of DEFAULT_POSE: {ceilings}")
    # the poses that need nothing clipped must be EXACTLY trackable
    for name in ("prone", "rock_back", "crouch", "stand"):
        assert ceilings[name] == pytest.approx(1.0), (name, ceilings[name])

    # planted regression: a pose a full radian past the clip must fail the bound
    far = W.DEFAULT_POSE + scale * clip * 2.0
    need = (far - W.DEFAULT_POSE) / scale
    reach = W.DEFAULT_POSE + scale * np.clip(need, -clip, clip)
    assert np.exp(-float(np.abs(far - reach).mean()) / std) < 0.80


def test_a_deep_squat_holds_for_a_second_and_is_on_the_floor_by_three():
    """There is no feet-only equilibrium below standing, and a 1 s check cannot
    see that. MEASURED 2026-10-04 over 700 leg configurations (hip -1.5..0.3,
    knee 0..1.5, ankle +/-0.35, placed upright, commanded to their own pose):

    | bar | passes |
    |---|---|
    | 1 s, feet-only, upright > 0.3 | **106 of 700** |
    | **6 s** | **1 of 106**, at pelvis 0.612 m — which IS standing |

    That is why the get-up cannot be staged: there is no state in the 0.33 m
    gap between the elbows-down rungs and `fold` to put a rung ON. This keeps
    two of those poses — one deep squat that passes at 1 s and fails at 6, and
    standing, which passes both — so a model or solver change that quietly
    creates (or destroys) a mid-height equilibrium shows up here.

    The episode's own rules are deliberately NOT in the loop. Breaking on
    `terminated` reported a 0.3 s episode as a 10 s hold and made a collapsing
    squat look like an equilibrium; `place()` after `reset()` trips the floor
    cut because the anchor is still the start rung's.
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    from probe_asimov_getup_ref import place

    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, start_phase="fold",
        track_tol=0.6, force_lag=True, free=True, getup_label="prone")
    d = env.data

    def trace(q, pitch=0.0):
        env.reset(seed=0)
        place(env, q, pitch)
        act = np.clip((np.asarray(q) - W.DEFAULT_POSE) / W.ACTION_SCALE,
                      -4.0, 4.0).astype(np.float32)
        out = []
        for k in range(300):                      # 6 s of PHYSICS, no cuts
            d.xfrc_applied[:] = 0.0
            env.step(act)
            out.append(float(env.data.xpos[env.trunk_body_id][2]))
        return out

    squat = trace(R.joints(hip_pitch=-0.3, knee=1.50, ankle_pitch=-0.35,
                           shoulder_pitch=-0.3, elbow=0.4))
    assert squat[24] > 0.30, (
        f"the squat was supposed to look fine at 0.5 s, got {squat[24]:.3f}")
    assert squat[-1] < 0.10, (
        f"a mid-height feet-only equilibrium now EXISTS at {squat[-1]:.3f} m — "
        "if that is real it is a rung for the gap and the roadmap's structural "
        "finding needs redoing, not this test")

    # ...and the floor of the standable region, which is where `fold` sits. The
    # first version of this test asserted NOTHING holds below standing; that was
    # a grid artefact (root pitch fixed at 0), and the lowest equilibria need
    # 14-40 deg of forward lean. 0.516 / 0.639 = 81 % of standing, the figure
    # add-a-robot already published from the +/-0.35 rad ankle.
    low = trace(R.joints(hip_pitch=-1.0, knee=1.00, ankle_pitch=-0.35,
                         shoulder_pitch=-0.4, elbow=0.6), pitch=0.25)
    assert low[-1] > 0.45, (
        f"the lowest standable pose stopped holding ({low[-1]:.3f} m) — the "
        "ankle range or the floor contact changed")
    assert abs(low[-1] - low[149]) < 0.03, "it drifts rather than holding"

    stand = trace(R.joints())                      # DEFAULT_POSE is standing
    assert stand[-1] > 0.55, f"standing stopped holding: {stand[-1]:.3f}"
    assert abs(stand[-1] - stand[24]) < 0.05, "standing drifted"


def test_the_harvested_clip_is_a_REACHABLE_reference():
    """The authored reference asked for joint angles the actuators cannot
    command: `ctrl = DEFAULT_POSE + 0.25 * action` with action clipped to +/-4
    reaches +/-1.000 rad, and six of the nine waypoints need up to 1.5 rad on a
    joint, so a perfect policy could earn at most 0.84-0.92 of the tracking term.

    A harvested clip cannot have that defect, because it is a recorded
    trajectory of a policy that actually stood up in this model
    (openhorizon-labs/asimov1-getup-safefall, 20/20 at a median 1.22 s on our
    own fetched asimov_1.xml). Replaying the demonstration's own joints through
    the tracker must therefore score FULL marks — and that is the control the
    authored reference never passed.
    """
    from pathlib import Path as _P

    import numpy as _np

    from microduck_local.robots import asimov
    from microduck_local.robots import spec as S

    clips = _P(asimov.CACHE_DIR) / "getup_clips.npz"
    if not clips.is_file():
        pytest.skip("no harvested clips (uv run python scripts/harvest_getup_clips.py)")

    env = S.get("asimov1").env_class("getup_clip")(
        domain_rand=False, obs_noise=False, seed=0)
    env.reset(seed=0)
    assert len(env._clips) >= 1 and len(env._clips) == len(env._clip_starts)

    got = []
    for t in range(len(env._clip)):
        env._ep_steps = t
        # The WHOLE pose, root included. This used to set only `env._q_adr` —
        # the 23 joints — leaving the root wherever reset put it, i.e. LYING ON
        # THE FLOOR, and it scored a perfect 10.0 of 10.0 anyway because `track`
        # priced joints alone. That is precisely the farm the reward was fixed
        # for (`test_lying_still_cannot_out_earn_getting_up`): a fallen robot
        # with the reference's joint angles is not on the reference. The test's
        # own setup was blind in the same place the reward was.
        env.data.qpos[:] = env._clip[t]
        mujoco.mj_forward(env.model, env.data)
        _r, terms = env._compute_reward()
        got.append(terms["track"])
    mean = float(_np.mean(got))
    assert mean > 0.99 * env.W_TRACK, (
        f"the demonstration scores only {mean:.2f} of {env.W_TRACK} against "
        "its own tracker — the reference or the index is wrong")

    # ...and a WRONG index must score badly, or the test proves nothing
    env._ep_steps = 0
    env.data.qpos[:] = env._clip[len(env._clip) // 2]
    mujoco.mj_forward(env.model, env.data)
    _r, off = env._compute_reward()
    assert off["track"] < 0.8 * env.W_TRACK, (
        f"tracking is insensitive to WHEN: {off['track']:.2f} for a frame from "
        "the middle of the clip scored at t=0")


def test_every_cached_rung_actually_HOLDS_when_replayed():
    """The invariant a curriculum depends on: a rung must hold.

    THE RETRACTION. This test used to assert that the stable set was two
    plateaus with a 15 cm BALLISTIC gap between them, which was reported as the
    best finding of 2026-10-06 and written into the roadmap. It was an artefact
    of the search, four ways over:

      * the 1.2 s filter required `trunk > 0.15 m`, so every rung near the
        floor was discarded by construction — the "nothing holds below 0.20 m"
        band was the filter;
      * the feet sweep pinned the ankle at -0.35, which emptied part of
        0.40-0.50;
      * the drift test compared 1.2 s against 6 s, so a pose that COLLAPSED in
        the first second and then lay still scored as a perfect hold — one
        claimed 0.456 m rung actually sits at 0.097 m;
      * the support classifier only looked for arms and ankles, so three poses
        resting on their PELVIS at 0.094 m were catalogued as feet-only
        equilibria.

    Searched properly the set is near-continuous: arm-supported 0.146-0.440 m
    (39 rungs), feet-only 0.471-0.606 m (5), a gap of 31 mm, and a real hole
    only at 0.20-0.25 m. The "one ballistic launch" story does not survive.

    So this asserts the thing that is actually load-bearing, and the thing all
    three earlier interpolated ladders failed: every pose in the cache must
    still be there 6 s after the env resets to it and commands it. The search
    applies that check itself now — of 55 poses that passed the earlier
    version, 11 failed this replay.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S

    if not asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    z = R.holdable_rungs()
    trunk_z, sup, J = z["trunk_z"], z["support"], z["joints"]
    assert len(trunk_z) > 20, f"only {len(trunk_z)} rungs cached"
    assert set(np.unique(sup)) <= {"arm", "feet"}

    env = S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, seed=0,
        action_clip=float(S.get("asimov1").action_clip))
    m, d = env.model, env.data
    scale = float(np.atleast_1d(env.robot.action_scale)[0])
    fails = []
    for i in range(len(trunk_z)):
        d.qpos[:] = np.asarray(J[i])
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        act = np.clip((d.qpos[env._q_adr] - np.asarray(env.robot.default_pose))
                      / scale, -float(S.get("asimov1").action_clip),
                float(S.get("asimov1").action_clip)).astype(np.float32)
        z0 = float(d.xpos[env.trunk_body_id][2])
        for _ in range(int(R.HOLDABLE_HOLD_S * 50)):
            d.xfrc_applied[:] = 0.0
            env._write_ctrl(env.robot.scale_action(act))
            for _ in range(4):
                mujoco.mj_step(m, d)
        z6 = float(d.xpos[env.trunk_body_id][2])
        if abs(z6 - z0) > R.HOLDABLE_DRIFT_M:
            fails.append((round(z0, 3), round(z6, 3)))
    assert not fails, (
        f"{len(fails)} of {len(trunk_z)} cached rungs do NOT hold for "
        f"{R.HOLDABLE_HOLD_S} s: {fails[:5]} (placed -> after). A curriculum "
        "built on these is the interpolated-ladder failure again.")

    # ...and a rung must be somewhere USEFUL: off the floor, below standing.
    assert trunk_z.min() > 0.12, f"lowest rung {trunk_z.min():.3f} m is the floor"
    assert trunk_z.max() < 0.639, f"highest rung {trunk_z.max():.3f} m is standing"


def test_hip_roll_and_hip_yaw_MIRROR_so_a_splay_is_not_a_lean():
    """`MIRRORED` was missing `hip_roll` and `hip_yaw`, and the cost was a
    whole plateau.

    The five stems originally listed mirror because the right-side axis is -Y
    against the left's +Y. These two are the opposite case: both sides share
    the SAME axis (+X for roll, -Z for yaw), so equal values rotate both legs
    the same way in WORLD space. Left out of the tuple, `joints(hip_roll=v)`
    therefore produced a LEAN where every other kwarg produces a symmetric
    pose — and since `_search_rungs` builds every pose through this helper and
    never passed these stems, both sweeps only ever searched a stance 0.215 m
    wide, where the support set is a line. That is why the feet-only plateau
    was reported to bottom out at 0.471 m.

    The numbers below are measured, and they are what separates the two cases:
    a lean keeps the width and moves the midline, a splay moves the width and
    keeps the midline.
    """
    import numpy as np

    from microduck_local.robots import asimov as _asimov
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    if not _asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    assert "hip_roll" in R.MIRRORED and "hip_yaw" in R.MIRRORED

    idx = {n: i for i, n in enumerate(W.JOINT_NAMES)}
    # the axes are the REASON, so assert them rather than the conclusion alone
    env = S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, seed=0,
        action_clip=float(S.get("asimov1").action_clip))
    m, d = env.model, env.data
    for stem, axis in (("hip_roll", 0), ("hip_yaw", 2)):
        la = m.joint(f"left_{stem}_joint").axis
        ra = m.joint(f"right_{stem}_joint").axis
        assert np.allclose(la, ra), (
            f"{stem} axes differ between sides {la} vs {ra} — if the model "
            "changed, the reason this stem is in MIRRORED changed with it")
        assert abs(la[axis]) > 0.99
    for stem in ("hip_pitch", "knee", "ankle_pitch"):
        la = m.joint(f"left_{stem}_joint").axis
        ra = m.joint(f"right_{stem}_joint").axis
        assert np.allclose(la, -np.asarray(ra)), f"{stem} should be anti-parallel"

    # the helper negates the right side...
    q = R.joints(hip_roll=0.7)
    assert q[idx["left_hip_roll_joint"]] == pytest.approx(0.7)
    assert q[idx["right_hip_roll_joint"]] == pytest.approx(-0.7)

    # ...and passing a joint its own default is still a no-op, the property
    # that makes a pose draft readable
    assert np.allclose(R.joints(hip_roll=0.0, hip_yaw=0.0),
                       np.array(W.DEFAULT_POSE))

    LF = m.body("left_ankle_roll_link").id
    RF = m.body("right_ankle_roll_link").id

    def stance(roll_left, roll_right):
        p = np.array(W.DEFAULT_POSE, dtype=float)
        p[idx["left_hip_pitch_joint"]] = -1.20
        p[idx["right_hip_pitch_joint"]] = +1.20
        p[idx["left_knee_joint"]] = +1.35
        p[idx["right_knee_joint"]] = -1.35
        p[idx["left_hip_roll_joint"]] = roll_left
        p[idx["right_hip_roll_joint"]] = roll_right
        env._place(p, 0.25)
        mujoco.mj_forward(m, d)
        ly, ry = float(d.xpos[LF][1]), float(d.xpos[RF][1])
        return abs(ly - ry), (ly + ry) / 2.0

    w0, mid0 = stance(0.0, 0.0)
    w_lean, mid_lean = stance(0.7, 0.7)      # what the helper used to produce
    w_splay, mid_splay = stance(0.7, -0.7)   # what it produces now

    assert w0 == pytest.approx(0.215, abs=0.01)
    # a LEAN: width unchanged, midline walks sideways
    assert w_lean == pytest.approx(w0, abs=0.01), (
        f"equal rolls changed the stance width {w0:.3f} -> {w_lean:.3f}; the "
        "lean/splay distinction this test rests on is gone")
    assert abs(mid_lean) > 0.15, f"equal rolls did not shift the midline ({mid_lean:+.3f})"
    # a SPLAY: width opens, midline stays put
    assert w_splay > 0.60, f"mirrored roll only reached {w_splay:.3f} m of stance"
    assert abs(mid_splay) < 0.01, f"mirrored roll shifted the midline {mid_splay:+.3f}"

    # and `joints()` must take the splay branch, not the lean one
    qs = R.joints(hip_pitch=-1.20, knee=1.35, hip_roll=0.7)
    env._place(qs, 0.25)
    mujoco.mj_forward(m, d)
    got = abs(float(d.xpos[LF][1] - d.xpos[RF][1]))
    assert got == pytest.approx(w_splay, abs=0.01), (
        f"joints(hip_roll=0.7) produced a {got:.3f} m stance, not the "
        f"{w_splay:.3f} m splay — hip_roll is not mirroring")


def _feet_rungs_by_splay():
    """Cached feet-only rungs, split on whether the hips are splayed."""
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    z = R.holdable_rungs()
    tz = z["trunk_z"]
    sup = np.array([str(s) for s in z["support"]])
    J = z["joints"]
    env = S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, seed=0,
        action_clip=float(S.get("asimov1").action_clip))
    roll_i = {n: i for i, n in enumerate(W.JOINT_NAMES)}["left_hip_roll_joint"]
    rolls = np.array([abs(float(np.asarray(J[i])[env._q_adr][roll_i]))
                      for i in range(len(tz))])
    feet = sup == "feet"
    return tz, sup, feet, rolls


def test_the_splay_is_what_reaches_below_the_old_feet_only_floor():
    """The splay's contribution, measured against its own control.

    CORRECTED 2026-10-07. The first version of this test asserted the two
    plateaus OVERLAP. They do at an action clip of 13.0 — which is what
    `_search_rungs` used to hardcode, and which nothing else in the repo uses.
    At the robot's own clip the cache is 134 rungs, not 221, and the picture
    is the opposite:

        arm plateau  0.134 - 0.280 m (56)
        feet plateau 0.416 - 0.606 m (78)
        a 136 mm GAP between them

    The splay still earns its place, and by more than before: feet-only rungs
    with the hips square bottom out at 0.529 m, splayed ones reach 0.416 m —
    **113 mm**, against 53 mm when this was measured at the wrong clip.

    Both numbers come from the same sweep, which is the only comparison that
    means anything: quoting the move from an older cache would conflate the
    splay with a denser knee grid and an unpinned arm reach.
    """

    from microduck_local.robots import asimov as _asimov

    if not _asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    tz, sup, feet, rolls = _feet_rungs_by_splay()
    flat = feet & (rolls < 0.01)
    wide = feet & (rolls >= 0.01)

    assert flat.sum() >= 5, f"only {int(flat.sum())} unsplayed feet rungs — no control"
    assert wide.sum() >= 20, f"only {int(wide.sum())} splayed feet rungs"

    # the control: hips square cannot get down here
    assert tz[flat].min() > 0.50, (
        f"unsplayed feet-only reached {tz[flat].min():.3f} m — if a square "
        "stance now goes this low, the splay is not what buys the depth")
    # ...and the splay does, by a margin far outside any grid jitter
    assert tz[wide].min() < 0.45, (
        f"splayed feet-only only reached {tz[wide].min():.3f} m")
    assert tz[flat].min() - tz[wide].min() > 0.08, (
        f"the splay bought only {(tz[flat].min()-tz[wide].min())*1000:.0f} mm")

    # AND the thing the first version got wrong: at the real clip the plateaus
    # do NOT meet. This is the headline the curriculum has to respect.
    arm_top = tz[sup == "arm"].max()
    gap = tz[feet].min() - arm_top
    assert gap > 0.05, (
        f"the arm plateau now tops at {arm_top:.3f} and feet start at "
        f"{tz[feet].min():.3f} — they meet or overlap, which they did only at "
        "the 13.0 clip nothing uses. If this is real, re-derive phase 3h.")


def test_MIRRORED_is_complete_rederived_from_the_model():
    """`MIRRORED` must list EVERY left/right stem, and the model says which.

    The tuple was wrong twice. `hip_roll`/`hip_yaw` missing cost a whole
    plateau — `joints(hip_roll=v)` leaned the robot where it reads as a splay,
    so both rung sweeps only ever searched a 0.215 m stance. The audit that
    found it then found `shoulder_roll`, `shoulder_yaw`, `ankle_roll` and
    `wrist_yaw` missing too.

    So this does not hard-code the answer: it re-derives the convention from
    the model by mirroring each left link's world ORIENTATION across the
    sagittal plane and comparing it to its right counterpart, and asserts the
    tuple agrees. Orientation, not position, because `ankle_roll` and
    `wrist_yaw` are terminal joints whose child link rotates in place — a
    position-based probe is blind to them and silently reports "no convention".

    Reasoning from `joint.axis` is NOT a substitute and was tried first: five
    stems have anti-parallel axes and six share an axis, and both groups turn
    out to negate.
    """
    import numpy as np

    from microduck_local.robots import asimov as _asimov
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    if not _asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    env = S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, seed=0)
    m, d = env.model, env.data
    idx = {n: i for i, n in enumerate(W.JOINT_NAMES)}
    names = [m.body(i).name for i in range(m.nbody)]
    pairs = [(m.body(n).id, m.body("right_" + n[5:]).id) for n in names
             if n.startswith("left_") and "right_" + n[5:] in names]
    assert len(pairs) >= 8, f"only {len(pairs)} mirrored body pairs to judge on"
    MIR = np.diag([1.0, -1.0, 1.0])

    def residual(vl, vr, stem):
        q = np.array(W.DEFAULT_POSE, dtype=float)
        q[idx[f"left_{stem}_joint"]] = vl
        q[idx[f"right_{stem}_joint"]] = vr
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[env._q_adr] = q
        mujoco.mj_forward(m, d)
        return sum(float(np.abs(MIR @ d.xmat[L].reshape(3, 3) @ MIR
                                - d.xmat[Rb].reshape(3, 3)).sum())
                   for L, Rb in pairs)

    stems = sorted({n[len("left_"):-len("_joint")] for n in W.JOINT_NAMES
                    if n.startswith("left_")})
    negating = []
    for s in stems:
        neg, same = residual(0.3, -0.3, s), residual(0.3, 0.3, s)
        assert neg < same, (
            f"{s}: mirroring gave residual {neg:.6f} against {same:.6f} for "
            "equal values — this robot's convention is not what MIRRORED assumes")
        assert neg < 1e-6, (
            f"{s}: even the mirrored pose has residual {neg:.6f}; the model's "
            "two sides are no longer mirror images and this test cannot judge")
        negating.append(s)

    assert set(negating) == set(R.MIRRORED), (
        f"MIRRORED disagrees with the model — missing "
        f"{sorted(set(negating) - set(R.MIRRORED))}, spurious "
        f"{sorted(set(R.MIRRORED) - set(negating))}")

    # the property the tuple exists for: a joint passed its own default is a
    # no-op. `shoulder_roll` failed this before the audit (-0.050 vs +0.050).
    for s in stems:
        v = float(np.array(W.DEFAULT_POSE)[idx[f"left_{s}_joint"]])
        assert np.allclose(R.joints(**{s: v}), np.array(W.DEFAULT_POSE)), (
            f"joints({s}={v:+.3f}) is not a no-op — it does not round-trip "
            "its own default")


def test_no_curriculum_stage_starts_on_a_rung_its_own_clip_cannot_hold():
    """The guard for the fault that cost `teach-asimov_getup_floor-cdd31c`.

    `_search_rungs` hardcoded an action clip of 13.0 while every trainer runs
    at `RobotSpec.action_clip`. So the cache certified 221 rungs, of which only
    82 hold at the clip training uses — and the `asimov_getup_floor` chain then
    started its episodes on poses the policy physically cannot hold. All 64 of
    stage 2's rungs demanded more than its clip. The chain scored **0/36 from
    the floor at every one of its five stages**, identical to commanding
    nothing, and nothing in training reported a problem: each stage's own
    numbers improved, because each was measured where it started.

    A rung has to be an equilibrium IN THE ENV THAT TRAINS ON IT. This asserts
    the two halves of that: the search derives its clip from the robot rather
    than naming one, and no stage of a rung-based curriculum trains at a
    different clip from the one the cache was built at.
    """
    import inspect
    import re

    from microduck_local import behaviors as B
    from microduck_local.robots import asimov as _asimov
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S

    if not _asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    src = inspect.getsource(R._search_rungs)
    code = re.sub(r"^\s*#.*$", "", src, flags=re.M)      # comments may cite it
    assert "action_clip" in src and "S.get" in src, (
        "_search_rungs no longer derives its clip from the robot spec")
    for bad in re.findall(r"action_clip\s*=\s*([0-9.]+)", code):
        pytest.fail(f"_search_rungs hardcodes action_clip={bad}; it must read "
                    "the spec, or the cache certifies rungs nothing trains on")

    spec_clip = float(S.get("asimov1").action_clip)
    for bid in ("asimov_getup_floor", "asimov_getup"):
        beh = B.BEHAVIORS.get(bid)
        if beh is None:
            continue
        for i, stage in enumerate(getattr(beh, "curriculum", ()) or (), 1):
            envd = dict(getattr(stage, "env", None) or {})
            rung = envd.get("MICRODUCK_ASIMOV_GETUP_START_RUNG", "")
            if not rung or rung == "floor":
                continue                      # starts where it fell, no rung
            override = envd.get("MICRODUCK_ASIMOV_GETUP_ACTION_CLIP")
            if override is None:
                continue                      # inherits the spec clip: correct
            assert float(override) == spec_clip, (
                f"{bid} stage {i} ({stage.label!r}) starts on rung band "
                f"{rung!r} but trains at clip {override}, while the rung cache "
                f"was validated at {spec_clip}. Those rungs are not "
                "equilibria in that env.")


def test_the_scripted_ramp_lands_between_the_two_plateaus():
    """The decomposition the handover task rests on.

    The arm plateau traverses OPEN LOOP (phase 3i): position servos only, no
    policy, 0.136 -> 0.280 m in seven legs at every tempo from 0.15 s to
    0.90 s. So the bottom of the get-up is played, not learned, and what is
    left to learn is the one transition above it.

    This pins where that ramp leaves the robot, because the whole premise
    fails quietly if it drifts: too low and the policy is being asked to
    re-learn the arm plateau, too high and the task is not the handover at
    all. It must land ABOVE the arm plateau's useful top and BELOW the
    feet-only plateau, inside the span cut, on forearms and feet.
    """
    import numpy as np

    from microduck_local.robots import asimov as _asimov
    from microduck_local.robots import asimov_getup_ref as R

    if not _asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    h = R.handover_states()
    q, v, z = h["qpos"], h["qvel"], h["trunk_z"]
    assert len(z) >= 8, f"only {len(z)} handover states — a rung is a SET"

    rungs = R.holdable_rungs()
    sup = np.array([str(x) for x in rungs["support"]])
    feet_floor = float(rungs["trunk_z"][sup == "feet"].min())
    arm_top = float(rungs["trunk_z"][sup == "arm"].max())

    assert z.min() > 0.20, (
        f"the ramp only reaches {z.min():.3f} m — it is not clearing the arm "
        "plateau, so the handover task still contains the push-up")
    assert z.max() < feet_floor, (
        f"the ramp reaches {z.max():.3f} m, at or above the feet-only floor "
        f"{feet_floor:.3f} — there would be no gap left to learn")
    assert z.max() <= arm_top + 0.02, (
        f"the ramp ends at {z.max():.3f} m, above the arm plateau's top "
        f"{arm_top:.3f}; it is no longer arm-supported")

    # and the states must be usable: inside the span cut the task trains under
    from microduck_local.robots.asimov_env import AsimovGetupEnv

    env = AsimovGetupEnv(domain_rand=False, obs_noise=False, getup_blend=0.0,
                         getup_assist=0.0, start_rung="handover",
                         max_span=R.EPISODE_SPAN_CUT_M)
    spans = []
    for i in range(len(z)):
        env.data.qpos[:] = q[i]
        env.data.qvel[:] = v[i]
        mujoco.mj_forward(env.model, env.data)
        spans.append(env._foot_span())
    spans = np.array(spans)
    # the EPISODE cut, not the rung filter: the ramp overshoots the rungs it
    # passes through (0.705-0.724 against <=0.70), and setting the two equal
    # voided a whole training run by ending every episode on its start pose
    assert spans.max() <= R.EPISODE_SPAN_CUT_M + 1e-9, (
        f"a handover state spans {spans.max():.3f} m against a cut at "
        f"{R.EPISODE_SPAN_CUT_M} — the episode would end on its own start pose")
    assert R.EPISODE_SPAN_CUT_M < 0.90, (
        f"the episode cut is {R.EPISODE_SPAN_CUT_M} m, which admits the "
        "teacher's 0.929 m sprawl — it forbids nothing")

    # the gap it leaves is the thing worth learning, and it must be real
    assert feet_floor - z.max() > 0.08, (
        f"only {(feet_floor - z.max())*1000:.0f} mm between the handover and "
        "the feet plateau — if this closed, the task is no longer a launch")
