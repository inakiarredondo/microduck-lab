"""The get-up env's INSTRUMENTATION — what a run records, and when it ends.

Four get-up runs (`docs/roadmap.md` Track 14) produced four unreadable
results, and these are the two faults that made them unreadable:

1. **Nothing recorded the curriculum.** `AsimovGetupEnv` read its blend and
   assist straight out of `os.environ`, so `run.json` carried only
   `{domain_rand, obs_noise}` and no stage said which rung it trained at.
   Every later render then took those knobs from whatever shell it ran in.
   `robots/moss.py` already states the rule: *"an env that reads its own
   environment trains fine and leaves no record of what it trained on."*
2. **Falling did not end the episode.** The inherited cut is off because a
   get-up BEGINS past it, and nothing replaced it — so an 8 s episode that
   toppled at 0.34 s spent 96 % of its steps on the floor still collecting
   `rise` and `stand`, and no return told a fall at 0.34 s from one at 7 s.

The replacement cut is armed by having STOOD, which is the only version that
can coexist with a reset pose that starts fallen. `test_the_cut_is_what_ends_a
_fallen_episode` plants the regression that proves the arming is load-bearing
(`AGENTS.md`: a test proves nothing until it has been shown to fail).

Skipped, not failed, when Asimov's model is not downloaded.
"""

from __future__ import annotations

import types
from pathlib import Path as _P

import numpy as np
import pytest

from microduck_local.robots import asimov
from microduck_local.robots import asimov_walk as W

pytestmark = pytest.mark.skipif(
    not asimov.ready(),
    reason="Asimov's model is not fetched (uv run fetch-robot asimov1)")

ZERO = np.zeros(len(W.JOINT_NAMES), dtype=np.float32)
KNOB_ENV = ("MICRODUCK_ASIMOV_GETUP_BLEND", "MICRODUCK_ASIMOV_GETUP_ASSIST",
            "MICRODUCK_ASIMOV_GETUP_POSE")


@pytest.fixture
def no_knob_env(monkeypatch):
    """No ambient knob reaches the env — which is the state a test must run in
    and the state four training runs were never checked in."""
    for k in KNOB_ENV:
        monkeypatch.delenv(k, raising=False)


def _env(**kw):
    from microduck_local.robots.asimov_env import AsimovGetupEnv
    kw.setdefault("domain_rand", False)
    kw.setdefault("obs_noise", False)
    return AsimovGetupEnv(**kw)


def _episode(env, shove=None, cap=400):
    """Run to termination or `cap`; returns (steps, terminated, info)."""
    env.reset(seed=0)
    tb = env.trunk_body_id
    mass = float(env.model.body_subtreemass[tb])
    info: dict = {}
    for n in range(1, cap + 1):
        if shove is not None:
            lo, hi, g = shove
            env.data.xfrc_applied[:] = 0.0
            if lo <= n - 1 < hi:
                env.data.xfrc_applied[tb, 0] = g * mass * 9.81
        *_, terminated, truncated, info = env.step(ZERO)
        if terminated or truncated:
            return n, bool(terminated), info
    return cap, False, info


# ------------------------------------------------------ what the run records

def test_the_knobs_arrive_as_kwargs(no_knob_env):
    """With no environment variable set anywhere, the kwargs still land."""
    env = _env(getup_blend=0.8, getup_assist=0.25)
    assert env._blend == pytest.approx(0.8)
    assert env._assist == pytest.approx(0.25)


def test_the_env_var_still_reaches_a_direct_construction(no_knob_env, monkeypatch):
    """The lab reaches the trainer subprocess through the environment, so the
    fallback stays — it is the TRAINER that must not depend on it."""
    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_BLEND", "0.55")
    assert _env()._blend == pytest.approx(0.55)


def test_train_env_kwargs_puts_the_curriculum_on_the_run(no_knob_env, monkeypatch):
    """The fix for fault 1: the trainer reads the knobs and records them."""
    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_BLEND", "0.55")
    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_ASSIST", "0.3")
    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_POSE", "prone")
    from microduck_local.robots.asimov_env import AsimovGetupEnv as E

    kw = W.spec().train_env_kwargs(types.SimpleNamespace(task="getup"))
    # EXACT on purpose: a new knob has to be added here deliberately, which is
    # what caught `action_clip` and the two arm weights when they were added.
    assert kw == {"getup_label": "prone", "getup_assist": 0.3,
                  "getup_blend": 0.55, "action_clip": 4.0,
                  "arm_push": E.W_ARM_PUSH, "arm_effort": E.W_ARM_EFFORT,
                  "head_up": E.W_HEAD_UP, "start_rung": "",
                  # the sprawl cut and the fall-depth curriculum axis, added
                  # 2026-10-08. Both were read from the environment and
                  # trained with while `run.json` said nothing:
                  # `teach-asimov_getup_catch-23657c` recorded
                  # start_rung="falling" but not WHICH DEPTH, and the depth is
                  # the entire curriculum.
                  "max_span": 0.0, "fall_depth": 1.0,
                  # the launch-window span, added 2026-10-08 — the hybrid's
                  # whole mechanism, and the third knob in a row that trained
                  # while `run.json` said nothing about it
                  "launch_span": 0.0,
                  # the CONTINUOUS span charge, added 2026-10-08 after the
                  # termination version taught the policy to stop standing
                  # rather than to stand narrow
                  "span_cost": 0.0, "span_target": 0.40,
                  # the scripted push-up, added 2026-10-09: a run whose
                  # episodes begin with 1.56 s of servo-driven prone->pike is
                  # not the same experiment as one starting at a placed pike
                  "prologue": "",
                  # ROTATION pay, added 2026-10-09. The get-up's blocker is
                  # rotation and `progress` pays only height, so a run with
                  # this on optimises a different thing.
                  "rotate_pay": 0.0,
                  # ROLL ONTO THE BACK, added 2026-10-13 — and this exact
                  # assertion is what caught it going unrecorded. The first
                  # launch sent ROLL=100 and `run.json` said nothing, because
                  # `train_env_kwargs` has THREE getup dicts and only two had
                  # been edited; the run trained the previous null under a new
                  # name. Sixth time for this bug in this file.
                  "roll_pay": 0.0,
                  # whether a robot that stood and FELL ends the episode.
                  # On for training; off for the recovery loop and the lab,
                  # where `Duck.step` resets on `terminated` and a viewer
                  # would otherwise see a respawn that looks like a recovery.
                  "fell_cut": True,
                  # seconds STANDING between shoves; 0 keeps the training
                  # cadence, which counts wall time and so hits the robot
                  # again the instant a 3.4 s rise finishes.
                  "push_every": 0.0,
                  # limb speed during the RISE, which the original
                  # joint_vel_penalty never charged because it is gated on
                  # already being upright.
                  "rise_vel_cost": 0.0,
                  # WHICH ACTUATOR, added 2026-10-13 — and this is the knob
                  # that makes every run above it a claim about a different
                  # robot. "xml" clamps the PD output at the URDF's
                  # continuous effort with no speed dependence; "dc" clamps
                  # it inside the published speed-torque envelope. Default
                  # "xml" so no historical run's recorded kwargs change
                  # meaning; the retrain sets it explicitly.
                  "actuator_force": "xml",
                  # HOW MANY FRAMES, added 2026-10-13 when `history` moved up
                  # to AsimovGetupEnv and this task could suddenly use it. A
                  # run at 5 frames and one at 1 are not two seeds of one
                  # experiment — they cannot even load in the same env (390
                  # floats against 78), and the ablation says 1 cannot do this
                  # task at all (the reference policy goes 24/24 -> 0/24 with
                  # its ring frozen to the present).
                  "history": 1,
                  # getting the FEET UNDER THE MASS, added 2026-10-09 — the
                  # move every other term charges for
                  "tuck_pay": 0.0,
                  # the FLIGHT charge, added 2026-10-09. The first policy that
                  # stood from the floor got there by HOPPING — 1.04 s of its
                  # 3.4 s rise with zero floor contacts, 105 mm of clearance —
                  # because every term paid for what a bounce produces and
                  # none priced the impulse.
                  "ground_cost": 0.0,
                  # the BALANCE charge, added 2026-10-09. While "standing",
                  # that policy's CoM was outside its own support polygon
                  # 74% of the time — it was never balanced, it passed
                  # through vertical on the way over.
                  "balance_cost": 0.0,
                  # COME HOME, and STOP JAMMING — added 2026-10-09 after the
                  # standing pose was measured 19 deg from `default_pose`
                  # (57 deg at the waist) with 8 joints past their stops.
                  "home_pay": 0.0, "limit_cost": 0.0,
                  # Added 2026-10-21. Their defaults are the CLASS
                  # CONSTANTS, not 0.0: "off" for these two means the
                  # charge the robot has always carried, so a run that
                  # recorded 0.0 would have DELETED it.
                  "joint_vel_cost": E.W_JOINT_VEL,
                  "over_clip_cost": E.W_OVER_CLIP,
                  # SHOVES, added 2026-10-10. In 38 M steps this robot was
                  # never once disturbed, so a recovery was a state no
                  # rollout contained — "it never corrects" could not have
                  # been fixed by any reward.
                  "push_bw": 0.0,
                  # ARRIVE QUIETLY, added 2026-10-10. It reaches standing at
                  # 0.89 m/s against a stance that absorbs 0.22 — the rise is
                  # a lunge, and no pose survives that.
                  "calm_cost": 0.0,
                  # SWING THE ARMS UP, added 2026-10-10 from Jonathan's
                  # suggestion. Measured: the symmetric swing is already the
                  # strongest correlate of the trunk rising (+0.287) and
                  # nothing paid for it.
                  "swing_pay": 0.0,
                  # TRAIN FOR THE HANDOFF, added 2026-10-10. The deliverable
                  # is a catchable state for Menlo's walker, not a solo
                  # 8-second stand — which is why the policy jams.
                  "handoff_pay": 0.0,
                  # THE THREE SCAFFOLDING KNOBS, added 2026-10-15, and the
                  # first of them is why this assertion exists at all in its
                  # EXACT form: `getup_attitude` was added to the env and not
                  # to the recorder, then trained at 120 for a whole
                  # 8,429,568-step chain while `run.json` said `None`. The
                  # env reads its own environment, so nothing errored — and
                  # `score_gap_chain` / `getup_acceptance` rebuild eval envs
                  # from `replay_kwargs`, which means every later score was of
                  # a robot WITHOUT the scaffold it trained with. Seventh time
                  # for this bug in this file.
                  #
                  # `getup_attitude`: a clamped PD on uprightness toward
                  # `fold`'s +0.83, because the LIFT assist buys height by
                  # tipping the trunk over its planted forearms (upright at
                  # the peak -0.53 at assist 0.00 down to -1.00 at 0.85, and
                  # zero action reproduces it).
                  # `fold_pay`: per-step pay for dwelling in a state the
                  # narrow stand policy can catch — joints within 0.10 rad of
                  # `fold`, lift <= 0.20 m/s, feet only, span <= 0.30.
                  # `arm_unload`: a charge on weight carried through the arms
                  # while NOT rising, which is what finally got the hands off
                  # the floor (feet-only frames 0.0 % -> 35.3 %).
                  # `span_axis` / `span_band`: the span charge measures ONE
                  # axis and only inside a pelvis band, because `_foot_span`
                  # conflates the two and the Asimov sprawl is a LUNGE —
                  # 0.353 m lateral against 0.702 m fore-aft — so a charge on
                  # the conflated span prices the wrong axis.
                  "getup_attitude": 0.0, "fold_pay": 0.0,
                  "arm_unload": 0.0, "arm_work": 0.0,
                  "tuck_work": 0.0, "tuck2": 0.0, "rotate_work": 0.0, "rise_work": 0.0,
                  "span_axis": "both",
                  "span_band": ""}


def test_the_walk_task_records_no_getup_knobs(no_knob_env):
    """A walk run must not claim a get-up rung it never had."""
    assert W.spec().train_env_kwargs(types.SimpleNamespace(task="walk")) == {}


def test_every_recorded_knob_is_a_kwarg_the_env_accepts(no_knob_env, monkeypatch):
    """The failure this guards is a knob recorded under a name the env ignores:
    `run.json` would then describe a run nobody trained."""
    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_ASSIST", "0.4")
    kw = W.spec().train_env_kwargs(types.SimpleNamespace(task="getup"))
    env = _env(**kw)                       # would TypeError on an unknown name
    assert env._assist == pytest.approx(0.4)
    assert env._getup_label == kw["getup_label"]


# -------------------------------------------------------- when it ends, and why

def test_a_pose_that_never_stands_runs_its_full_episode(no_knob_env):
    """The negative control. From the real fallen pose a zero-action robot never
    satisfies the gate, so the cut must never arm."""
    steps, terminated, info = _episode(_env(getup_blend=0.0))
    assert steps == 400 and not terminated
    assert info["stood"] is False
    assert info["held_s"] == 0.0


def test_the_cut_is_what_ends_a_fallen_episode(no_knob_env):
    """The positive, and its planted regression.

    At blend 1.0 the pose IS `DEFAULT_POSE` and a zero-action Asimov holds it
    (measured 2026-10-04, 6/6) — so it stands, the cut arms, and a shove ends
    the episode early. Disarming the cut must put those steps back, or this
    test is measuring something else.
    """
    shove = (100, 125, 0.8)
    steps, terminated, info = _episode(_env(getup_blend=1.0), shove=shove)
    assert terminated, "the cut did not fire on a robot that stood and then fell"
    assert steps < 400, "the episode ran to truncation despite falling"
    assert info["stood"] is True

    disarmed = _env(getup_blend=1.0)
    disarmed.FELL_GRAVITY = disarmed.FELL_HEIGHT_FRAC = -9.0   # uncrossable
    plant_steps, plant_term, _ = _episode(disarmed, shove=shove)
    assert plant_steps == 400 and not plant_term, (
        "the planted regression changed nothing, so the assertions above are "
        "not testing the cut")


def test_held_s_measures_the_longest_hold_not_the_total(no_knob_env):
    """`standing` already sums the gate. A sum cannot tell 40 scattered frames
    from one 0.8 s hold, and that difference is the whole task."""
    _, _, info = _episode(_env(getup_blend=1.0), shove=(100, 125, 0.8))
    # It held from step 0 to the shove, ~2 s, and never recovered after.
    assert 1.5 < info["held_s"] < 3.0, info["held_s"]
    assert info["peak_rise"] > 0.9


def test_the_gate_is_satisfiable_on_the_way_down(no_knob_env):
    """Why `standing` 2.19 was never a skill.

    The blend-0.8 rung starts at height 0.995 and u_z 0.94 — both past the
    `up` gate — and topples. The gate is therefore brushed for a few frames by
    the reset pose DECAYING, with no policy acting at all. Pinned here because
    it is the measurement that retired the reverse curriculum, and because any
    future reward that pays `up` per step inherits it.
    """
    _, terminated, info = _episode(_env(getup_blend=0.8))
    assert info["stood"] is True, "the gate was not brushed — remeasure the rung"
    assert info["held_s"] <= 0.5, (
        f"held {info['held_s']} s under ZERO action: that is a hold, not a "
        "decay, and the alpha=0.94 basin measurement needs revisiting")
    assert terminated, "having brushed the gate, the topple must end the episode"


# ------------------------------- the chain must survive a dirty teardown

def test_a_stage_that_finished_its_work_is_not_treated_as_failed(tmp_path):
    """A 2.7 M-step chain was abandoned at stage 2 of 8 because the finished
    stage crashed WHILE EXITING:

        libc++abi: terminating due to uncaught exception of type
        std::__1::system_error: recursive_mutex lock failed

    A MuJoCo/threading teardown race on macOS, with model.zip, vecnormalize.pkl
    and the terminal `{"done": true}` record all already on disk. "Did the
    process exit cleanly" and "did the work finish" are different questions,
    and the terminal record exists to answer the second one from the ARTEFACT.
    """
    import json as _json

    from microduck_local.viz_server import TrainingJob

    job = TrainingJob.__new__(TrainingJob)          # no lab, no subprocess
    job.dir = tmp_path

    assert job._wrote_terminal_record() is False, "an empty dir cannot be done"

    (tmp_path / "progress.jsonl").write_text(
        _json.dumps({"steps": 100, "total": 400}) + "\n")
    assert job._wrote_terminal_record() is False, "mid-run is not done"

    with open(tmp_path / "progress.jsonl", "a") as f:
        f.write(_json.dumps({"steps": 400, "total": 400, "done": True}) + "\n")
    assert job._wrote_terminal_record() is True


def test_a_torn_progress_line_does_not_crash_the_check(tmp_path):
    """progress.jsonl is appended to by a process that may die mid-write."""
    from microduck_local.viz_server import TrainingJob

    job = TrainingJob.__new__(TrainingJob)
    job.dir = tmp_path
    (tmp_path / "progress.jsonl").write_text(
        '{"steps": 1, "total": 2}\n{"steps": 2, "tot')
    assert job._wrote_terminal_record() is False


def test_the_lab_previews_a_getup_policy_under_its_own_recorded_flags(tmp_path):
    """A get-up policy shown in the lab must start where it was TRAINED to.

    Only `clip_name` and `max_episode_s` used to reach the preview env, so the
    lab built every Asimov get-up at the env's DEFAULT start — the `pike`,
    where these policies score 0/12 — while the policy had been trained to
    stand from `fold`, where it scores 12/12 and holds 7.76 s. Loading the
    stages one by one showed nothing working; they worked, and the lab was
    starting them in a pose they had never been trained for.
    """
    import json as _json

    from microduck_local.viz_server import env_kwargs_for_task_run

    run = tmp_path / "teach-asimov_getup_ref-probe-s1"
    run.mkdir()
    (run / "run.json").write_text(_json.dumps({
        "robot": "asimov1", "task": "getup_ref",
        "env_kwargs": {"domain_rand": True, "obs_noise": True,
                       "start_phase": "fold", "getup_label": "prone",
                       "track_tol": 0.6, "force_lag": True, "free": False,
                       "history": 1},
    }))
    kw = env_kwargs_for_task_run(run)
    assert kw["task"] == "getup_ref"
    assert kw["start_phase"] == "fold", kw
    assert kw["force_lag"] is True, kw
    assert kw["getup_label"] == "prone", kw
    # training-only knobs stay OUT: a preview shows the clean motion
    assert "domain_rand" not in kw and "obs_noise" not in kw, kw


def test_every_replay_consumer_shares_one_definition_of_the_flags(tmp_path):
    """The lab's preview and `render-rollout` each had this bug, separately.

    A consumer that rebuilds an env to replay a policy must ask the RUN what it
    trained under. Two of them did not, and each would have been fixed alone —
    so the list lives in `run_record.REPLAY_KWARGS` and both read it.
    """
    import json as _json

    from microduck_local.run_record import REPLAY_KWARGS, replay_kwargs
    from microduck_local.viz_server import env_kwargs_for_task_run

    run = tmp_path / "teach-asimov_getup_ref-probe-s1"
    run.mkdir()
    (run / "run.json").write_text(_json.dumps({
        "robot": "asimov1", "task": "getup_ref",
        "env_kwargs": {"domain_rand": True, "obs_noise": True,
                       "start_phase": "fold", "force_lag": True,
                       "track_tol": 0.6, "history": 1},
    }))
    shared = replay_kwargs(run / "policy.onnx")   # a path INSIDE the run
    lab = env_kwargs_for_task_run(run)
    for key in ("start_phase", "force_lag", "track_tol"):
        assert shared[key] == lab[key] == {"start_phase": "fold",
                                           "force_lag": True,
                                           "track_tol": 0.6}[key]
    # training-only knobs are in NEITHER — a replay shows the clean motion
    for only_training in ("domain_rand", "obs_noise", "random_yaw"):
        assert only_training not in REPLAY_KWARGS
        assert only_training not in shared and only_training not in lab


def test_eval_walk_refuses_a_policy_that_is_not_a_walker(tmp_path):
    """Scoring a get-up in the WALKING env prints a gait's numbers off a
    different robot. The kind check one level up only catches a WHEELED body;
    a legged body with a non-walk policy sailed straight past it."""
    import json as _json
    import subprocess
    import sys

    run = tmp_path / "fake-getup"
    run.mkdir()
    (run / "run.json").write_text(_json.dumps(
        {"robot": "asimov1", "task": "getup_ref", "env_kwargs": {}}))
    (run / "policy.onnx").write_bytes(b"not really an onnx")
    out = subprocess.run(
        [sys.executable, "-m", "microduck_local.eval_onnx",
         str(run / "policy.onnx")],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "getup_ref" in (out.stdout + out.stderr)
    assert "GAIT" in (out.stdout + out.stderr)


def test_a_fine_tunes_own_env_knobs_are_not_overridden_by_the_last_stage():
    """`TeachRequest.env` is documented as the escape hatch for the one case a
    curriculum cannot express — "an `initFrom` fine-tune has no stage, so it
    inherits the LAB's own environment" — and for any behaviour that HAS a
    curriculum it was silently dropped, because a single run still borrows
    `curriculum[-1].env` as `_stage_env` and the merge put the stage last.

    MEASURED 2026-10-04: a fine-tune launched to attack the one uncrossed leg
    with `{"MICRODUCK_ASIMOV_GETUP_START": "pike,quadruped"}` recorded
    `start_phase=fold,rock_back,pike,quadruped,draw_knees_in,plant_forearms,
    prone` instead and trained the wrong experiment. `run.json` is what caught
    it, which is the whole reason a run records what it trained under.

    A STAGED chain keeps the opposite order: the stage is the recipe.
    """
    from microduck_local.viz_server import TrainingJob

    job = TrainingJob.__new__(TrainingJob)      # no lab, no subprocess
    job.extra_env = {"MICRODUCK_ASIMOV_GETUP_START": "pike,quadruped"}
    job._stage_env = {"MICRODUCK_ASIMOV_GETUP_START": "fold,rock_back,pike",
                      "MICRODUCK_ASIMOV_GETUP_LAG": "1"}

    job.stages = ()                              # a single-run initFrom job
    assert job._job_env()["MICRODUCK_ASIMOV_GETUP_START"] == "pike,quadruped", (
        "the request's own knob was overridden by the recipe's last stage")
    # the stage's other knobs still apply — this is an override, not a reset
    assert job._job_env()["MICRODUCK_ASIMOV_GETUP_LAG"] == "1"

    job.stages = ("a", "b")                      # a staged chain
    assert job._job_env()["MICRODUCK_ASIMOV_GETUP_START"] == "fold,rock_back,pike", (
        "a request must not be able to redefine a rung of a staged curriculum")


def test_the_pose_cache_can_serve_every_fallen_posture():
    """`load_cache` matched ONE label exactly, so training drew 240 of the 477
    poses — `supine` only, which this repo's own ledger calls the hardest single
    posture ("it cannot rise from supine at all").

    The one policy known to get up on this model
    (openhorizon-labs/asimov1-getup-safefall: 99.8 %, 1.70 s median, trained on
    Menlo's STOCK MJCF and cross-checked in CPU MuJoCo) covers "back, front,
    side, sitting, kneeling" — which is exactly the six labels this cache
    already holds. Training on a sixth of the distribution and concluding the
    task is impossible is not a measurement of the task.
    """
    from microduck_local.robots import asimov
    from microduck_local.robots import asimov_getup as G

    if not asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    one = G.load_cache("supine")
    both = G.load_cache("supine,prone")
    every = G.load_cache("all")
    assert len(one) < len(both) < len(every), (len(one), len(both), len(every))
    # The SIX POSTURES, not the total: `fall_ladder` appends `fall_0..fall_7`
    # rungs to the same cache, so a bare total is a number that any later
    # recording changes — it went 477 -> 485 and failed this test for a reason
    # that was not a regression. What must not change is that each posture is
    # still served and still the size it was measured at.
    POSTURES = {"supine": 240, "prone": 220, "side_left": 5,
                "side_right": 3, "sitting": 5, "kneeling": 4}
    for name, n in POSTURES.items():
        assert len(G.load_cache(name)) == n, (
            f"{name} holds {len(G.load_cache(name))} poses, not {n}")
    assert len(every) >= sum(POSTURES.values()), (
        f"the cache LOST poses: {len(every)} < {sum(POSTURES.values())}")
    # a bare name still means exactly that label — the old behaviour
    assert len(one) + len(G.load_cache("prone")) == len(both)
    # and an unknown posture is refused by NAME rather than silently empty
    with pytest.raises(RuntimeError, match="nosuchpose"):
        G.load_cache("supine,nosuchpose")


def test_the_penalties_cannot_outweigh_the_task_at_initialisation():
    """`teach-asimov_getup-4c5c79` ran 14.7 M steps with the penalties at full
    weight from step one, and its own reward log says what that bought:

    | steps | task pays | penalties |
    |---|---|---|
    | 43 k | 11.0 | **-62.4** (5.7x) |
    | 12.0 M | 41.9 | -13.8 (0.3x) |

    The steepest early gradient was "stop moving", so it learned that —
    action-rate -30.0 -> -4.4, arm effort -32.4 -> -9.4, episode reward -29 ->
    +28 almost entirely by shedding penalty. `standing` paid 0.0 for all 14.7 M
    steps and the policy got up 0/24, the same as its donor.

    So the penalties ramp. This pins the property that matters — at step 0 they
    cannot outrank the task — rather than the stage table's exact numbers.
    """
    # NO os.environ here: `getup_label` is passed as a kwarg, and a
    # `setdefault` leaks for the whole pytest session — it made
    # `test_the_prone_rung_needs_a_looser_cut_than_the_default` fail in the
    # suite and pass alone, which is a test bug wearing a product bug's hat.
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0,
        getup_blend=0.0, getup_assist=0.0, getup_label="all")

    assert env._penalty_scale() == pytest.approx(0.05), (
        "a fresh env must start with the penalties almost off")
    # the MEASURED init budget: 11.0 of task reward against 62.4 of penalty
    assert 62.4 * env._penalty_scale() < 0.5 * 11.0, (
        "at the measured init budget the penalties still outrank the task — "
        f"{62.4 * env._penalty_scale():.1f} against 11.0")
    # ...and they must reach FULL weight, or this is a deletion not a ramp
    env._lifetime_steps = 10 ** 7
    assert env._penalty_scale() == pytest.approx(1.0), (
        "the penalties never reach full weight — that is removing them, and "
        "they exist to stop twitchy commands and arm strain")
    # monotone, so no stage can lower the pressure as training proceeds
    env2 = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0,
        getup_blend=0.0, getup_assist=0.0, getup_label="all")
    seen = []
    for n in (0, 10_000, 20_000, 40_000, 70_000, 100_000, 10 ** 6):
        env2._lifetime_steps = n
        seen.append(env2._penalty_scale())
    assert seen == sorted(seen), f"the ramp is not monotone: {seen}"


def test_a_tasks_action_clip_reaches_the_STEP_and_the_ACTION_SPACE():
    """A per-env clip that reaches only the reset pose reads as "no effect".

    There are THREE clip sites and they have to be one number: the action space
    (SB3 clips to it in `collect_rollouts`, so `step` never sees a wider action
    during training), `walk_env.step`, and the reset pose. The `getup_clip`
    override landed on the third only, while `step` kept clipping at
    `self.robot.action_clip` — so the 13.0-vs-4.0 A/B came back BYTE-IDENTICAL
    and was written up as "the clip is not the cause". It was never applied.

    The teacher this task clones exceeds 4.0 on 15.5 % of its frames and peaks
    at 11.5, so a 4.0 clip makes it inimitable by construction.
    """
    import numpy as np

    from microduck_local.robots import spec as S

    body = S.get("asimov1")
    # `action_delay=False`: the per-episode 0/1-step lag is drawn independently
    # of `domain_rand`, and with it on, step 1 applies the PREVIOUS action — so
    # ctrl reads DEFAULT_POSE and this test fails for a reason that has nothing
    # to do with the clip. It cost one debugging round; it is the whole reason
    # the number below is read off ctrl and not asserted from the signature.
    tight = body.env_class("getup")(domain_rand=False, obs_noise=False,
                                    action_delay=False, seed=0)
    wide = body.env_class("getup_clip")(domain_rand=False, obs_noise=False,
                                        action_delay=False, seed=0)
    assert wide._act_clip > tight._act_clip, "getup_clip must widen the clip"

    for env in (tight, wide):
        c = env._act_clip
        # The space TRACKS the clip at a fixed multiple — it is deliberately
        # wider, because SB3 clips to the space before `step` runs and an env
        # that cannot see past its own clip cannot price a runaway (see
        # `test_the_env_can_SEE_an_action_past_the_clip`). What must hold is
        # that widening the clip widens the space with it.
        from microduck_local.walk_env import ACTION_SPACE_SLACK
        assert float(env.action_space.high[0]) == pytest.approx(
            c * ACTION_SPACE_SLACK), (
            f"{type(env).__name__} action space is {env.action_space.high[0]} "
            f"but its clip is {c} x slack {ACTION_SPACE_SLACK}")
        # What `step` APPLIES is the clipped action: drive one joint past the
        # clip and read the position target back off mjData.ctrl.
        env.reset(seed=0)
        a = np.zeros(env.nj, np.float32)
        a[0] = c * 3.0
        env.step(a)
        want = float(env.robot.scale_action(np.clip(a, -c, c))[0])
        got = float(env.data.ctrl[0] if env._ctrl_whole
                    else env.data.ctrl[env.ctrl_adr[0]])
        scale = np.atleast_1d(env.robot.action_scale)[0]
        applied = (got - float(np.atleast_1d(env.robot.default_pose)[0])) / float(scale)
        assert abs(got - want) < 1e-6, (
            f"{type(env).__name__} applied a target for clip {applied:.2f}, "
            f"not {c} — the clip did not reach step()")


def test_lying_still_cannot_out_earn_getting_up():
    """The imitation reward must not pay a fallen robot for straightening up.

    MEASURED, before the root was priced: a zero-action policy — and action 0 is
    `DEFAULT_POSE`, which is what an untrained network emits — earned **+7.409 of
    `track` per step against the riser's +6.067**, because the error was
    joint-space only and a robot lying on the floor with straight legs matches a
    STANDING reference joint for joint. It never stood, so the `_stood`-armed
    fall cut never fired and it banked the full episode: **2775 a episode
    against the clone's 788**. A 2 M-step warm-started fine-tune converged to
    `ep_rew` 2942 — the do-nothing return — while its rise fell from 24/24 to
    15/24.

    Same fault as the authored reference's ("same joints, different body"):
    contact mode and root pose set the posture, and joints alone cannot see it.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov
    from microduck_local.robots import spec as S

    if not asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    env = S.get("asimov1").env_class("getup_clip")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0)
    d = env.data

    # ON the demonstration, at a frame where it is upright: near-maximal pay.
    env.reset(seed=0)
    t = len(env._clip) - 1
    env._ep_steps = t
    d.qpos[:] = env._clip[t]
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)
    _r, on_ref = env._compute_reward()
    assert on_ref["track"] > 0.8 * env.W_TRACK, (
        f"being exactly ON the demonstration pays only {on_ref['track']:.2f} "
        f"of {env.W_TRACK}")

    # THE FARM: the demonstration's standing JOINTS, but still on the floor.
    env.reset(seed=0)
    env._ep_steps = t
    d.qpos[:] = env._clip_starts[0]          # lying where the episode begins
    d.qpos[env._q_adr] = env._clip[t][env._q_adr]   # ...with STANDING joints
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)
    _r, farmed = env._compute_reward()
    assert farmed["track"] < 0.25 * on_ref["track"], (
        f"a robot on the floor with the reference's joints collects "
        f"{farmed['track']:.3f} against {on_ref['track']:.3f} for actually "
        "being there — the root is not priced")

    # And over an episode, doing nothing must not out-earn the demonstration.
    def total(mode: str, steps: int = 150) -> float:
        env.reset(seed=0)
        d.qpos[:] = env._clip_starts[0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(env.model, d)
        tot = 0.0
        for k in range(steps):
            if mode == "replay":                 # follow the demonstration
                tgt = env._clip[min(k, len(env._clip) - 1)][env._q_adr]
                a = (np.asarray(tgt) - env.robot.default_pose) / env.robot.action_scale
                a = np.clip(a, -env._act_clip, env._act_clip).astype(np.float32)
            else:
                a = np.zeros(env.nj, np.float32)
            _o, r, term, trunc, _i = env.step(a)
            tot += float(r)
            if term or trunc:
                break
        return tot

    idle, replay = total("zero"), total("replay")
    assert idle < replay, (
        f"doing nothing earns {idle:.1f} and following the demonstration "
        f"{replay:.1f} — the reward is paying for the floor")


def test_the_arm_reward_cannot_be_farmed_by_flailing():
    """`arm_push` pays for pushing off the floor, not for thrashing.

    WHY THE TERM EXISTS. This body cannot get up on its legs alone and the one
    get-up known to work on it does not try. Measured over its demonstration, by
    height band (fraction of standing):

        0.00-0.20  arm load 0.556   trunk dz/dt +0.14
        0.20-0.35  arm load 0.526   trunk dz/dt +0.15
        0.35-0.50  arm load 0.594   trunk dz/dt +1.01   <- the launch
        0.50-0.70  arm load 0.037   trunk dz/dt +0.79
        0.70+      arm load 0.000   trunk dz/dt +0.81

    The arms carry over half the body weight from the floor to half standing
    height, do the launch, and release. This env previously only ever CHARGED
    for that (`arm_effort_penalty`) and never paid for it.

    WHY IT IS SHAPED THE WAY IT IS. Two earlier versions were priced and
    rejected on measurement:

      `gain x arm_load`          0.20 a episode against a 356 return — correct
                                 in shape, far too weak to steer anything,
                                 because most height is won on the LEGS after
                                 the arms release.
      `up_speed x arm_load`      a real signal (57) and FARMABLE: bouncing
                                 scored 32.5 and RANDOM ACTIONS 35.4, i.e. 60 %
                                 of a real get-up's pay for flailing.

    The shipped version counts only motion that takes the trunk ABOVE its own
    best height this episode, which oscillation cannot farm.
    """
    import numpy as np

    from microduck_local.robots import asimov
    from microduck_local.robots import spec as S

    if not asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    def earned(act, eps=4):
        env = S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=False, action_delay=False, seed=0,
            arm_push=4.0, arm_effort=0.0, action_clip=13.0, getup_label="all")
        tot = 0.0
        for ep in range(eps):
            env.reset(seed=ep)
            env.reward_sums.clear()
            for k in range(int(env.max_steps)):
                _o, _r, term, trunc, _i = env.step(act(env, k))
                if term or trunc:
                    break
            tot += float(env.reward_sums.get("arm_push", 0.0))
        return tot / eps

    still = earned(lambda e, k: np.zeros(e.nj, np.float32))
    press = earned(lambda e, k: np.where(
        np.isin(np.arange(e.nj), e._arm_ids), -3.0, 0.0).astype(np.float32))
    bounce = earned(lambda e, k: (np.sin(k * 0.6) * 3.0
                                  * np.ones(e.nj)).astype(np.float32))
    rng = np.random.default_rng(0)
    flail = earned(lambda e, k: rng.normal(0, 2.0, e.nj).astype(np.float32))

    floor = max(still, press, bounce, flail)
    assert floor < 6.0, (
        f"a cheat earns {floor:.2f} of arm_push: still {still:.2f}, press "
        f"{press:.2f}, bounce {bounce:.2f}, flail {flail:.2f}")
    # Pressing an arm down while going nowhere must pay essentially nothing —
    # load alone is not the thing being bought.
    assert press < 2.5 * still, (
        f"pressing the arms down earns {press:.2f} against {still:.2f} for "
        "doing nothing — load is being paid for without any rise")


def test_the_pose_cache_really_is_supine_and_prone_not_just_labelled_so():
    """A `supine` pose must have its CHEST pointing up, measured.

    THE MISTAKE THIS EXISTS FOR. Asked whether the cache actually held face-up
    and face-down poses, I classified all 485 of them with `xmat[2][2]` — the
    torso's own Z axis, which is UPRIGHTNESS. For any robot lying down that is
    ~0 whichever way it faces, so the test called every pose "side-lying" and I
    reported, as the headline finding of a session, that 96 % of the cache was
    mislabelled and the labels only recorded how each pose had been DROPPED.

    The cache was correct the whole time. Supine/prone lives in `xmat[2][0]`,
    the chest normal, which is exactly what Menlo's own `label_fallen_states`
    uses — `argmax([f_z, -f_z, -l_z, l_z])` over `LABELS`. Measured properly:
    supine f_z **+0.99**, prone **-0.96**, side_left/right l_z **-+0.86**. The
    same wrong index also had me tell Jonathan the distilled clone had "never
    been face-up" when its 24 demonstrations are 14 supine and 10 prone.

    Same class as the velocity-frame trap already in this repo's notes: the
    "forward" index was sideways. An orientation claim needs the axis named.
    """
    import collections

    import mujoco
    import numpy as np

    from microduck_local.robots import asimov
    from microduck_local.robots import asimov_getup as G
    from microduck_local.robots import spec as S

    if not asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0)
    m, d = env.model, env.data
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, G.TORSO_BODY)
    z = np.load(G.cache_path(None), allow_pickle=True)

    axes = collections.defaultdict(list)
    for q, lab in zip(z["qpos"], z["label"]):
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        rot = d.xmat[tid].reshape(3, 3)
        axes[str(lab)].append((rot[2, 0], rot[2, 1]))   # f_z (chest), l_z (left)

    # The chest normal, not the long axis. These are the numbers measured.
    sup = np.array(axes["supine"])
    pro = np.array(axes["prone"])
    assert len(sup) > 100 and len(pro) > 100, (
        f"the cache must hold both postures in bulk: supine {len(sup)}, "
        f"prone {len(pro)}")
    assert sup[:, 0].mean() > 0.8, (
        f"'supine' poses average f_z {sup[:, 0].mean():+.2f} — chest is not up")
    assert pro[:, 0].mean() < -0.8, (
        f"'prone' poses average f_z {pro[:, 0].mean():+.2f} — chest is not down")
    # and they are genuinely distinct, not one blob either side of zero
    assert sup[:, 0].min() > 0.0 and pro[:, 0].max() < 0.0

    for side, sign in (("side_left", -1.0), ("side_right", +1.0)):
        a = np.array(axes.get(side, []))
        if len(a):
            assert sign * a[:, 1].mean() > 0.5, (
                f"'{side}' poses average l_z {a[:, 1].mean():+.2f}")


def test_the_env_can_SEE_an_action_past_the_clip():
    """An out-of-clip penalty is a dead hook unless the SPACE is wider.

    SB3 clips every action to the action space before `step` runs
    (`on_policy_algorithm.collect_rollouts` line ~216: `clipped_actions =
    np.clip(actions, low, high)`, then `env.step(clipped_actions)`). With a
    space equal to the clip, `last_action` is the CLIPPED value during training
    whatever the network emitted — so `walk_env.step`'s documented reason for
    storing the raw output ("unbounded output growth is not free") describes a
    mechanism that cannot operate under this trainer.

    MEASURED: a 6 M-step run's exported deterministic policy reached a per-step
    max output of **2055** while `over_clip_penalty` recorded **0.0** for the
    whole run, and the A/B arm carrying that penalty produced a BYTE-IDENTICAL
    policy to the arm without it — the signature of a dead hook, not of a
    variable that does not matter.
    """
    import numpy as np

    from microduck_local.robots import asimov
    from microduck_local.robots import spec as S
    from microduck_local.walk_env import ACTION_SPACE_SLACK

    if not asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0,
        action_clip=13.0)
    assert ACTION_SPACE_SLACK > 1.0
    assert float(env.action_space.high[0]) > env._act_clip, (
        "the action space must be WIDER than the applied clip, or SB3 clips "
        "the runaway away before the env can price it")

    env.reset(seed=0)
    runaway = float(env._act_clip) * 3.0
    assert runaway <= float(env.action_space.high[0]), "space too narrow to test"
    env.step(np.full(env.nj, runaway, np.float32))
    _r, terms = env._compute_reward()

    assert float(np.abs(env.last_action).max()) == pytest.approx(runaway), (
        "the env did not see the raw action")
    assert terms["over_clip_penalty"] < -1.0, (
        f"over_clip_penalty is {terms['over_clip_penalty']:.3f} for an action "
        f"{runaway / env._act_clip:.0f}x the clip — the hook is dead")
    # ...and what is APPLIED is still inside the clip. Seeing is not obeying.
    scale = float(np.atleast_1d(env.robot.action_scale)[0])
    applied = (env.data.ctrl - np.asarray(env.robot.default_pose)) / scale
    assert float(np.abs(applied).max()) <= env._act_clip + 1e-6, (
        "a wider SPACE must not widen what reaches the actuators")


def test_the_verifier_refuses_a_run_that_recorded_no_flags(tmp_path):
    """`verify_run` must not guess the conditions a policy trained under.

    Evaluating under the wrong flags does not look like an error, it looks like
    a RESULT — this repo's own ledger has three instances, including 0/12 picks
    reported for a policy that scored 10/12 under its real flags. So the
    verifier builds its env from `replay_kwargs(run)` and exits rather than
    fall back to defaults.
    """
    import json
    import subprocess
    import sys

    run = tmp_path / "runs" / "nothing-recorded"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "run_name": "nothing-recorded", "robot": "asimov1"}))
    (run / "model.zip").write_bytes(b"not a model")
    (run / "vecnormalize.pkl").write_bytes(b"nor this")

    r = subprocess.run(
        [sys.executable, "scripts/verify_run.py", str(run), "--episodes", "2"],
        capture_output=True, text=True, cwd=str(_P(__file__).parents[1]))
    assert r.returncode != 0, "a run with no recorded flags must not be scored"
    assert "recorded no env_kwargs" in (r.stdout + r.stderr), (
        f"the refusal must say WHY:\n{r.stdout}\n{r.stderr}")


def test_the_verifier_reports_rise_and_hold_separately():
    """One number cannot say "it gets up" and "it stays up", and this task's
    whole history is the difference: a clone rose 24/24 and held 4/24, and a
    chain whose rise fell 24/24 -> 15/24 had every training number improving.
    So the verifier's row carries both, plus the zero-action baseline each has
    to beat and the resolution of the episode count it used.
    """
    src = (_P(__file__).parents[1] / "scripts" / "verify_run.py").read_text()
    for key in ('"rose"', '"at_buzzer"', '"hold_median_s"',
                '"fell_after_standing"'):
        assert key in src, f"the verifier must report {key}"
    # the three things it has to SAY, not merely compute
    assert "must beat this" in src, "it must print the do-nothing baseline"
    assert "resolution:" in src, "it must print what the episode count resolves"
    assert "NOT THE LAST ONE" in src, "it must flag best-is-not-last"
    # ...and it must not quietly reuse another run's export
    assert "run.name" in src and "onnx.unlink()" in src, (
        "exports must be keyed by RUN and never reused — a step-keyed scratch "
        "directory reported one run's numbers for another's")


def test_the_watcher_verifies_every_finished_run():
    """The train loop must not hand back a one-number report and nothing else.

    `watch_training.py` exported the finished run and ran a task report on
    `policy.onnx` — the FINAL model — which on 2026-10-05/06 was wrong four
    ways on one task: a training curve read as a result, a final checkpoint
    scored when an earlier one was twice as good, a stale export scored as
    fresh, and a policy scored under flags it never trained with. So the
    watcher runs `verify_run.py` first, for every task.
    """
    import ast

    path = _P(__file__).parents[1] / "scripts" / "watch_training.py"
    src = path.read_text()
    assert "verify_run.py" in src, (
        "the watcher must verify a finished run, not just report on it")

    # By the CODE, not the prose: the module docstring names pick_report.py
    # long before any code does, so a plain string search says the wrong thing.
    tree = ast.parse(src)
    order = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node.value.endswith(".py") and "report" in node.value:
            order.append((node.lineno, "report"))
        elif node.value == "verify_run.py":
            order.append((node.lineno, "verify"))
    body = ast.parse(src)
    fn = next(n for n in body.body
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    inside = [(ln, k) for ln, k in order if fn.lineno <= ln <= (fn.end_lineno or ln)]
    assert inside, "neither the verifier nor a report is called from main()"
    assert inside[0][1] == "verify", (
        f"main() reaches a report before the verifier: {inside[:3]}")


def _verify_run_module():
    """`scripts/` is not a package, so import the verifier by path."""
    import importlib.util

    path = _P(__file__).parents[1] / "scripts" / "verify_run.py"
    spec = importlib.util.spec_from_file_location("_verify_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_verifier_prices_doing_nothing_on_THE_DELIVERABLE():
    """Guard 6 was applied to the training env and not to the headline number.

    `verify_run.py` priced the zero-action policy in the env a run RECORDED,
    then printed a separate "THE DELIVERABLE" block — the floor-only score —
    with no baseline at all. On 2026-10-06 that mattered: in the training env
    (`start_rung='mix'`, which can begin an episode already on a plateau)
    doing nothing "rose" 13 of 36, while on the floor it rose **0 of 36** and
    peaked at 0.134 m. A reader comparing the deliverable's 20/36 against the
    13/36 printed above it would have concluded the rise was nearly worthless;
    they are different episode sets, which is the marginal-vs-paired error one
    level up. So the deliverable is priced against zero action on ITS OWN
    episodes, and `_score_with` grew the `onnx=None` path to do it.
    """
    import inspect

    V = _verify_run_module()

    # the zero-action path must exist on the function the deliverable uses
    sw = inspect.getsource(V._score_with)
    assert "if sess is None" in sw, (
        "_score_with must score ZERO ACTION (onnx=None) — without it the "
        "deliverable is the one number never priced against doing nothing"
    )
    assert "onnx: Path | None" in sw, "its signature must admit None"

    src = (_P(__file__).parents[1] / "scripts" / "verify_run.py").read_text()
    deliverable = src.split("THE DELIVERABLE", 1)[1]
    assert "_score_with(None, run, kw" in deliverable, (
        "the DELIVERABLE block must call _score_with with onnx=None on the "
        "same kwargs and episodes as the policy"
    )
    assert "SAME episodes" in deliverable, (
        "it must say the baseline is on the same episodes, so nobody pairs it "
        "against the training env's"
    )


def test_the_resolution_GUARD_is_in_the_resolution_s_OWN_UNITS():
    """A dead guard reads exactly like a passing one.

    `_mde` returns a PROPORTION (0.23 at n=36), which its first call site
    assumes. The rise-vs-zero-action guard added beside it computed its gap in
    PERCENTAGE POINTS and compared the two, so it tested `56 < 0.23` — false
    for every possible gap, including a gap of zero. It printed
    "resolution 0" and passed, which is what a working guard also looks like.
    Both sides of every resolution comparison are proportions; the x100 is
    display only.
    """
    import re

    V = _verify_run_module()

    assert 0.0 < V._mde(36) < 1.0, "_mde returns a proportion, not points"
    src = (_P(__file__).parents[1] / "scripts" / "verify_run.py").read_text()
    # every `gap` compared against an mde must be a proportion: no 100.0 *
    for line in src.splitlines():
        if re.match(r"\s*gap = ", line):
            assert "100" not in line, (
                f"a gap compared against _mde must be a proportion, not "
                f"points: {line.strip()!r}"
            )


def test_the_clip_reward_charges_a_SAG_and_not_a_lead():
    """The hold failure, priced.

    59 of 72 episodes of the best get-up policy stand and then go down, and the
    going-down is a slow SAG, not a topple: the trunk drifts 0.613 -> 0.452 m
    while uprightness stays above +0.95, the commanded knee climbs 0.383 ->
    0.645 rad, and peak torque is 12.8 Nm against 45 rated. The policy commands
    the sag. What let it is the reward:

      * `up` is a BOOLEAN times 8.0 — a cliff that pays nothing until the robot
        is already below the standing threshold, by which point it is too late
        to be a gradient.
      * `track`'s height factor used `ROOT_Z_STD = 0.10`, so a 2 cm sag cost
        18 %, and it was `abs()` — two-sided — so a robot AHEAD of the
        demonstration was charged exactly as much as one sagging below it.

    This pins both halves of the fix: a sag is charged, and a lead is not.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov as _asimov

    if not _asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup_clip")(
        domain_rand=False, obs_noise=False, seed=0)
    env.reset(seed=0)
    ref = env._clip[env._clip_t()]

    def root_err_at(z):
        env.data.qpos[:] = ref
        env.data.qpos[2] = z
        mujoco.mj_forward(env.model, env.data)
        return env._root_err()[0]

    z_ref = float(ref[2])
    # a SAG is charged, and by more than it used to be
    sag = root_err_at(z_ref - 0.02)
    assert sag == pytest.approx(0.02, abs=1e-6), f"a 2 cm sag reads {sag:.4f}"
    factor = float(np.exp(-sag / env.ROOT_Z_STD))
    assert factor < 0.75, (
        f"a 2 cm sag only costs the track term {1-factor:.0%}; at "
        "ROOT_Z_STD=0.10 it cost 18 % and the policy sagged straight through it")

    # a LEAD is NOT charged — AGENTS.md: overshooting toward the goal cannot be
    # wrong on a rise
    lead = root_err_at(z_ref + 0.05)
    assert lead == 0.0, (
        f"being 5 cm ABOVE the demonstration is charged {lead:.4f}; the body "
        "terms must be one-sided")

    # and the two must not be symmetric, which is the whole point
    assert root_err_at(z_ref - 0.05) > root_err_at(z_ref + 0.05)


def test_the_span_cut_forbids_the_splits_but_admits_every_cached_rung():
    """The sprawl cut, sized against the ladder it has to let through.

    teach-asimov_getup_clip-79ef4d rises 288/288 and does it by sprawling: a
    MEDIAN foot-to-foot span of 0.989 m and up to 1.042 m, on a 1.2 m robot
    whose standing stance is 0.215 m. That is the splits, and it is the route
    ruled out at the start of this track as dangerous on hardware.

    The threshold is not taste. The staged route has to pass through the
    splayed feet-only rungs, which sit at 0.59-0.74 m, so the cut must admit
    every pose in the cache and still forbid a 0.99 m scramble. This asserts
    both sides, so a future rung wider than the cut fails here rather than
    silently making the curriculum unreachable.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov as _asimov
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots.asimov_env import AsimovGetupEnv

    if not _asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    # the EPISODE cut. `MAX_FOOT_SPAN_M` filters the rung CACHE; the episode
    # cut must also admit the scripted ramp, which overshoots its own rungs.
    env = AsimovGetupEnv(domain_rand=False, obs_noise=False,
                         getup_blend=0.0, getup_assist=0.0,
                         max_span=R.EPISODE_SPAN_CUT_M)
    assert AsimovGetupEnv(domain_rand=False, obs_noise=False, getup_blend=0.0,
                          getup_assist=0.0)._max_span == 0.0, "must be OFF by default"
    env.reset(seed=0)

    # standing is nowhere near the cut
    env._place(np.array(W.DEFAULT_POSE, float), 0.0)
    mujoco.mj_forward(env.model, env.data)
    assert env._foot_span() == pytest.approx(0.215, abs=0.02)

    # EVERY cached rung must pass, or the cut makes the ladder unreachable
    z = R.holdable_rungs()
    spans = []
    for q in z["joints"]:
        env.data.qpos[:] = np.asarray(q)
        mujoco.mj_forward(env.model, env.data)
        spans.append(env._foot_span())
    spans = np.array(spans)
    # the env cuts on `> max_span`, so a rung exactly AT the cut is admissible
    assert spans.max() <= env._max_span + 1e-9, (
        f"the widest cached rung spans {spans.max():.3f} m against a cut at "
        f"{env._max_span:.2f} — the curriculum could not reach its own rungs")

    # ...and the sprawl it exists to forbid does NOT pass
    assert 0.989 > env._max_span, (
        "the measured median sprawl of the unconstrained policy (0.989 m) is "
        "inside the cut, so the cut forbids nothing")
    assert spans.max() > 0.5, (
        f"widest rung is only {spans.max():.3f} m — if the splayed rungs are "
        "gone, this test is no longer pinning the tension it was written for")


def test_every_knob_a_recipe_SETS_is_a_knob_the_run_RECORDS(no_knob_env,
                                                            monkeypatch):
    """The sixth guard against the same bug, and the first generic one.

    Five times now a stage has set a `MICRODUCK_ASIMOV_GETUP_*` variable that
    the env read straight out of the environment while `train_env_kwargs` —
    and so `run.json` — said nothing about it: `track_tol`, then `max_span`
    and `fall_depth`, then `launch_span`, then `start_rung` and the span pair
    again on `getup_clip`. Each time the fix was one task's branch and each
    time the next branch was missed, because the test named the knobs it knew.

    This one names none of them. It walks every registered recipe, sets each
    knob its own curriculum sets, and asserts the recorded dict MOVES. A knob
    that changes the training and not the record is the failure, whatever it
    is called and whichever task owns it.
    """
    from microduck_local.behaviors import asimov_tasks as T

    probes = {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "pike",
              "MICRODUCK_ASIMOV_GETUP_CLIPS": "/tmp/probe_clips.npz",
              "MICRODUCK_ASIMOV_GETUP_START": "quadruped",
              "MICRODUCK_ASIMOV_GETUP_POSE": "prone"}
    missed = []
    for name in dir(T):
        b = getattr(T, name, None)
        if getattr(b, "robot", None) != "asimov1" or not getattr(
                b, "curriculum", None):
            continue
        argv = list(b.trainer)
        task = argv[argv.index("--task") + 1] if "--task" in argv else "walk"
        args = types.SimpleNamespace(task=task)
        for stage in b.curriculum:
            for key in stage.env:
                if not key.startswith("MICRODUCK_ASIMOV_"):
                    continue
                base = W.spec().train_env_kwargs(args)
                moved = base
                # A value that cannot coincide with the default. Which KIND of
                # value a knob takes is the knob's business, so try each until
                # one parses rather than keeping a table of types here.
                for probe in (probes.get(key), "0.37", "5", "1", "pike"):
                    if probe is None:
                        continue
                    with monkeypatch.context() as mp:
                        mp.setenv(key, probe)
                        try:
                            moved = W.spec().train_env_kwargs(args)
                        except ValueError:
                            continue
                    if moved != base:
                        break
                if moved == base:
                    missed.append(f"{b.id} ({task}) sets {key}")
    assert not missed, (
        "these knobs train but are not recorded on the run:\n  "
        + "\n  ".join(sorted(set(missed))))


def test_the_span_charge_never_makes_lying_down_the_better_deal(no_knob_env):
    """A penalty that can eat its own task is the failure this body already
    has on record twice — the duck's action-rate ramp drove the G1 idle to
    `ep_rew` -92, and a span TERMINATION at 0.85 m taught this very policy to
    stop standing (`up` 0.0 from 98 k steps, return 35.5 against a do-nothing
    39.8). A charge is only an improvement while rising still beats the floor.

    Priced per EPISODE off the measured rollout, not per step off a worst
    case. The first version of this test charged the peak span on every one of
    the 400 steps, which is a robot that stands permanently splayed — nothing
    here does that. Measured on `teach-asimov_getup_rise-8cec81-s4`: income
    5700.7 an episode, the span is above the target for **38 steps of 400**,
    and the policy's settled stance is 0.366 m, BELOW the target, so standing
    itself costs nothing at all. The conservative version capped the weight at
    about 16, where the charge is 4 % of income — too small to move anything,
    which is exactly what the four stages measured.
    """
    from microduck_local.behaviors import asimov_tasks as T

    WIDEST = 1.026          # the donor's peak foot span
    WIDE_STEPS = 38         # of 400, measured on …-8cec81-s4
    EPISODE = 400
    INCOME = 5700.7         # track + up over an episode that rises
    FLOOR = 39.8            # zero action over the same episode
    stages = T.ASIMOV_GETUP_RISE.curriculum
    weights = [float(st.env["MICRODUCK_ASIMOV_GETUP_SPAN_COST"])
               for st in stages]
    targets = [float(st.env["MICRODUCK_ASIMOV_GETUP_SPAN_TARGET"])
               for st in stages]
    assert weights, "the rise curriculum charges nothing for width"
    for w, tgt in zip(weights, targets):
        charge = w * max(0.0, WIDEST - tgt) * WIDE_STEPS
        assert INCOME - charge > FLOOR, (
            f"at span_cost={w} a rise nets {INCOME - charge:.0f} an episode, "
            f"which does not beat lying still ({FLOOR})")
    # And the other half: a charge nothing can feel is not a curriculum. The
    # four stages at 3-16 cost 0.7-4.0 % of income and moved the span 0.989 ->
    # 0.991, which is no movement at all.
    worst = max(weights) * max(0.0, WIDEST - min(targets)) * WIDE_STEPS
    assert worst > 0.10 * INCOME, (
        f"the heaviest stage charges {100 * worst / INCOME:.1f} % of income — "
        "too small to move the policy, which is how the 3-16 ladder measured "
        "nothing")
    assert EPISODE >= WIDE_STEPS


def test_the_span_charge_is_one_sided_and_off_by_default(no_knob_env):
    """Off by default, so every run before 2026-10-08 reproduces; and
    one-sided, so a stance NARROWER than the target is not paid for being
    narrow — AGENTS.md's rule for body terms, and the reason the robot has no
    reason to bring its feet together past a human stance."""
    assert _env()._span_charge() == 0.0
    env = _env(span_cost=6.0, span_target=10.0)   # target far above any stance
    assert env._span_charge() == 0.0, "a narrow stance is being PAID"
    env = _env(span_cost=6.0, span_target=0.0)
    assert env._span_charge() > 0.0, "the charge never fires"


def test_a_start_rung_either_MOVES_the_start_or_is_refused(no_knob_env):
    """Recorded is not applied, and this is the gap the recording test missed.

    `test_every_knob_a_recipe_SETS_is_a_knob_the_run_RECORDS` proves a knob
    reaches `run.json`. It cannot prove the ENV uses it. `getup_clip` resets
    onto a demonstration frame and overwrote whatever `start_rung` placed, so
    three chains recorded `start_rung="pike"`, trained from clip frames at a
    median trunk height of 0.153 m, and were evaluated in an env built the
    same way — the knob was on the run, in the recipe, and in the write-up,
    and nowhere in the physics.

    So: for every task, a rung either MOVES the start or the env refuses it.
    Silence is the one outcome that is not allowed.
    """
    import numpy as np

    from microduck_local.robots import spec as S

    def start_z(task, **kw):
        env = S.get("asimov1").env_class(task)(
            domain_rand=False, obs_noise=False, seed=0, action_clip=13.0, **kw)
        zs = []
        for ep in range(4):
            env.reset(seed=ep)
            zs.append(float(env.data.xpos[env.trunk_body_id][2]))
        return float(np.median(zs))

    for task, extra in (("getup", {}), ("getup_clip", {"free": True})):
        base = start_z(task, **extra)
        for rung, kw in (("pike", {}), ("falling", {"fall_depth": 0.0})):
            try:
                moved = start_z(task, start_rung=rung, **extra, **kw)
            except ValueError:
                continue                      # refused outright: acceptable
            assert abs(moved - base) > 0.05, (
                f"{task} accepted start_rung={rung!r} and started in the same "
                f"place anyway ({moved:.3f} m vs {base:.3f} m) — the knob is "
                "recorded and discarded")


def _settled(name, clip=4.0, steps=120):
    """Place an authored waypoint and let it settle, as every probe does."""
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, action_clip=clip)
    m, d = env.model, env.data
    sc = float(np.atleast_1d(env.robot.action_scale)[0])
    DEF = np.asarray(env.robot.default_pose, float)
    w = R.BY_NAME[name]
    env.reset(seed=0)
    env._place(np.asarray(w.joints, float), w.root_pitch)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    a = np.clip((np.asarray(w.joints, float) - DEF) / sc,
                -clip, clip).astype(np.float32)
    for _ in range(steps):
        d.xfrc_applied[:] = 0.0
        env.step(a)
    return env


def test_the_START_pose_at_fold_is_not_a_stand(no_knob_env):
    """The fake that designed a whole curriculum.

    A `fold` episode BEGINS at 0.523 m and the height bar is
    0.86 x 0.608 = 0.523 m, so a peak-height test passes every episode before
    the robot has done anything. That is how "zero action rises 23/24 from
    fold" was measured, and the chain built on it learned exactly nothing.
    """
    from microduck_local.robots import asimov_getup_ref as R

    env = _settled("fold")
    rep = R.standing_report(env)
    bar = R.STAND_FRAC * R.BY_NAME["stand"].settled_z
    peak_says = rep["trunk_z"] >= bar - 0.01        # the criterion that lied
    assert peak_says, "fold no longer sits at the bar; re-tune this test"
    assert not rep["standing"], rep
    assert not R.is_standing(env)


def test_a_face_down_robot_is_not_a_stand_however_high_its_trunk(no_knob_env):
    """The other fake: a search maximising peak height threw the hips in the
    air and lay on the elbows — 0.615 m, uprightness -0.61, four contacts."""
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _settled("stand")
    assert R.is_standing(env), "the stand itself must pass"
    d = env.data
    # pitch the root right over, keeping the trunk high
    d.qpos[3:7] = np.array([0.0, 0.0, 1.0, 0.0])      # 180 deg about y
    mujoco.mj_forward(env.model, d)
    rep = R.standing_report(env)
    assert rep["upright"] < 0.0
    assert not rep["standing"], rep
    assert "upright" in rep["why"]


def test_knees_and_forearms_on_the_floor_are_not_feet(no_knob_env):
    """The contact clause, isolated. This body holds a quadruped on knees and
    forearms and an arm plateau at 0.59 m; neither is standing."""
    from microduck_local.robots import asimov_getup_ref as R

    quad = R.standing_report(_settled("quadruped"))
    assert not quad["only_feet"], quad
    assert any("not only feet" in quad["why"] for _ in (0,)), quad
    stand = R.standing_report(_settled("stand"))
    assert stand["only_feet"] and stand["on_feet"], stand


def test_every_get_up_scorer_asks_the_SAME_question(no_knob_env):
    """Ten scripts had their own copy of the test and they disagreed — some
    asked about the PEAK. One definition, or the next one drifts too."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for f in sorted((root / "scripts").glob("*.py")):
        src = f.read_text()
        # ANY height-fraction comparison, even in a file that also calls the
        # helper: the first version of this test let one through because the
        # file imported `standing_report` elsewhere, and a private copy
        # sitting beside the real test is exactly how the two drifted apart.
        if re.search(r"0\.[5-9][0-9]?\s*\*\s*[\w.\[\]\"]*"
                     r"(SZ|stand_z|settled_z)", src):
            offenders.append(f.name)
    assert not offenders, (
        "these re-implement the stand test instead of calling "
        "asimov_getup_ref.is_standing / standing_report: " + ", ".join(offenders))


def test_a_pose_past_a_JOINT_STOP_is_refused_not_simulated(no_knob_env):
    """`qpos` is not clamped on assignment, so a pose past a stop is accepted
    in silence and the constraint solver fights it for the whole episode —
    which looks exactly like an unstable pose and is not one.

    On 2026-10-09 a kneel search ran ~26 000 samples with the knee between
    -1.2 and -2.0 rad. That joint's range is +0.00 to +1.50 and it bends ONE
    WAY; every pose was illegal and the toppling was the solver ejecting them.
    """
    import numpy as np
    import pytest

    from microduck_local.robots import asimov_getup_ref as R

    env = _env()
    # the authored route is legal and must keep placing
    for w in R.WAYPOINTS:
        env._place(np.asarray(w.joints, float), w.root_pitch)
    with pytest.raises(ValueError, match="joint limits"):
        env._place(R.joints(**{"left_knee_joint": -1.6}), 0.0)
    with pytest.raises(ValueError, match="left_knee_joint"):
        env._place(R.joints(**{"left_knee_joint": 2.0}), 0.0)


def test_this_body_cannot_KNEEL_and_the_reason_is_geometry(no_knob_env):
    """Why the proposal pose was dropped, kept as a measurement.

    With the rear femur vertical — the hip stacked over the kneeling knee,
    which is what carries the body — the knee geom sits ABOVE the foot geom
    at every reachable ankle angle, and there is no shin collision geom
    between them. So the foot lands first and the knee never touches. A human
    gets round this by folding the toe back ~90 degrees; this ankle has 40 in
    total.
    """
    import mujoco

    from microduck_local.robots import asimov_getup_ref as R

    env = _env()
    m, d = env.model, env.data

    def lowest(sub):
        gs = [i for i in range(m.ngeom) if m.geom_contype[i] != 0
              and sub in m.body(m.geom_bodyid[i]).name]
        assert gs, sub
        return min(float(d.geom_xpos[i][2]) for i in gs)

    shins = [i for i in range(m.ngeom) if m.geom_contype[i] != 0
             and any(t in m.body(m.geom_bodyid[i]).name for t in ("shin", "calf"))]
    assert not shins, "a shin geom appeared — re-measure the kneel"
    jid = {m.joint(j).name: j for j in range(m.njnt)}
    lo, hi = m.jnt_range[jid["left_ankle_pitch_joint"]]
    assert float(hi - lo) < 0.8, "the ankle got more range — re-measure the kneel"
    for ankle in (float(lo), 0.0, float(hi)):
        q = R.joints(**{"left_hip_pitch_joint": -0.05, "left_knee_joint": 1.45,
                        "left_ankle_pitch_joint": ankle})
        env._place(q, 0.0)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        assert lowest("left_knee") > lowest("left_ankle_roll"), (
            "the knee reached the floor before the foot — the kneel may now "
            "be possible, so re-open it")


def test_the_prologue_plays_the_PUSH_UP_in_stepped_time_and_hands_over_at_pike(
        no_knob_env):
    """The push-up has to happen in STEPPED time or nobody can watch it.

    Every get-up run for a week reset at `fold` or `pike`, above the push-up,
    so the lab never showed one and the user kept asking where it was. Doing
    it inside `reset` would not fix that — a reset is one frame. This drives
    the servos for the first ~1.5 s of the episode and IGNORES the policy,
    which is also honest: that half needs no policy and plays at a 0.215 m
    foot span.
    """
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _env(prologue="pushup", action_clip=4.0)
    env.reset(seed=0)
    assert env.in_prologue()
    TB = env.trunk_body_id
    z0 = float(env.data.xpos[TB][2])
    spans, n = [], 0
    # a deliberately WRONG policy output: the prologue must override it
    while env.in_prologue() and n < 400:
        env.step(np.full(env.nj, 3.0, np.float32))
        spans.append(env._foot_span())
        n += 1
    assert not env.in_prologue()
    z1 = float(env.data.xpos[TB][2])
    assert z0 < 0.12, f"the prologue must start prone, not at {z0:.3f}"
    assert abs(z1 - R.BY_NAME["pike"].settled_z) < 0.06, (
        f"handover at {z1:.3f}, not pike ({R.BY_NAME['pike'].settled_z})")
    assert max(spans) < 0.30, f"the push-up sprawled to {max(spans):.3f}"
    assert 50 < n < 120, f"{n} steps is not the measured tempo"
    # and afterwards the policy really is driving. Allow a few steps: this
    # env can carry a one-step COMMAND LAG, so the first post-handover action
    # does not reach `ctrl` on the same step.
    before = env.data.ctrl.copy()
    for _ in range(5):
        env.step(np.full(env.nj, -2.0, np.float32))
    assert not np.allclose(before, env.data.ctrl), (
        "the policy's action never reached ctrl after the handover")


def test_the_vendor_link_names_mean_what_the_CHAIN_says(no_knob_env):
    """`knee_link` is the SHIN, and reading it by eye cost a false finding.

    An MJCF link is named for the joint ABOVE it, so the segment it
    represents runs from that joint to its own child joint. On 2026-10-09 a
    grep of body names for "shin"/"calf" found nothing and was reported as
    "this robot has no shin collision geom", which was then used to argue a
    half-kneel is impossible. The shin exists — a 90 mm x 40 mm capsule on
    `left_knee_link` — and so do the thighs.

    This re-derives every part from the chain and fails if the table drifts
    from the model, so nobody has to recognise a vendor's naming convention.
    """
    from microduck_local.robots import asimov_getup_ref as R

    env = _env()
    m = env.model
    expect = {
        "left_knee_link": "left SHIN",
        "right_knee_link": "right SHIN",
        "left_hip_yaw_link": "left thigh",
        "left_ankle_roll_link": "left foot",
        "left_elbow_link": "left forearm",
        "left_shoulder_roll_link": "left upper arm",
        "waist_yaw_link": "torso",
        "pelvis_link": "pelvis",
    }
    for link, part in expect.items():
        assert R.part_of(m, link) == part, (link, R.part_of(m, link))

    # the derivation itself: knee_link really does span knee -> ankle
    def spans(body_name):
        b = int(np.argmax([m.body(i).name == body_name
                           for i in range(m.nbody)]))
        own = m.joint(m.body_jntadr[b]).name if m.body_jntnum[b] else None
        kids = [m.joint(m.body_jntadr[c]).name for c in range(1, m.nbody)
                if m.body_parentid[c] == b and m.body_jntnum[c]]
        return own, kids

    own, kids = spans("left_knee_link")
    assert own == "left_knee_joint"
    assert "left_ankle_pitch_joint" in kids, kids

    # and the shin really does carry collision geometry
    shin = [g for g in range(m.ngeom)
            if m.body(m.geom_bodyid[g]).name == "left_knee_link"
            and (m.geom_contype[g] or m.geom_conaffinity[g])]
    assert shin, "the shin lost its collision geom — re-open the kneel question"


def test_a_states_builder_pins_its_own_rung_and_cannot_recurse(no_knob_env,
                                                               monkeypatch):
    """`seat_states` builds an env; that env reads the rung from the
    ENVIRONMENT, which the trainer has set to the very rung being built.

    Unpinned it recurses until Python gives up, and the run dies with nothing
    but a `train.log` — which is what happened to
    `teach-asimov_getup_seat-d949c9-s1` on 2026-10-09. Every `*_states`
    builder must pin `start_rung=""`.
    """
    from microduck_local.robots import asimov_getup_ref as R

    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_START_RUNG", "seat")
    q, v = R.seat_states(12)           # would RecursionError if unpinned
    # FEWER than asked for is correct: the builder now drops any variant that
    # does not come to rest ON THE THIGHS with the chest up. Asserting an
    # exact count would pin the jitter's luck, not the contract.
    assert 1 <= len(q) == len(v) <= 12
    env = _env(start_rung="seat")
    env.reset(seed=0)
    z = float(env.data.xpos[env.trunk_body_id][2])
    u = float(-env._projected_gravity()[2])
    # 0.107 m on the thighs, chest up. The earlier 0.33-0.52 band was the
    # BROKEN seat — the robot mid-fall, which settles onto its feet and reads
    # like a crouch.
    assert 0.06 < z < 0.20, z
    assert u > 0.40, u


def test_a_seat_is_SETTLED_on_the_thighs_not_a_robot_mid_fall(no_knob_env):
    """A start state is settled, in contact, and the pose it claims to be.

    The first `seat_states` settled for eight steps and handed back the robot
    moving at **0.28 m/s** — mid-fall — which landed on its feet at 0.43 m and
    read like a crouch. Trained on it, the policy's best moment was trunk
    0.633 at +0.87 uprightness with NO FOOT ON THE FLOOR: airborne, not
    standing, 0/24. The teacher's own signature is uprightness +0.49 at
    0.06-0.12 m, and that is what a seat has to be.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    q, v = R.seat_states(24)
    assert len(q) >= 4, f"only {len(q)} seats came to rest"
    env = _env()
    m, d = env.model, env.data
    for i in range(len(q)):
        d.qpos[:] = q[i]
        d.qvel[:] = v[i]
        mujoco.mj_forward(m, d)
        assert float(np.linalg.norm(v[i][:6])) < 0.08, "the seat is still moving"
        assert int(d.ncon) > 0, "the seat is not touching anything"
        parts = R.floor_parts(env)
        assert any("thigh" in p or "pelvis" in p for p in parts), parts
        assert not any("head" in p or "torso" in p for p in parts), parts
        assert float(-env._projected_gravity()[2]) > 0.40, "chest is not up"
        assert float(d.xpos[env.trunk_body_id][2]) < 0.20, "that is not a seat"


def test_the_rotation_pay_actually_fires_and_resets_each_episode(no_knob_env):
    """The term was DEAD when it was written, and the A/B was byte-identical.

    `_best_upright` was initialised once in the constructor, so it latched
    onto the standing pose (+1.00) and nothing in any episode ever beat it:
    `rotate` paid 0.00 at a weight of 40 while the robot went from +0.83 to
    +1.00. The height term resets `_best_trunk_z` per episode for exactly
    this reason, and the rotation term has to sit beside it.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _env(rotate_pay=40.0, action_clip=4.0)
    m, d = env.model, env.data
    scale = float(np.atleast_1d(env.robot.action_scale)[0])
    default = np.asarray(env.robot.default_pose, float)
    paid = []

    def run_once():
        w = R.BY_NAME["fold"]
        env.reset(seed=0)
        env._place(np.asarray(w.joints, float), w.root_pitch)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        # NO manual reset here. The first version of this test set
        # `_best_upright = None` itself, which bypassed the reset it exists to
        # check — the planted bug passed it.
        cur = np.asarray(w.joints, float)
        u0 = float(-env._projected_gravity()[2])
        total = 0.0
        orig = env._compute_reward

        def spy():
            r, t = orig()
            nonlocal total
            total += t["rotate"]
            return r, t

        env._compute_reward = spy
        for nm in ("crouch", "stand"):
            tgt = np.asarray(R.BY_NAME[nm].joints, float)
            for k in range(13):
                f = (k + 1) / 13
                f = f * f * (3.0 - 2.0 * f)
                d.xfrc_applied[:] = 0.0
                env.step(np.clip((cur + (tgt - cur) * f - default) / scale,
                                 -4.0, 4.0).astype(np.float32))
            cur = tgt
            for _ in range(20):
                d.xfrc_applied[:] = 0.0
                env.step(np.clip((cur - default) / scale,
                                 -4.0, 4.0).astype(np.float32))
        env._compute_reward = orig
        return total, u0, float(-env._projected_gravity()[2])

    for _ in range(2):          # TWICE: the second proves it resets
        total, u0, u1 = run_once()
        paid.append(total)
        assert total > 0.5 * 40.0 * (u1 - u0), (
            f"rotate paid {total:.2f} while uprightness went {u0:+.2f} -> "
            f"{u1:+.2f} — the term is dead")
    assert paid[1] > 0.5 * paid[0], (
        "the second episode paid far less — `_best_upright` is not resetting")
    assert _env(action_clip=4.0)._rotate_pay == 0.0, "it must be OFF by default"


def test_the_tuck_term_pays_bringing_the_feet_UNDER_the_mass(no_knob_env):
    """The move every other term charges for.

    The policy that learned to sit up ends upright on its feet at 0.11 m with
    the knee at **-1 deg, its 0 deg stop** — legs straight out, centre of
    mass **300 mm from the feet**. 86 deg of knee travel, unused. Tucking the
    feet in lowers the trunk and costs uprightness on the way, so `rise`,
    `stand`, `progress` and `rotate` all penalise it; nothing paid for it.

    Monotone like the other two progress terms, or shuffling the feet in and
    out would farm it.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _env(tuck_pay=50.0, action_clip=4.0)
    m, d = env.model, env.data

    def place(name):
        w = R.BY_NAME[name]
        env._place(np.asarray(w.joints, float), w.root_pitch)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    env.reset(seed=0)
    env._best_tuck = None
    place("pike")
    assert env._tuck_gain() == 0.0          # first call sets the baseline
    far = env._best_tuck
    place("fold")
    gain = env._tuck_gain()
    assert gain > 0.1, f"pike -> fold paid {gain:.3f}; the term is dead"
    assert env._best_tuck < far
    place("pike")
    assert env._tuck_gain() == 0.0, "going back out must pay nothing"
    assert _env(action_clip=4.0)._tuck_pay == 0.0, "it must be OFF by default"


def test_the_flight_charge_fires_ONLY_when_nothing_touches_the_floor(
        no_knob_env):
    """It did not push itself up. It HOPPED, and nothing charged for it.

    `teach-asimov_getup_turn-6a1490-s7` is the first policy here that stands
    from the floor, and measured frame by frame its 3.4 s rise contains 8
    separate flights totalling **1.04 s with zero floor contacts**, the lowest
    geom of a 32 kg robot reaching **105 mm** of clearance — and 0.00 s
    airborne once it is standing. Every term in the economy pays for what a
    bounce produces (height, uprightness, feet under the mass) and none of
    them priced the impulse, so ballistic was the cheap route.

    The charge has to ask MuJoCo's CONTACT LIST, not a height: the pose that
    matters is a 1.2 m robot resting on one heel with every other geom well
    clear of the floor, and a clearance threshold calls that flight.
    """
    import mujoco
    import numpy as np

    env = _env(ground_cost=7.0, action_clip=4.0)
    m, d = env.model, env.data
    env.reset(seed=0)

    # On the floor, as it resets: supported, so nothing is owed.
    mujoco.mj_forward(m, d)
    assert env._on_floor_contacts() > 0, "the reset pose is not on the floor"
    assert env._flight_charge() == 0.0

    # Lifted bodily off it: every geom clear, nothing in contact.
    d.qpos[2] += 0.60
    mujoco.mj_forward(m, d)
    assert env._on_floor_contacts() == 0
    assert env._flight_charge() == pytest.approx(7.0)
    lowest = float(np.min(d.geom_xpos[m.geom_bodyid > 0][:, 2]))
    assert lowest > 0.05, f"not actually airborne: lowest geom {lowest:.3f}"

    # And it is OFF unless bought: every run before 2026-10-09 reproduces.
    quiet = _env(action_clip=4.0)
    quiet.reset(seed=0)
    quiet.data.qpos[2] += 0.60
    mujoco.mj_forward(quiet.model, quiet.data)
    assert quiet._on_floor_contacts() == 0
    assert quiet._flight_charge() == 0.0


def test_the_flight_charge_reaches_the_REWARD_on_every_get_up_task(
        no_knob_env):
    """A term that reaches one of three reward dicts is the bug `span_penalty`
    already had: it was added to the two subclasses and not the base, and a
    `descend` curriculum trained at a charge of exactly 0.0 with `span_cost=80`
    sitting in its kwargs. Ask each task's own reward for the term by name."""
    import mujoco

    from microduck_local.robots import asimov_env as E

    for cls in (E.AsimovGetupEnv, E.AsimovGetupRefEnv, E.AsimovGetupClipEnv):
        try:
            env = cls(domain_rand=False, obs_noise=False, ground_cost=7.0,
                      action_clip=4.0)
        except (SystemExit, FileNotFoundError, RuntimeError) as exc:
            pytest.skip(f"{cls.__name__} needs an asset it has not got: {exc}")
        env.reset(seed=0)
        env.data.qpos[2] += 0.60
        mujoco.mj_forward(env.model, env.data)
        env.step(env.action_space.sample() * 0.0)
        terms = env._compute_reward()
        terms = terms[1] if isinstance(terms, tuple) else terms
        assert "flight_penalty" in terms, (
            f"{cls.__name__}'s reward has no flight term")


def test_a_recorded_knob_is_a_REPLAYED_knob(no_knob_env, monkeypatch):
    """The mirror of `test_every_knob_a_recipe_SETS_is_a_knob_the_run_RECORDS`,
    and the half that was missing.

    That test proved a knob reaches `run.json`. Nothing proved it comes BACK
    out: `REPLAY_KWARGS` is what the lab and `render-rollout` rebuild a policy
    with, and `prologue` — a SCRIPTED push-up driving the first 1.56 s of an
    episode — was recorded by the trainer and dropped by every replay. A
    prologue run previewed in the lab would have shown a motion the trainer
    never trained, which is the exact fault that had Jonathan loading stage
    after stage and seeing none of them work.

    Deliberate exclusions are named here, so the next knob has to be
    classified rather than forgotten.
    """
    from microduck_local.robots import spec as S
    from microduck_local.run_record import REPLAY_KWARGS

    # Training conditions, not motion: a preview shows the CLEAN motion.
    EXCLUDED = {"domain_rand", "obs_noise"}

    body = S.get("asimov1")
    missed = []
    for task in ("getup", "getup_ref", "getup_clip"):
        fn = getattr(body, "train_env_kwargs", None)
        if fn is None:
            pytest.skip("this body records no train kwargs")
        rec = dict(fn(types.SimpleNamespace(task=task)))
        missed += [f"{task}.{k}" for k in rec
                   if k not in REPLAY_KWARGS and k not in EXCLUDED]
    assert not missed, (
        "recorded by the trainer and dropped by every replay: "
        + ", ".join(sorted(set(missed))))


def test_the_support_polygon_knows_inside_from_outside(no_knob_env):
    """Geometry first, before any policy is judged by it.

    Built from MuJoCo's CONTACT POINTS, not from geom positions: a pad
    hovering 2 mm clear is not support, and that exact confusion
    (`geom_rbound`) already cost this project a day.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _env(action_clip=4.0)
    env.reset(seed=0)
    w = R.BY_NAME["stand"]
    env._place(np.asarray(w.joints, float), w.root_pitch)
    R.settle_onto_floor(env.model, env.data)
    for _ in range(40):
        mujoco.mj_step(env.model, env.data)

    standing = R.support_margin(env)
    assert standing is not None, "a settled stand has no support polygon"
    assert standing > 0.0, f"a settled stand is not balanced: {standing}"

    # Shove the whole body sideways by half a metre, leaving the pose alone:
    # same joints, same uprightness, mass now nowhere near the feet.
    env.data.qpos[0] += 0.5
    mujoco.mj_forward(env.model, env.data)
    shoved = R.support_margin(env)
    assert shoved is None or shoved < standing, (
        f"moving the body off its feet did not reduce the margin: "
        f"{standing} -> {shoved}")

    # Airborne: no contacts, so there is no polygon — None, never a number.
    env.data.qpos[2] += 0.60
    mujoco.mj_forward(env.model, env.data)
    assert R.support_margin(env) is None

    # THE DISCRIMINATOR, and the reason this reads contacts and not geom
    # positions: `_place` leaves the robot 2 mm CLEAR of the floor. Every
    # pad is a hair above it, so a position-based hull happily reports a
    # healthy margin for a robot that is touching nothing and about to drop.
    # That is the `geom_rbound` trap, which cost this project a day.
    env.reset(seed=0)
    env._place(np.asarray(w.joints, float), w.root_pitch)
    mujoco.mj_forward(env.model, env.data)
    assert int(env.data.ncon) == 0, "the placed pose is already in contact"
    assert R.support_margin(env) is None, (
        "a robot touching NOTHING was given a support polygon — this is "
        "reading geometry instead of contacts")


def test_balance_is_charged_only_while_the_robot_is_UP(no_knob_env,
                                                       monkeypatch):
    """The gate is the whole safety of this term.

    Ungated it charges the entire get-up — a robot mid-rise has its mass
    nowhere near its feet — and a penalty that can eat its own task is the
    recorded failure of `teach-asimov_getup_rise-126f42-s2`, where the
    policy correctly learned not to stand.

    The margin is PINNED here rather than posed into existence. The first
    version of this test shoved a real pose around and passed with the gate
    deleted, twice: flat on the floor the robot's mass sits over its own
    contact patch (a POSITIVE margin), and a hard enough shove lifts the
    forearms so the pose becomes feet-only after all. Neither ever reached
    the branch it was written to protect.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _env(balance_cost=30.0, action_clip=4.0)
    env.reset(seed=0)
    monkeypatch.setattr(R, "support_margin", lambda e: -0.10)  # always falling

    def pose(name):
        w = R.BY_NAME[name]
        env._place(np.asarray(w.joints, float), w.root_pitch)
        R.settle_onto_floor(env.model, env.data)
        for _ in range(20):
            mujoco.mj_step(env.model, env.data)

    # DOWN: prone, on forearms and knees. Not upright, not feet-only.
    pose("quadruped")
    assert float(-env._projected_gravity()[2]) <= R.STAND_UPRIGHT
    assert env._balance_charge() == 0.0, (
        "the RISE is being charged — ungated, this term eats the task it "
        "exists to finish")

    # UP: settled on its feet, upright, and (by the pin) off balance.
    pose("stand")
    assert env._on_feet_only(), "the settled stand is not on its feet alone"
    assert float(-env._projected_gravity()[2]) > R.STAND_UPRIGHT
    assert env._balance_charge() == pytest.approx(30.0 * 0.10)

    # UP but not on its feet alone — a hand down is support, so nothing owed.
    monkeypatch.setattr(env, "_on_feet_only", lambda: False)
    assert env._balance_charge() == 0.0

    # Inside the polygon: one-sided, like the span charge.
    monkeypatch.setattr(env, "_on_feet_only", lambda: True)
    monkeypatch.setattr(R, "support_margin", lambda e: +0.04)
    assert env._balance_charge() == 0.0
    # ...and capped, so one bad step cannot dominate an episode.
    monkeypatch.setattr(R, "support_margin", lambda e: -9.0)
    assert env._balance_charge() == pytest.approx(30.0 * env.BALANCE_CAP)

    # And off by default, so every run before 2026-10-09 reproduces.
    quiet = _env(action_clip=4.0)
    quiet.reset(seed=0)
    monkeypatch.setattr(quiet, "_on_feet_only", lambda: True)
    monkeypatch.setattr(R, "support_margin", lambda e: -0.10)
    assert quiet._balance_charge() == 0.0


def test_a_stand_REPORTS_whether_it_was_balanced(no_knob_env):
    """`standing` stays what it was — redefining it would silently rewrite
    every number this repo has published — but no scorer should have to
    rediscover that a vertical robot can be falling over. The report carries
    both, so a claim can say which one it means."""
    from microduck_local.robots import asimov_getup_ref as R

    env = _env(action_clip=4.0)
    env.reset(seed=0)
    rep = R.standing_report(env)
    assert "com_margin" in rep and "balanced" in rep
    # Lying on the floor is not standing, and therefore not balanced.
    assert rep["standing"] is False and rep["balanced"] is False
    # `balanced` is strictly stronger than `standing`, never weaker.
    assert not (rep["balanced"] and not rep["standing"])


def test_coming_HOME_pays_only_when_up_and_peaks_at_the_default_pose(
        no_knob_env):
    """The get-up has to END somewhere usable, and it did not.

    Measured on `teach-asimov_getup_turn-dd413e`: the pose it stands in is
    19 deg from `default_pose` on average and 57 deg at the waist, with
    EIGHT joints outside the model's limits — it jams against its stops and
    calls it a stand. The home pose is not invented: under zero action the
    servos hold it 8.00 s of 8 with the CoM +47 mm inside the polygon.
    """
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _env(home_pay=10.0, action_clip=4.0)
    env.reset(seed=0)

    # DOWN: flat on the floor, which is where an ungated version would pay
    # for lying neatly instead of standing.
    assert float(-env._projected_gravity()[2]) <= R.STAND_UPRIGHT
    assert env._home_gain() == 0.0, "the floor is being paid to look tidy"

    # UP, at home: the maximum, and it is the WEIGHT (err = 0 -> exp(0) = 1).
    # STEP, don't just settle: `settle_onto_floor` leaves ~1 mm of
    # clearance, so a feet-only check run straight after it sees NO contacts
    # and the gate reads "not on its feet". That is the contact-before-
    # settling trap this repo has already paid for twice, and it is why this
    # assertion read 0.0 the first time.
    w = R.BY_NAME["stand"]
    env._place(np.asarray(w.joints, float), w.root_pitch)
    R.settle_onto_floor(env.model, env.data)
    for _ in range(40):
        mujoco.mj_step(env.model, env.data)
    assert env._on_feet_only(), "the settled stand is not on its feet"
    at_home = env._home_gain()
    # Settled under gravity the joints drift ~0.8 deg off the commanded
    # pose, so the SETTLED value is near the maximum, not at it...
    assert at_home == pytest.approx(10.0, rel=0.1), at_home

    # ...and the maximum itself is exactly the weight, checked on the pose
    # as placed (err = 0 -> exp(0) = 1), with the gate pinned so this tests
    # the arithmetic and not the contact set.
    env._place(np.asarray(w.joints, float), w.root_pitch)
    mujoco.mj_forward(env.model, env.data)
    pinned = env._on_feet_only
    try:
        env._on_feet_only = lambda: True
        assert env._home_gain() == pytest.approx(10.0, rel=1e-6)
    finally:
        env._on_feet_only = pinned

    # UP, but 20 deg off at every joint: strictly less, and still positive so
    # there is a gradient to climb rather than a cliff.
    adr = np.atleast_1d(getattr(env, "_q_adr", env.joint_qpos_adr))
    env.data.qpos[adr] += np.radians(20.0)
    mujoco.mj_forward(env.model, env.data)
    off = env._home_gain() if env._on_feet_only() else None
    if off is None:            # the shove lifted it off its feet; re-pin
        env._on_feet_only = lambda: True
        off = env._home_gain()
    assert 0.0 < off < at_home, f"home pay does not fall off: {at_home} {off}"

    # A KNEEL in the home pose is vertical and would score uprightness ~+1.
    # Under an uprightness-only gate it collects full price for never
    # standing up — the proxy that failure also satisfies.
    env._place(np.asarray(w.joints, float), w.root_pitch)
    mujoco.mj_forward(env.model, env.data)
    monkey = env._on_feet_only
    try:
        env._on_feet_only = lambda: False
        assert env._home_gain() == 0.0, "a kneel is being paid to come home"
    finally:
        env._on_feet_only = monkey

    quiet = _env(action_clip=4.0)          # off by default
    quiet.reset(seed=0)
    quiet._place(np.asarray(w.joints, float), w.root_pitch)
    mujoco.mj_forward(quiet.model, quiet.data)
    assert quiet._home_gain() == 0.0


def test_standing_ON_a_joint_stop_is_charged(no_knob_env):
    """MuJoCo's joint limits are SOFT, so a policy can push a joint through
    its stop and stand on the constraint force. Nothing charged for that,
    and the first policy that stood from the floor did it on eight joints at
    once — a pose `_refuse_illegal_joints` will not even load, and one that
    cannot exist on the real robot."""
    import mujoco
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _env(limit_cost=50.0, action_clip=4.0)
    env.reset(seed=0)
    w = R.BY_NAME["stand"]
    env._place(np.asarray(w.joints, float), w.root_pitch)
    mujoco.mj_forward(env.model, env.data)
    assert env._limit_charge() == 0.0, "a legal pose is being charged"

    # BOTH stops, separately. The measured jam ran through the UPPER limit on
    # the ankles and the LOWER one on `left_knee` (-0.018 against a +0.000
    # floor), and a charge written for one direction misses half of it — the
    # first version of this test only ever drove a joint past its upper stop
    # and passed with the lower branch deleted.
    m = env.model
    adr = np.atleast_1d(getattr(env, "_q_adr", env.joint_qpos_adr))
    rng = {int(a): (float(m.jnt_range[int(np.argmax(m.jnt_qposadr == a))][0]),
                    float(m.jnt_range[int(np.argmax(m.jnt_qposadr == a))][1]))
           for a in adr}
    pair = [a for a in adr if rng[int(a)][1] > rng[int(a)][0]]
    assert len(pair) >= 2, "need two limited joints to test both directions"

    for direction, a in (("above", pair[0]), ("below", pair[1])):
        env._place(np.asarray(w.joints, float), w.root_pitch)
        lo, hi = rng[int(a)]
        env.data.qpos[int(a)] = (hi + 0.1) if direction == "above" else lo - 0.1
        mujoco.mj_forward(m, env.data)
        assert env._limit_charge() == pytest.approx(50.0 * 0.1, rel=1e-3), (
            f"a joint 0.1 rad {direction} its stop was charged "
            f"{env._limit_charge()}")

    # ...and both at once add up, so eight jammed joints cost eight times.
    env._place(np.asarray(w.joints, float), w.root_pitch)
    env.data.qpos[int(pair[0])] = rng[int(pair[0])][1] + 0.1
    env.data.qpos[int(pair[1])] = rng[int(pair[1])][0] - 0.1
    mujoco.mj_forward(m, env.data)
    assert env._limit_charge() == pytest.approx(50.0 * 0.2, rel=1e-3)

    quiet = _env(action_clip=4.0)
    quiet.reset(seed=0)
    quiet.data.qpos[adr[0]] = 99.0
    mujoco.mj_forward(quiet.model, quiet.data)
    assert quiet._limit_charge() == 0.0


def test_a_shove_lands_only_on_a_STANDING_robot_and_rearms_each_episode(
        no_knob_env):
    """The state 38 M steps never sampled.

    Across this whole curriculum `xfrc_applied` carried the vertical ASSIST
    and nothing else, so the robot was never once disturbed — and "it stands
    but never corrects" was therefore not a reward problem at all. A recovery
    is a state no rollout contained, and an unsampled state's value is never
    learned.

    Two things have to hold. A shove must not land during the RISE, where it
    is just noise on a motion that is already hard. And the clock must re-arm
    PER EPISODE — `rotate` paid 0.00 at a weight of 40 for exactly one
    reason, a per-episode quantity initialised once in the constructor.
    """
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _env(push_bw=0.25, action_clip=4.0)
    zero = np.zeros(env.action_space.shape, np.float32)
    tb = env.trunk_body_id

    # ON THE FLOOR: never shoved, however long we run.
    env.reset(seed=0)
    for _ in range(120):
        _, _, t, tr, info = env.step(zero)
        assert not R.is_standing(env), "this fixture is supposed to stay down"
        assert float(np.abs(env.data.xfrc_applied[tb, 0:2]).sum()) == 0.0, (
            "the RISE is being shoved")
        if t or tr:
            break
    assert info.get("pushes", 0) == 0

    # STANDING: shoves land, and they are horizontal only — the vertical
    # channel belongs to the assist and clobbering it would silently switch
    # the assist off mid-episode.
    def stand_and_count(e):
        e.reset(seed=0)
        w = R.BY_NAME["stand"]
        e._place(np.asarray(w.joints, float), w.root_pitch)
        R.settle_onto_floor(e.model, e.data)
        seen_z = 0.0
        for _ in range(300):
            _, _, t, tr, info = e.step(zero)
            seen_z = max(seen_z, abs(float(e.data.xfrc_applied[tb, 2])))
            if t or tr:
                break
        return info.get("pushes", 0), seen_z

    n, seen_z = stand_and_count(env)
    assert n > 0, "a standing robot was never shoved"
    assert seen_z == 0.0, "the shove wrote the assist's vertical channel"

    # ...and it re-arms: a second episode shoves again rather than staying
    # latched on the first episode's clock.
    n2, _ = stand_and_count(env)
    assert n2 > 0, "the shove clock did not re-arm for the next episode"

    # Off by default, so every run before 2026-10-10 reproduces.
    quiet = _env(action_clip=4.0)
    n3, _ = stand_and_count(quiet)
    assert n3 == 0


def test_arriving_FAST_is_charged_and_arriving_quietly_is_free(no_knob_env):
    """The measurement three stages of pose work were missing.

    At the instant it first satisfies `is_standing`, the trunk is moving
    **0.89 m/s** sideways against a servo-held home stand that absorbs
    **0.22** — four times too fast — and inside the same 24 episodes the
    ones that stayed up arrived at 0.60-0.87 while the ones that fell
    arrived at 0.88-0.98. No pose survives that, which is why `balance`,
    `home` and a servo handoff each bought little and then stalled.

    One-sided above `CALM_FREE` so a quiet stand pays nothing, and gated on
    being on its feet so the RISE — which has to move fast — is never
    charged for moving.
    """
    import numpy as np

    from microduck_local.robots import asimov_getup_ref as R

    env = _env(calm_cost=10.0, action_clip=4.0)
    zero = np.zeros(env.action_space.shape, np.float32)

    # ON THE FLOOR, mid-rise: never charged, however fast it is moving.
    env.reset(seed=0)
    for _ in range(60):
        env.step(zero)
        assert env._calm_charge() == 0.0, "the RISE is being charged for speed"

    # STANDING AND STILL: free, because 0 < CALM_FREE.
    w = R.BY_NAME["stand"]
    env._place(np.asarray(w.joints, float), w.root_pitch)
    R.settle_onto_floor(env.model, env.data)
    for _ in range(40):
        env.step(zero)
    assert env._on_feet_only()
    assert env._calm_charge() == 0.0, "a still stand is being charged"

    # STANDING AND MOVING, INSIDE THE ARRIVAL WINDOW: charged on the EXCESS
    # only, not the whole speed.
    env._stood_at = int(getattr(env, "_ep_k", 0))
    env._prev_trunk_xy = (np.asarray(env.data.xpos[env.trunk_body_id][:2],
                                     float) - np.array([0.89 / 50.0, 0.0]))
    pinned = env._on_feet_only
    try:
        env._on_feet_only = lambda: True
        got = env._calm_charge()
    finally:
        env._on_feet_only = pinned
    assert got == pytest.approx(10.0 * (0.89 - env.CALM_FREE), rel=1e-3), got

    # ...AND NOT AFTER IT. Charging speed for as long as the robot is up
    # prices the CORRECTIONS that keep it there: measured, that version took
    # `stays` from 13/24 to 6/24. The lunge is how it ARRIVES, not how it
    # holds, and forcing the arms to neutral on a standing policy takes it
    # from 13/24 to 0/24 — the arms are load-bearing because of that lunge.
    env._stood_at = (int(getattr(env, "_ep_k", 0))
                     - int(env.CALM_WINDOW_S * 50) - 1)
    env._prev_trunk_xy = (np.asarray(env.data.xpos[env.trunk_body_id][:2],
                                     float) - np.array([0.89 / 50.0, 0.0]))
    try:
        env._on_feet_only = lambda: True
        assert env._calm_charge() == 0.0, (
            "a correction long after arriving is being charged")
    finally:
        env._on_feet_only = pinned

    # Off by default, so every run before 2026-10-10 reproduces.
    quiet = _env(action_clip=4.0)
    quiet.reset(seed=0)
    quiet._prev_trunk_xy = (np.asarray(quiet.data.xpos[quiet.trunk_body_id][:2],
                                       float) - np.array([0.89 / 50.0, 0.0]))
    quiet._on_feet_only = lambda: True
    assert quiet._calm_charge() == 0.0


def test_a_WIDER_arm_sweep_earns_more_and_the_same_sweep_earns_once(
        no_knob_env):
    """Jonathan's kip, and the third design of this term.

    What the kip needs is AMPLITUDE: measured open loop with the legs
    passive, a 200 deg sweep lifts the trunk 16 mm and rotates it -0.13 ->
    -0.05, while holding the arms overhead WITHOUT sweeping does nothing at
    all. The first two designs could not ask for that:

      * paid on the rate alone, `swing` earned **1807 an episode** for lying
        on the floor waving;
      * multiplied by upward trunk speed it was unfarmable and UNLEARNABLE —
        a policy that cannot yet rise has `up` about zero, so the term meant
        to cause the rise only paid once the rise existed, and the arm sweep
        moved 90.9 -> 96.6 deg, i.e. not at all;
      * paid per step against a per-episode budget, random thrashing hit the
        cap inside two seconds, so a BIGGER swing earned nothing extra.

    Monotone gain in peak-to-peak sweep asks for the amplitude directly.
    """
    import mujoco
    import numpy as np

    env = _env(swing_pay=40.0, action_clip=4.0)
    env.reset(seed=0)
    env._shoulder_pitch()
    lsh, rsh = env._sh_adr

    def sweep_episode(amp_deg, steps=100, anti=False):
        env.reset(seed=0)
        env._shoulder_pitch()
        paid = 0.0
        for k in range(steps):
            f = np.sin(k / 8.0) * np.radians(amp_deg) / 2.0
            env.data.qpos[lsh] = (+f if anti else -f)
            env.data.qpos[rsh] = +f
            mujoco.mj_forward(env.model, env.data)
            env._ep_k = k + 1
            paid += env._swing_gain()
        return paid

    narrow = sweep_episode(90.0)
    wide = sweep_episode(200.0)
    assert wide > narrow * 1.5, (
        f"a 200 deg sweep ({wide:.1f}) must earn much more than 90 deg "
        f"({narrow:.1f}) — this is not asking for amplitude")

    # MONOTONE: the same sweep repeated inside one episode earns nothing
    # more. A bare rate is what made this farmable at 1807 an episode.
    env.reset(seed=0)
    env._shoulder_pitch()
    first = second = 0.0
    for cycle in range(2):
        for k in range(60):
            f = np.sin(k / 8.0) * np.radians(200.0) / 2.0
            env.data.qpos[lsh] = -f
            env.data.qpos[rsh] = +f
            mujoco.mj_forward(env.model, env.data)
            env._ep_k = cycle * 60 + k + 1
            got = env._swing_gain()
            if cycle == 0:
                first += got
            else:
                second += got
    assert first > 0.0
    assert second == pytest.approx(0.0, abs=1e-9), (
        f"waving again earned {second:.3f} more — the term is not monotone")

    # ANTI-SYMMETRIC (one arm up, one down) is a balance flail, not a swing.
    assert sweep_episode(200.0, anti=True) < narrow * 0.5

    # THE WINDOW: a kip happens at the start or not at all.
    #
    # The baseline has to be ESTABLISHED first. Asserting straight after a
    # reset proves nothing — with no min/max yet the sweep is 0 and the gain
    # is 0 whether the window exists or not, and that version of this
    # assertion passed with the window deleted.
    env.reset(seed=0)
    env._shoulder_pitch()
    env._ep_k = 1
    env.data.qpos[lsh] = 0.0
    env.data.qpos[rsh] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env._swing_gain()                       # seeds lo/hi at 0
    env.data.qpos[lsh] = -1.0
    env.data.qpos[rsh] = +1.0
    mujoco.mj_forward(env.model, env.data)
    env._ep_k = 2
    inside = env._swing_gain()
    assert inside > 0.0, "a wide sweep INSIDE the window earns nothing"

    env.reset(seed=0)
    env._shoulder_pitch()
    env._ep_k = 1
    env.data.qpos[lsh] = 0.0
    env.data.qpos[rsh] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env._swing_gain()
    env.data.qpos[lsh] = -1.0
    env.data.qpos[rsh] = +1.0
    mujoco.mj_forward(env.model, env.data)
    env._ep_k = int(env.SWING_WINDOW_S * 50) + 1
    assert env._swing_gain() == 0.0, "the same wave LATE is being paid"

    # PER EPISODE. A best-so-far that lives in the constructor is the bug
    # that made `rotate` pay 0.00 at a weight of 40.
    again = sweep_episode(200.0)
    assert again == pytest.approx(wide, rel=0.05), (
        "the sweep baseline did not reset between episodes")

    # Off by default, so every run before 2026-10-10 reproduces.
    quiet = _env(action_clip=4.0)
    quiet.reset(seed=0)
    quiet._shoulder_pitch()
    ql, qr = quiet._sh_adr
    quiet.data.qpos[ql] = -1.0
    quiet.data.qpos[qr] = +1.0
    mujoco.mj_forward(quiet.model, quiet.data)
    quiet._ep_k = 1
    assert quiet._swing_gain() == 0.0


def test_the_handoff_ENDS_the_episode_and_pays_what_standing_would_have(
        no_knob_env):
    """The job is to deliver the robot, not to hold it up for eight seconds.

    Everything before this trained the policy to be its own balance
    controller for a whole episode, which is why its standing pose puts
    EIGHT joints outside their limits and sits 49 deg from default — holding
    itself up is the only tool it has. It does not need that tool: handing to
    Menlo's walker beats it on every axis (20/24 vs 13/24 standing, 3.7 deg
    from idle vs 49.1, 100 % shove recovery vs 75 %).

    Measured over 128 handoffs (97 caught), trunk SPEED is the only strong
    separator — 0.409 m/s caught against 0.793 dropped, 1.19 sd — so the
    score ramps on speed alone.

    The bonus has to REPLACE the forgone income. A terminal bonus smaller
    than the rest of the episode teaches the policy to avoid the terminal,
    which is the asymmetric-cut failure this repo already has twice.
    """
    import numpy as np

    env = _env(handoff_pay=14.0, action_clip=4.0)
    env.reset(seed=0)

    # The SCORE is 1 at the catchable speed and 0 at the speed the walker
    # drops, and it never goes negative outside that band.
    for speed, want in ((0.0, 1.0), (env.HANDOFF_GOOD, 1.0),
                        (env.HANDOFF_DROP, 0.0), (9.0, 0.0)):
        env._handoff_speed = speed
        assert env._handoff_score() == pytest.approx(want, abs=1e-6), speed
    env._handoff_speed = (env.HANDOFF_GOOD + env.HANDOFF_DROP) / 2
    assert 0.4 < env._handoff_score() < 0.6

    # THE DWELL: one good step is not a handoff.
    env.reset(seed=0)
    env._handoff_streak = 0
    pinned = {"ok": True}
    import microduck_local.robots.asimov_getup_ref as R
    orig = R.is_standing
    try:
        R.is_standing = lambda e: pinned["ok"]
        env._handoff_prev_xy = np.asarray(
            env.data.xpos[env.trunk_body_id][:2], float)
        fired = [env.handoff_ready() for _ in range(env.HANDOFF_DWELL)]
        assert fired[:-1] == [False] * (env.HANDOFF_DWELL - 1), fired
        assert fired[-1] is True
        # ...and a single bad step resets it, so this cannot be accumulated
        # across a stumble.
        pinned["ok"] = False
        assert env.handoff_ready() is False
        assert env._handoff_streak == 0
    finally:
        R.is_standing = orig

    # THE BONUS REPLACES THE FORGONE INCOME, so it must be PROPORTIONAL to
    # the steps left. A flat terminal bonus is smaller than the rest of the
    # episode and teaches the policy to AVOID the terminal — the asymmetric
    # cut this repo has already paid for twice, and the first version of
    # this test passed with exactly that bug planted.
    def bonus_at(step):
        e = _env(handoff_pay=14.0, action_clip=4.0)
        e.reset(seed=0)
        e._ep_k = step
        e._handoff_speed = 0.0                   # score 1.0
        e.handoff_ready = lambda: True
        a = np.zeros(e.action_space.shape, np.float32)
        r_with = e.step(a)[1]
        q = _env(action_clip=4.0)                # same step, no handoff pay
        q.reset(seed=0)
        q._ep_k = step
        r_without = q.step(a)[1]
        return float(r_with - r_without)

    early, late = bonus_at(50), bonus_at(350)
    assert early > late * 3, (
        f"the bonus does not scale with the steps left ({early:.0f} at step "
        f"50 vs {late:.0f} at step 350) — a flat terminal is one the policy "
        f"learns to avoid")
    steps = int(_env(handoff_pay=14.0, action_clip=4.0).max_steps)
    assert early == pytest.approx(14.0 * (steps - 50), rel=0.02), early

    # OFF BY DEFAULT: no terminal, no bonus, so every earlier run reproduces.
    quiet = _env(action_clip=4.0)
    quiet.reset(seed=0)
    assert quiet._handoff_pay == 0.0
    ended = False
    for _ in range(60):
        _, _, t, tr, _i = quiet.step(
            np.zeros(quiet.action_space.shape, np.float32))
        ended = ended or bool(t)
        assert "handoff_score" not in _i
    assert not ended


def test_the_recipe_and_the_reward_ask_the_SAME_handoff_question(no_knob_env):
    """One definition. The lab deploys `behaviors.asimov_tasks`' predicate
    and the trainer rewards the env's; two copies of a threshold kept in
    step by a comment is how this repo got ten disagreeing definitions of
    "did it stand"."""
    from microduck_local.behaviors import asimov_tasks as T

    env = _env(action_clip=4.0)
    env.reset(seed=0)
    calls = []
    env.handoff_ready = lambda: (calls.append(1), True)[1]
    assert T._getup_handoff_due(env) is True
    assert calls, "the recipe does not delegate to the env's definition"


def test_the_roll_term_reads_the_SUPINE_AXIS_and_is_monotone():
    """`roll` pays `f_z`, not uprightness — and pays for TOUCHING, once.

    Uprightness is ~0 for prone, supine AND side alike, so it is structurally
    blind to a roll: that is why `rotate_pay` (which pays uprightness) came
    back a null on prone. `f_z` — the world-z component of the torso's
    forward axis, the quantity Menlo's own classifier takes the argmax of —
    runs -1 face down to +1 face up.
    """
    import mujoco

    from microduck_local.robots import asimov_getup as G
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="prone",
        roll_pay=10.0)
    env.reset(seed=0)

    # THE AXIS: prone is negative, supine positive, and it agrees with the
    # classifier both times.
    f_prone = env._supine_axis()
    assert f_prone < -0.8, f_prone
    assert G.label(env.model, env.data)[0] == "prone"

    up = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="supine",
        roll_pay=10.0)
    up.reset(seed=0)
    assert up._supine_axis() > 0.8, up._supine_axis()
    assert G.label(up.model, up.data)[0] == "supine"

    # UPRIGHTNESS CANNOT TELL THEM APART — the reason this term exists.
    assert abs(-env._projected_gravity()[2]) < 0.3
    assert abs(-up._projected_gravity()[2]) < 0.3

    # MONOTONE: the first call seeds the baseline and pays nothing; turning
    # the chest up pays the gain ONCE; turning back pays nothing.
    env.reset(seed=0)
    assert env._best_supine is None, "the baseline must reset per episode"
    assert env._roll_gain() == 0.0
    adr = env.model.joint("waist_yaw_joint").qposadr[0]
    q = env.data.qpos.copy()
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(q[3:7], float),
                       np.array([0.0, 1.0, 0.0, 0.0]))
    env.data.qpos[3:7] = out
    mujoco.mj_forward(env.model, env.data)
    first = env._roll_gain()
    assert first > 0.3, first
    assert env._roll_gain() == 0.0, "a monotone term must not pay twice"
    env.data.qpos[:] = q                      # back onto its face
    mujoco.mj_forward(env.model, env.data)
    assert env._roll_gain() == 0.0, "rolling back must not be paid"
    assert adr >= 0


def test_the_roll_terms_are_OFF_by_default_and_recorded_when_on():
    """Off by default, so every earlier run reproduces — and RECORDED.

    "A knob a run SETS and does not RECORD is the bug this file has now had
    five times."
    """
    from microduck_local.robots import spec as S
    from microduck_local.run_record import REPLAY_KWARGS

    off = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="prone")
    off.reset(seed=0)
    _r, terms = off._compute_reward()
    assert terms["roll"] == 0.0 and terms["on_back"] == 0.0

    on = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="supine",
        roll_pay=100.0)
    on.reset(seed=0)
    _r, terms = on._compute_reward()
    # on its back: the per-step term pays, and it is SMALL next to standing
    assert terms["on_back"] == pytest.approx(100.0 * on.ROLL_HOLD_W)
    assert terms["on_back"] < on.W_STANDING * 2, (
        "a per-step pay for LYING on the back must not rival standing")
    assert "roll_pay" in REPLAY_KWARGS


def test_limb_speed_is_charged_DURING_the_rise_not_only_once_up():
    """`joint_vel_penalty` is gated on `up`, so the rise was never charged.

    Measured peak joint rate: walking 3 deg/s median, RISING 532 median and
    1753 max — at or above a human punch. The hazard is the get-up, not the
    fall (which topples away from the pusher 19/19), and nothing priced it.
    """
    from microduck_local.robots import spec as S

    # OFF by default, so every earlier run reproduces.
    off = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="supine")
    off.reset(seed=0)
    assert off._compute_reward()[1]["rise_vel_penalty"] == 0.0

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="supine",
        rise_vel_cost=10.0)
    env.reset(seed=0)

    # LYING DOWN (not `up`): the new term charges, the old one does not.
    _r, terms = env._compute_reward()
    assert terms["rise_vel_penalty"] <= 0.0
    assert terms["joint_vel_penalty"] == 0.0, (
        "the original term is gated on `up` — that is the gap this fills")

    # the two are COMPLEMENTARY, never both live: whatever the pose, exactly
    # one of them can be non-zero, so limb speed is charged everywhere.
    for _ in range(40):
        env.step(env.action_space.sample() * 0.0)
        _r, terms = env._compute_reward()
        assert not (terms["rise_vel_penalty"] < 0 and terms["joint_vel_penalty"] < 0)

    # ...and it scales with the weight. BOTH envs stepped identically from
    # the same seed, because at reset the joint velocities are zero and the
    # first version of this compared a 40-step-old env against a fresh one.
    import numpy as _np

    pair = [S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="supine",
        rise_vel_cost=w) for w in (10.0, 20.0)]
    act = None
    for e in pair:
        e.reset(seed=0)
        if act is None:
            act = (_np.ones(e.action_space.shape, _np.float32) * 0.3)
        for _ in range(10):
            e.step(act)
    a = pair[0]._compute_reward()[1]["rise_vel_penalty"]
    b = pair[1]._compute_reward()[1]["rise_vel_penalty"]
    assert a < 0.0, "ten steps of real motion must cost something"
    assert b == pytest.approx(2 * a, rel=1e-6)


def test_the_observation_ring_is_CLEAN_after_a_reset(no_knob_env):
    """At `history > 1`, every frame must hold the same thing after a reset.

    `reset` calls `_get_obs` MORE THAN ONCE — the base builds an observation,
    then this class repositions the robot (a prologue rung, a fall state, a
    pose-cache draw) and builds another. While `_get_obs` advanced the ring on
    every read, that left four copies of a PRE-reposition frame beside one
    real one: a 5-frame velocity transient at the start of every episode that
    no episode contained, silently, in training only.

    Planted regressions (both confirmed to fail this):
      * `if not self._obs_stack` instead of `if not self._obs_live or not
        self._obs_stack` -> the frames differ after reset. This IS the
        original bug.
      * never setting `_obs_live` in `step` -> the second assertion fails,
        because the ring then never advances and the policy is blind to motion.

    Deliberately NOT claimed: moving the `_obs_stack = []` clear to after
    `super().reset()` was planted and this test still passed, because
    `_obs_live` already forces a rebuild — so the clear is belt-and-braces and
    the order is not load-bearing. Said here rather than left implied, since a
    plant list that names a regression it does not catch is worse than a short
    one.
    """
    import numpy as np
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, history=5,
        getup_label="prone")
    assert env.observation_space.shape == (390,), env.observation_space
    o, _ = env.reset(seed=0)
    for k in range(1, 5):
        assert np.allclose(o[:78], o[78 * k:78 * (k + 1)]), (
            f"frame {k} differs from frame 0 straight after a reset — the ring "
            "advanced during reset instead of rebuilding")
    for _ in range(6):
        o, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
    assert not np.allclose(o[:78], o[-78:]), (
        "the ring never advanced while stepping — the policy cannot see motion")
    a, _ = env.reset(seed=1)
    assert np.allclose(a[:78], a[-78:]), (
        "a second reset left the previous episode's frames in the ring")


def test_history_1_is_BIT_IDENTICAL_to_no_history(no_knob_env):
    """The contract arm must not be perturbed by the control arm existing.

    This one has NO planted regression that fails it, and that is worth
    stating: at N == 1 the stacked path and the plain path are genuinely
    equivalent — `78 * 1` is 78 and a one-slot ring returns its only frame —
    so there is nothing to catch today. It is kept as a GUARD against a future
    change that makes N == 1 diverge (an extra slot, a different fill, a
    normalisation applied only on the stacked path), not as evidence that
    anything is currently right.
    """
    import numpy as np
    from microduck_local.robots import spec as S

    rows = {}
    for h in (None, 1):
        env = S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=False, seed=0, history=h,
            getup_label="prone")
        o, _ = env.reset(seed=3)
        seq = [o.copy()]
        for _ in range(8):
            o, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
            seq.append(o.copy())
        rows[h] = np.asarray(seq)
        assert env.observation_space.shape == (78,), (h, env.observation_space)
    assert np.array_equal(rows[None], rows[1]), (
        "history=1 is not bit-identical to history unset")


def test_arm_work_is_signed_and_cannot_be_farmed_by_oscillation():
    """`arm_work` pays SIGNED work through the arms, so a bounce nets zero.

    WHY THE TERM EXISTS. `arm_push` is `load x max(_trunk_up_speed(), 0)` and
    `_trunk_up_speed()` returns only the gain ABOVE the episode's best trunk
    height. That makes it unfarmable — after the first bounce nothing exceeds
    the best again — and it also makes it unable to START a rise. Measured on
    `teach-asimov_getup_ref-140db6` from `pike`, `arm_push` earned **0.00 per
    episode at w=150**, because the robot only ever sinks and so never
    exceeds its start height. The whole budget there was `track` 85.3 +
    `contact` 73.0 with every other term exactly zero, so nothing at all paid
    for getting up.

    Signed work is dense from the first step and still unfarmable, by a
    DIFFERENT mechanism: the integral of signed `dz` round a closed cycle is
    zero, so oscillation churns the per-step value and nets nothing. This
    pins both halves of that claim — a large per-step magnitude with a small
    net — because asserting only the net would also pass for a term that is
    dead.
    """
    import numpy as np

    from microduck_local.robots import asimov
    from microduck_local.robots import spec as S

    if not asimov.ready():
        pytest.skip("Asimov's model is not fetched")

    def series(act, label="all", steps=120):
        env = S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=False, action_delay=False, seed=0,
            arm_work=1.0, arm_push=0.0, arm_effort=0.0, action_clip=13.0,
            getup_label=label)
        env.reset(seed=1)
        env.reward_sums.clear()
        prev, deltas = 0.0, []
        for k in range(min(steps, int(env.max_steps))):
            _o, _r, term, trunc, _i = env.step(act(env, k))
            cur = float(env.reward_sums.get("arm_work", 0.0))
            deltas.append(cur - prev)
            prev = cur
            if term or trunc:
                break
        return np.asarray(deltas)

    bounce = series(lambda e, k: (np.sin(k * 0.6) * 3.0
                                  * np.ones(e.nj)).astype(np.float32))
    churn, net = np.abs(bounce).sum(), abs(bounce.sum())
    assert churn > 1.0, (
        f"arm_work churned only {churn:.3f} over a bounce — the term is DEAD, "
        "not unfarmable, and the net assertion below would pass vacuously")
    assert net < 0.25 * churn, (
        f"a bounce nets {net:.3f} of {churn:.3f} churned — signed work is "
        "supposed to cancel round a closed cycle, so this is farmable")

    # ...AND IT IS DENSE FROM STEP ONE, which is the whole point. Zero action
    # from the pose mix sinks, and the term must SAY SO rather than reading
    # 0.00 the way `arm_push` does from a rung the robot cannot leave.
    still = series(lambda e, k: np.zeros(e.nj, np.float32))
    assert still.sum() < 0.0, (
        f"zero action nets {still.sum():+.3f}: a policy resting on its "
        "forearms while the trunk sinks must be charged, or the term has the "
        "same can't-initiate defect as `arm_push`")


@pytest.mark.parametrize("rung", ["route:0.45,0.62", "falling", "pike"])
def test_every_start_rung_clears_the_per_episode_baselines(rung):
    """A rung that returns early from `reset` must still clear EVERY
    per-episode baseline.

    The `_fall_q` and `_handover_q` branches used to return having cleared
    only `_best_rise`, `_stood` and `_held`, so `falling`, `pike`, `seat`,
    `handover` and `route` runs carried `_best_trunk_z`, `_prev_trunk_z`,
    `_prev_tuck_gap`, `_prev_tuck2`, `_best_upright`, `_best_supine`,
    `_best_tuck`, `_best_head`, `_peak_rise`, `_held_best` and the whole push
    clock ACROSS episodes — monotone terms comparing two different episodes,
    which this file already records twice as a cause of a term paying 0.00.
    It surfaced as a hard crash only when `calm_cost` was first switched on
    for such a run (`_calm_charge` reads `_stood_at`, created by
    `_reset_push`): the trainer raised on the first step of every episode and
    sat at **0 of 4,000,000 steps** while the lab reported `running: True`.

    Asserting on the SECOND reset, after steps have dirtied the baselines —
    a first reset would pass even with the bug present.
    """
    import numpy as np

    from microduck_local.robots import asimov
    from microduck_local.robots import spec as S

    if not asimov.ready():
        pytest.skip("Asimov's model is not fetched")
    try:
        env = S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=False, action_delay=False, seed=0,
            actuator_force="dc", history=5, start_rung=rung, calm_cost=10.0,
            fall_depth=1.0, fell_cut=False, max_episode_s=8.0)
    except RuntimeError as e:                       # cache not harvested here
        pytest.skip(str(e)[:60])

    env.reset(seed=0)
    for _ in range(12):                             # dirty every baseline
        env.step(np.zeros(env.action_space.shape, np.float32))
    assert env._best_trunk_z is not None, "the step loop did not dirty them"

    env.reset(seed=1)
    stale = [n for n in ("_best_trunk_z", "_prev_trunk_z", "_prev_tuck_gap",
                         "_prev_tuck2", "_best_upright", "_best_supine",
                         "_best_tuck", "_best_head")
             if getattr(env, n, "MISSING") is not None]
    assert not stale, (
        f"{rung}: {stale} survived the reset — a monotone term will compare "
        "this episode against the last one")
    assert env._stood_at is None, "_reset_push did not run on this rung"
    assert env._held == 0 and env._held_best == 0


def assert_term_is_not_farmable(term, weight, *, label="all", churn_min=1.0,
                                net_frac=0.25, steps=120, **env_kw):
    """REUSABLE FARM TEST for any signed reward term.

    A signed term is farm-resistant by CANCELLATION: the integral of a signed
    rate round a closed cycle is zero, so an oscillation churns the per-step
    value and nets nothing. This pins BOTH halves, because asserting only the
    small net ALSO passes for a term that is simply dead — which is the shape
    of a green test that checks nothing.

    Measured on `arm_work`: a bounce captures **100 % of its churn** under the
    rejected `max(dz, 0)` clamp and **under 25 %** signed.

    `tuck_work` and `rotate_work` shipped WITHOUT a test like this and were
    both gamed in training — `tuck_work` bought the 0.917 m splits,
    `rotate_work` bought a roll onto the back. Call this for every new signed
    term.
    """
    import numpy as np

    from microduck_local.robots import spec as S

    kw = dict(action_clip=13.0, getup_label=label)
    kw[term] = weight
    kw.update(env_kw)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0, **kw)
    env.reset(seed=1)
    env.reward_sums.clear()
    prev, deltas = 0.0, []
    for k in range(min(steps, int(env.max_steps))):
        act = (np.sin(k * 0.6) * 3.0 * np.ones(env.nj)).astype(np.float32)
        _o, _r, t, tr, _i = env.step(act)
        cur = float(env.reward_sums.get(term.replace("_pay", ""), 0.0))
        deltas.append(cur - prev)
        prev = cur
        if t or tr:
            break
    d = np.asarray(deltas)
    churn, net = float(np.abs(d).sum()), abs(float(d.sum()))
    assert churn > churn_min, (
        f"{term} churned only {churn:.3f} — the term is DEAD, not unfarmable, "
        "and the net assertion below would pass vacuously")
    assert net < net_frac * churn, (
        f"{term} nets {net:.3f} of {churn:.3f} churned — a closed cycle should "
        "cancel, so this is farmable by oscillation")
    return churn, net


@pytest.mark.parametrize("term,weight", [("arm_work", 1.0), ("rise_work", 20.0)])
def test_every_signed_term_survives_the_shared_farm_test(term, weight):
    """Each signed term must pass the reusable exploit test above.

    Add a row here when you add a signed term — `tuck_work` and `rotate_work`
    had no such test and were both gamed in training.
    """
    from microduck_local.robots import asimov

    if not asimov.ready():
        pytest.skip("Asimov's model is not fetched")
    churn, net = assert_term_is_not_farmable(term, weight)
    assert churn > 0 and net >= 0


#: The seven terms a REFERENCE-TRACKING get-up replaces on purpose. The ref
#: and clip tasks score progress against an authored waypoint (`track`,
#: `reach`, `rise_work`) instead of against absolute trunk height, so the
#: base task's height-shaped terms are genuinely not wanted there. Anything
#: NOT on this list that the base charges and a subclass does not is the bug
#: this test exists for — it has now happened five times in this one file.
_TASK_TERMS_THE_REF_REPLACES = frozenset({
    "stand", "rise", "progress", "standing", "head_up", "on_back", "roll",
    # NOT regularisers that were forgotten -- two charges on the LIFTING
    # phase, excluded from the tracking tasks on purpose.
    # `arm_effort_penalty` is excluded by measurement: `quadruped` and `pike`
    # need the elbows at 58-66 % of rated torque on this body, so it charged
    # the robot for the only phase that lifts it
    # (`test_asimov_getup_ref.test_the_arm_penalty_is_gone_from_the_tracking_reward`).
    # `rise_vel_penalty` is gated on `not up`, which IS the rise. A first
    # version of `_regulariser_terms` returned both and the ref env's
    # exact-set tests caught it -- which is why this allowlist is named and
    # commented rather than inferred from whatever currently passes.
    "arm_effort_penalty", "rise_vel_penalty",
})


def test_every_getup_task_charges_the_ROBOTS_own_costs():
    """Limb speed, arm effort and over-clip commands belong to the HARDWARE.

    No curriculum makes a 326 deg/s limb or a command four times the clip
    safe, so every get-up task must sum all four. They were written inline in
    the base `_compute_reward` and nowhere else, which left every route run —
    every policy this repo has shipped for the floor — training with all four
    absent. `teach-asimov_getup_whole-c4bdba` went over its 4.0 clip on 37 %
    of steps (max 16.0) and jiggled at a median 326 deg/s while standing,
    against the shipped walker's 3 deg/s in the same posture.
    """
    import numpy as np

    from microduck_local.robots.asimov_env import (
        AsimovGetupClipEnv,
        AsimovGetupEnv,
        AsimovGetupRefEnv,
    )

    def summed(cls):
        env = cls()
        try:
            env.reset(seed=0)
            for _ in range(3):
                env.step(np.zeros(env.action_space.shape, np.float32))
            return set(env.reward_sums)
        finally:
            env.close()

    base = summed(AsimovGetupEnv)
    # A green result must not be able to mean "swept nothing".
    assert len(base) >= 25, f"base task has only {len(base)} terms"
    for cls in (AsimovGetupRefEnv, AsimovGetupClipEnv):
        missing = (base - summed(cls)) - _TASK_TERMS_THE_REF_REPLACES
        assert not missing, (
            f"{cls.__name__} does not sum {sorted(missing)}, which "
            f"{AsimovGetupEnv.__name__} charges. A knob live on the base task "
            "and summed nowhere here is recorded onto the run and reported as "
            "set while being worth exactly 0.0."
        )
