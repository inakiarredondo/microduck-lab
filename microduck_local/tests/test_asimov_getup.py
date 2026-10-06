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

import os
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
                  "head_up": E.W_HEAD_UP, "start_rung": ""}


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
    from microduck_local.robots import asimov, asimov_getup as G

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
