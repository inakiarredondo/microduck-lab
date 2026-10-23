"""The distilled warm start for locomotion recipes.

Why it exists (README, measured): under BAM every policy trained here from
scratch either falls (18/20) or is stable and slow (0.13 m/s), while the
shipped `alpha_walking` is stable AND 0.21 m/s. Cloning it turns "discover a
stable gait from nothing" — which this sample budget cannot do — into "make
an existing gait faster", which it might. Confirmed again in this session:
two 1.5M-step from-scratch runs reached `ep_rew` 2300-2900 and **0.001 m/s**,
i.e. they learned to stand still and not fall.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.distill import default_teacher, ensure_distilled

TEACHER = default_teacher()
pytestmark = pytest.mark.skipif(
    not TEACHER.exists() or not C.MICRODUCK_RL_DIR.exists(),
    reason="upstream checkouts not found")

TINY = dict(episodes=3, epochs=2)


def test_the_clone_is_cached_and_keyed_on_teacher_and_seed(tmp_path):
    """Cloning costs a few hundred episodes plus the fit, which is pure
    overhead to pay on every launch — but a stale cache that ignored a
    swapped teacher would be worse than no cache."""
    a = ensure_distilled(seed=0, cache_dir=tmp_path, **TINY)
    assert (a / "model.zip").exists() and (a / "vecnormalize.pkl").exists()
    meta = json.loads((a / "distill.json").read_text())
    assert meta["seed"] == 0 and meta["teacher"] == str(TEACHER)
    assert meta["mse"] < 0.05, "a clone this far off is not a warm start"

    mtime = (a / "model.zip").stat().st_mtime_ns
    again = ensure_distilled(seed=0, cache_dir=tmp_path, **TINY)
    assert again == a
    assert (a / "model.zip").stat().st_mtime_ns == mtime, "cache hit must not refit"

    b = ensure_distilled(seed=1, cache_dir=tmp_path, **TINY)
    assert b != a, "a different seed is a different clone"


def test_a_missing_teacher_is_an_error_not_a_silent_scratch_start(tmp_path):
    """Silently falling back to from-scratch would hand back a policy that
    stands still, with nothing in the log to say why."""
    with pytest.raises(FileNotFoundError):
        ensure_distilled(tmp_path / "nope.onnx", cache_dir=tmp_path, **TINY)


def test_the_clone_walks_at_roughly_the_teachers_speed(tmp_path):
    """The point of the whole exercise. Scored the same way `select-run`
    scores a checkpoint: deterministic rollouts, mean body-frame forward
    velocity. The teacher reads 0.196 m/s on this harness."""
    from microduck_local.export_onnx import export
    from microduck_local.select_run import _score_one

    d = ensure_distilled(seed=0, cache_dir=tmp_path, episodes=60, epochs=20)
    onnx = d / "policy.onnx"
    export(d, onnx)
    teacher = _score_one((str(TEACHER), "teacher", 123, 3, "run", 0.4, None))
    clone = _score_one((str(onnx), "clone", 123, 3, "run", 0.4, None))
    assert teacher["speed"] > 0.15, "the teacher itself must walk, or the harness is wrong"
    assert clone["speed"] > 0.5 * teacher["speed"], (
        f"clone {clone['speed']:.3f} m/s against teacher {teacher['speed']:.3f} — "
        "the gait did not survive cloning")


def test_distill_runs_after_the_fork_and_the_run_completes(tmp_path):
    """The trap this ordering exists for: `ensure_distilled` imports torch and
    builds an env, and a torch-initialized parent DEADLOCKS the macOS fork.
    Calling it before `make_vec_env` hung four real runs at 0% CPU with only
    behavior.json written. This is the end-to-end guard."""
    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(tmp_path),
               MICRODUCK_VEC_ENV="fork", MICRODUCK_DISTILL="1")
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", "run",
         "--run-name", "d", "--envs", "4", "--steps", "6000", "--snap-steps", "3000",
         "--distill-teacher", str(TEACHER)],
        env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "warm-starting from the distilled walker" in r.stdout
    rows = [json.loads(x) for x in
            (tmp_path / "d" / "progress.jsonl").read_text().splitlines() if x.strip()]
    assert rows and rows[-1].get("done") is True, "the run must reach its end, not hang"
    assert (tmp_path / "d" / "policy.onnx").exists()


def test_a_trick_recipe_declines_the_walker_clone(tmp_path):
    """A cloned walker is the wrong prior for a headstand, and inheriting one
    silently would be worse than ignoring the flag."""
    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(tmp_path),
               MICRODUCK_VEC_ENV="fork", MICRODUCK_DISTILL="1")
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", "one_leg",
         "--run-name", "t", "--envs", "4", "--steps", "4000", "--snap-steps", "2000"],
        env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "--distill ignored" in r.stdout
    assert "warm-starting from the distilled walker" not in r.stdout
    assert not (Path(tmp_path) / ".distill").exists(), "and it must not even clone"


def _build_cache_entry(cache, q):
    """Module level on purpose: `spawn` pickles the target by qualified name,
    so a function nested in the test cannot be one."""
    try:
        from microduck_local.distill import cache_is_valid, ensure_distilled
        d = ensure_distilled(seed=0, cache_dir=Path(cache), **TINY)
        q.put(("ok", str(d), cache_is_valid(d)))
    except Exception as exc:                       # pragma: no cover
        q.put(("err", f"{type(exc).__name__}: {exc}", False))


def test_a_half_written_cache_entry_is_not_a_hit(tmp_path):
    """`model.zip` EXISTING is not the same as it being usable. A lost race
    leaves a truncated one behind, and SB3 then fails deep inside training
    with "wasn't a zip-file" — AFTER the vec-env workers have forked. It cost
    two real runs. Validating on read means the entry is rebuilt instead."""
    from microduck_local.distill import cache_is_valid

    d = tmp_path / "entry"
    d.mkdir()
    assert cache_is_valid(d) is False, "nothing there"
    (d / "model.zip").write_bytes(b"not a zip at all")
    (d / "vecnormalize.pkl").write_bytes(b"x")
    assert cache_is_valid(d) is False, "present but corrupt must not be a hit"

    good = ensure_distilled(seed=0, cache_dir=tmp_path / "c", **TINY)
    assert cache_is_valid(good) is True


def test_two_concurrent_builds_of_the_same_key_do_not_corrupt_each_other(tmp_path):
    """Both arms of a paired A/B share a cache key — same teacher, same seed —
    and launch together. Before the atomic rename they both missed, both fit
    into the same directory, and interleaved their writes."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_build_cache_entry, args=(str(tmp_path), q))
          for _ in range(2)]
    for p in ps:
        p.start()
    results = [q.get(timeout=1800) for _ in ps]
    for p in ps:
        p.join(timeout=60)

    from microduck_local.distill import cache_is_valid
    assert all(r[0] == "ok" for r in results), results
    assert results[0][1] == results[1][1], "both must end up at the same entry"
    assert all(r[2] for r in results), "and it must be loadable, not half-written"
    assert cache_is_valid(Path(results[0][1]))
    # The loser's private directory must not be left lying around as a hit.
    assert not any(p.name.startswith(".tmp-") for p in tmp_path.iterdir())


def test_fitting_budget_is_set_by_what_keeps_the_clone_upright():
    """Cloning FIDELITY is what decides whether the clone falls, and the
    default fitting budget exists to buy it. Measured over five budgets on
    the deterministic export, correlation(action MSE, fall rate) = +0.93:

        episodes  epochs   mse rad^2   falls   ep_len
             120      30     0.00030    0.75      289
             250      40     0.00017    0.08      924   <- the old default
             250     120     0.00011    0.00     1000
             600     120     0.00006    0.00     1000

    Below ~0.00011 the falling stops outright. The extra epochs cost 33 s
    once (17 -> 50) and the result is cached, so this is close to free.
    """
    from microduck_local.distill import DISTILL_EPISODES, DISTILL_EPOCHS

    assert DISTILL_EPOCHS >= 120, (
        "40 epochs reaches ~0.00017 rad^2, which measured 8% falls")
    assert DISTILL_EPISODES >= 250


@pytest.mark.slow
def test_a_default_fidelity_clone_matches_the_teacher_and_does_not_fall(tmp_path):
    """The headline: at the default budget the clone walks at the teacher's
    speed and stays up for a full episode. Measured over 8 eval seeds —
    teacher 0.197 m/s / 1000 steps / 0.00 falls, clone 0.202 / 1000 / 0.00.
    Before the budget change it was 0.238 m/s while falling 75% of episodes,
    which is what makes 'fast' the wrong thing to have optimised for."""
    from microduck_local.export_onnx import export
    from microduck_local.select_run import _score_one

    d = ensure_distilled(seed=0, cache_dir=tmp_path)      # DEFAULT budget
    onnx = d / "policy.onnx"
    export(d, onnx)
    rs = [_score_one((str(onnx), "c", sd, 3, "run", 0.4, None)) for sd in (123, 456)]
    falls = float(np.mean([r["falls"] for r in rs]))
    speed = float(np.mean([r["speed"] for r in rs]))
    teach = [_score_one((str(TEACHER), "t", sd, 3, "run", 0.4, None)) for sd in (123, 456)]
    t_speed = float(np.mean([r["speed"] for r in teach]))
    assert falls == 0.0, f"the clone fell {falls:.0%} of episodes at the default budget"
    assert speed > 0.8 * t_speed, f"clone {speed:.3f} against teacher {t_speed:.3f}"


def test_a_teachers_dead_observation_slot_is_refused_before_training():
    """A slot the teacher's data never moved is a bug the LOSS CANNOT SEE.

    `scripts/distill_getup.py` stepped the physics by hand and never wrote
    `env.last_action`, so all 23 of `obs[55:78]` were exactly 0.0 in all
    104 000 samples. The fit converged to 0.000228 rad^2 — an 11.6x improvement
    over the previous arm, and the best clone fidelity this robot has had — and
    the student stood **23/24** when driven with those slots still zeroed. Fed
    the real previous action, which is what the deployment contract puts there,
    it stood **0/24**. The loss could not see it because the loss was computed
    on the same dead data.

    `twist_cmd` is legitimately flat for a get-up (nothing is commanded), so
    the caller declares the slots it MEANS to leave dead and everything else
    raises.
    """
    from microduck_local.distill import dead_slots

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(400, 78))
    obs[:, 6:9] = 0.0                                   # twist_cmd, on purpose
    assert dead_slots(obs, "asimov1", ("twist_cmd",)) == []

    obs[:, 55:78] = 0.0                                 # THE BUG
    with pytest.raises(ValueError, match=r"last_action \(obs\[55:78\]"):
        dead_slots(obs, "asimov1", ("twist_cmd",))

    # And an undeclared flat slot is not excused by declaring a DIFFERENT one.
    with pytest.raises(ValueError, match="twist_cmd"):
        dead_slots(obs, "asimov1", ("last_action",))


def _repo_runs():
    """`runs/` of the real checkout, not a tmp_path a fixture redirected to."""
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "runs"


def test_a_distilled_student_must_be_driven_through_its_normalizer():
    """`PPO.predict(obs)` on a distilled student is WRONG, and silently so.

    `distill.fit` trains the student on the VecNormalize view and leaves the
    stats in `vecnormalize.pkl` for `export_onnx` to bake in later. Drive the
    checkpoint directly and the net gets raw observations. Measured on
    teach-asimov_getup_clip-0e31e5, which `verify_run` scores at 34/36 rose
    and 26/36 still up:

        PPO.predict on raw obs                     stood 0/8
        policy.onnx (normaliser baked in)          stood 8/8
        PPO.predict + VecNormalize stats by hand   stood 8/8

    `scripts/distill_getup.py` drove its DAgger rollouts the first way, so
    every student state it ever aggregated came from a policy that cannot
    stand — which is the likeliest explanation for plain DAgger "hurting"
    (aggregate MSE 0.000176 -> 0.000470, rise 24/24 -> 20/24) and means DAgger
    had never actually been tested on this task.

    This pins the property rather than the rollout: the stats are real, and a
    student driver that applies them does not agree with one that does not.
    """
    import pickle

    import numpy as np

    run = _repo_runs() / "teach-asimov_getup_clip-0e31e5"
    if not (run / "vecnormalize.pkl").is_file() or not (run / "model.zip").is_file():
        pytest.skip("the reference clone run is not on this machine")

    with open(run / "vecnormalize.pkl", "rb") as fh:
        vn = pickle.load(fh)
    mu = np.asarray(vn.obs_rms.mean, np.float64)
    va = np.asarray(vn.obs_rms.var, np.float64)

    # if the stats were identity the distinction would not matter — and a
    # dead slot would make them identity on that slot, which `dead_slots`
    # already raises on. Assert they are real.
    assert np.abs(mu).max() > 1e-3, "normalizer mean is all but zero"
    assert va.min() > 0.0 and np.ptp(va) > 1e-6, (
        "normalizer variance is flat — nothing is actually being normalised")

    from stable_baselines3 import PPO

    net = PPO.load(str(run / "model"), device="cpu")
    rng = np.random.default_rng(0)
    # observations drawn where the policy actually lives, not N(0,1)
    sample = (mu + np.sqrt(va) * rng.normal(0, 1, size=(16, len(mu)))
              ).astype(np.float32)

    raw = np.stack([net.predict(o, deterministic=True)[0] for o in sample])
    nrm = np.stack([
        net.predict(np.clip((o - mu) / np.sqrt(va + vn.epsilon),
                            -vn.clip_obs, vn.clip_obs).astype(np.float32),
                    deterministic=True)[0]
        for o in sample])
    gap = float(np.abs(raw - nrm).max())
    assert gap > 1e-3, (
        f"raw and normalised driving agree to {gap:.2e} — either the stats "
        "stopped being applied or this policy no longer depends on them, and "
        "in both cases the guard above is no longer testing anything")
