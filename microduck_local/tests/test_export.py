"""End-to-end: train a few PPO steps, export ONNX, verify the baked normalizer
and that the artifact matches SB3's deterministic prediction bit-for-bit-ish."""

import numpy as np
import onnxruntime as ort
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from microduck_local import contract as C
from microduck_local.export_onnx import OnnxWalkPolicy
from microduck_local.walk_env import MicroduckWalkEnv


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    venv = DummyVecEnv([lambda: MicroduckWalkEnv(seed=0)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100.0)
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, n_epochs=1, device="cpu")
    model.learn(total_timesteps=64)
    model.save(str(out / "model"))
    venv.save(str(out / "vecnormalize.pkl"))
    return out


def test_export_matches_sb3_prediction(tiny_run):
    from microduck_local.export_onnx import export
    onnx_path = export(tiny_run, tiny_run / "policy.onnx")  # runs its own allclose check

    sess = ort.InferenceSession(str(onnx_path))
    i, o = sess.get_inputs()[0], sess.get_outputs()[0]
    assert i.name == "obs" and list(i.shape) == [1, C.OBS_DIM]
    assert o.name == "actions" and list(o.shape) == [1, C.NUM_JOINTS]

    # Against model.predict(deterministic=True) with external normalization —
    # the exported graph must reproduce train-time behavior end to end.
    model = PPO.load(str(tiny_run / "model"), device="cpu")
    import pickle
    with open(tiny_run / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    rng = np.random.default_rng(7)
    raw = rng.normal(0, 0.5, (1, C.OBS_DIM)).astype(np.float32)
    norm = np.clip((raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8),
                   -vn.clip_obs, vn.clip_obs).astype(np.float32)
    want, _ = model.predict(norm, deterministic=True)
    got = sess.run(["actions"], {"obs": raw})[0]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_unnormalized_would_differ(tiny_run):
    """The reason baking matters: raw-vs-normalized obs give different actions."""
    model = PPO.load(str(tiny_run / "model"), device="cpu")
    import pickle
    with open(tiny_run / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    wrapper = OnnxWalkPolicy(model.policy, vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs).eval()
    ident = OnnxWalkPolicy(model.policy, np.zeros(C.OBS_DIM), np.ones(C.OBS_DIM), 100.0).eval()
    obs = torch.tensor(np.random.default_rng(3).normal(0, 0.5, (1, C.OBS_DIM)), dtype=torch.float32)
    with torch.no_grad():
        assert not torch.allclose(wrapper(obs), ident(obs))


def test_a_slot_that_never_moved_is_not_amplified_7071x():
    """`distill` writes `obs_rms.var = obs.var(axis=0) + 1e-8`, so a slot that
    was CONSTANT while the policy trained lands at exactly 1e-8 — and
    `sqrt(1e-8 + 1e-8)` is a 7071x gain baked into the exported graph.

    MEASURED 2026-10-04 on a cloned Asimov get-up: 14 of its 23 `last_action`
    slots are joints no waypoint ever moves, so their action was 0 in every
    collected sample. The clone's mean |da| per joint per control step was
    **1.15** against its own teacher's **0.001**, `action_rate_penalty` charged
    it -277, and a 1.2 M-step fine-tune correctly threw the warm start away.
    Three teacher designs were rewritten chasing a fault that was in the
    exporter. Re-exporting the SAME weights with dead slots passed through took
    it to 0.002 and -2.5.

    `train.py` had this guard and tested `var < 1e-8` against the value
    `distill` writes, so it matched 0 of the 16 slots it was written for.
    """
    from microduck_local.export_onnx import DEAD_OBS_VAR, OnnxWalkPolicy

    venv = DummyVecEnv([lambda: MicroduckWalkEnv(seed=0)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100.0)
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, n_epochs=1, device="cpu")
    n = int(venv.observation_space.shape[0])

    var = np.full(n, 0.25)
    mean = np.zeros(n)
    var[3] = DEAD_OBS_VAR          # a slot that never moved, as distill writes it
    var[4] = DEAD_OBS_VAR
    mean[4] = 0.7                  # ...and one that was constant at a NON-zero value
    wrapped = OnnxWalkPolicy(model.policy, mean, var, 100.0)

    std = wrapped.obs_std.detach().numpy()
    assert std[3] == pytest.approx(1.0, abs=1e-4), (
        f"a dead slot still carries a {1.0 / std[3]:.0f}x gain")
    assert std[4] == pytest.approx(1.0, abs=1e-4)
    assert wrapped.obs_mean.detach().numpy()[4] == 0.0, "a dead slot's mean is dropped"
    # the live slots are untouched, so no policy that was working changes
    live = [i for i in range(n) if i not in (3, 4)]
    assert np.allclose(std[live], np.sqrt(0.25 + 1e-8), atol=1e-6)

    # and the regression this replaces: WITHOUT the fix the gain is ~7071
    assert 1.0 / float(np.sqrt(DEAD_OBS_VAR + 1e-8)) > 7000


def test_the_warm_start_guard_fires_on_the_value_distill_writes():
    """The producer and the detector have to agree on the constant, and they did
    not: `distill` writes `obs.var(axis=0) + 1e-8` and `train.py` tested
    `var < 1e-8`, which that value can never satisfy. 0 of 16 slots caught.

    This pins the arithmetic rather than the spelling: a column that never moved
    lands EXACTLY on the floor, so the comparison has to include the boundary.
    """
    from microduck_local.export_onnx import DEAD_OBS_VAR

    # a collected observation where column 2 never moved — distill's own line
    obs = np.tile(np.array([0.3, -1.2, 0.5, 2.0]), (64, 1))
    obs[:, 0] += np.linspace(-1, 1, 64)
    obs[:, 3] += np.linspace(-0.5, 0.5, 64)
    var = obs.var(axis=0).astype(np.float64) + 1e-8

    assert var[1] == DEAD_OBS_VAR, "the floor moved; this test's premise is stale"
    assert var[2] == DEAD_OBS_VAR
    caught_old = np.flatnonzero(var < DEAD_OBS_VAR)          # the shipped bug
    caught_new = np.flatnonzero(var <= DEAD_OBS_VAR)         # the fix
    assert caught_old.size == 0, "a strict `<` was supposed to catch nothing"
    assert set(caught_new.tolist()) == {1, 2}
    assert var[0] > DEAD_OBS_VAR and var[3] > DEAD_OBS_VAR   # live slots excluded
