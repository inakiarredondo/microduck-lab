"""The 78-float adapter IS their 375-input policy, action for action.

The cross-checks in `term_slices` prove the layouts agree; this proves the
adapter built from them reproduces their own runner's output, which is the
only thing that rules out a wrong permutation. "A wrong permutation still
produces actions and still moves the robot; it just falls over."
"""
from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from microduck_local.robots import asimov_motor as M
from microduck_local.robots import asimov_openhorizon as OH


def _skip_unless_fetched():
    if not (M.cache_dir() / "run_policy_mujoco.py").is_file():
        pytest.skip(f"{M.REPO} not fetched")


def test_the_adapter_reproduces_their_runners_actions():
    """Same observation values, same actions — over a whole get-up.

    The observation is assembled by placing THEIR OWN `_term` values into our
    78-float slots, which isolates what this adapter is responsible for (the
    slot layout, the term order, the 5-frame ring) from what it is not (when
    in the substep cycle the observation is sampled -- see
    `test_our_env_samples_the_gyro_one_substep_fresher_than_their_runner`).
    If any slice, joint order or history slot were wrong the actions would
    diverge immediately.

    Planted regressions, each confirmed to fail this:
      * swapping the `joint_pos` and `joint_vel` slices
      * filling the first history with zeros instead of the first value
      * `buf.append` without `buf.pop(0)` (the ring stops advancing)
      * reversing the per-term history to newest->oldest
    """
    _skip_unless_fetched()
    sys.path.insert(0, str(M.cache_dir()))
    import mujoco

    import run_policy_mujoco as R

    from microduck_local.robots import asimov

    spec = json.loads(M.spec_path().read_text())
    scales = {t_["name"]: float(t_["scale"])
              for t_ in spec["observation"]["terms"]}
    model = R.build_model(str(asimov.xml_path()), spec["physics_dt"])
    d = mujoco.MjData(model)
    theirs = R.Policy(str(OH.onnx_path()), spec, model)
    decim = round(1.0 / spec["rate_hz"] / model.opt.timestep)
    ours = OH.OpenHorizonPolicy("getup")
    # ASSEMBLED FROM `asimov_walk`'s OWN SLICE CONSTANTS, deliberately not
    # from `OH.term_slices()`. Using the adapter's own mapping to build the
    # input it is being checked against makes the test self-consistent with
    # any permutation bug in that mapping: planted, a joint_pos/joint_vel
    # swap passed, because the test wrote the terms into the same wrong slots
    # the adapter then read them back out of. "A test that seeds the value it
    # checks proves nothing."
    from microduck_local.robots import asimov_walk as W
    slots = (("base_ang_vel", *W.OBS_ANG_VEL),
             ("projected_gravity", *W.OBS_GRAVITY),
             ("joint_pos", *W.OBS_JOINT_POS),
             ("joint_vel", *W.OBS_JOINT_VEL),
             ("actions", *W.OBS_LAST_ACTION))

    rng = np.random.default_rng(0)
    R.random_fallen_state(model, d, theirs, rng)
    mujoco.mj_forward(model, d)

    worst = 0.0
    for _k in range(150):
        obs78 = np.zeros(78, np.float32)
        for name, a, b in slots:
            obs78[a:b] = theirs._term(name, d) * scales[name]
        a_ours = ours(obs78)
        theirs.step(d)
        worst = max(worst, float(np.abs(a_ours - theirs.last).max()))
        R._physics(model, d, theirs, decim, None, None)

    assert worst < 1e-5, f"worst |action difference| {worst}"


def test_our_env_samples_the_gyro_one_substep_fresher_than_their_runner():
    """A real, measured difference between the two harnesses — not a bug here.

    `walk_env._refresh_derived` exists because `mj_step` computes kinematics
    and sensors BEFORE it integrates, so at the end of a substep loop
    `xpos`/`xquat`/`sensordata` describe the state 5 ms behind `qpos`/`qvel`.
    Our env refreshes them; their runner reads `d.sensordata` as it stands.
    So a policy on our stage sees a gyro and a gravity vector one substep
    FRESHER than the reference implementation feeds it.

    The joint blocks are unaffected (they are read from `qpos`/`qvel`
    directly), which is why this is pinned separately rather than folded into
    the fidelity test: it bounds the only part of the observation the two
    harnesses disagree about, so a later "their policy scores differently
    here" has a measured candidate instead of a guess.
    """
    _skip_unless_fetched()
    sys.path.insert(0, str(M.cache_dir()))
    import mujoco

    import run_policy_mujoco as R

    from microduck_local.robots import asimov
    from microduck_local.robots.asimov_env import AsimovGetupEnv

    spec = json.loads(M.spec_path().read_text())
    model = R.build_model(str(asimov.xml_path()), spec["physics_dt"])
    d = mujoco.MjData(model)
    theirs = R.Policy(str(OH.onnx_path()), spec, model)
    decim = round(1.0 / spec["rate_hz"] / model.opt.timestep)
    env = AsimovGetupEnv(domain_rand=False, obs_noise=False,
                         getup_label="prone", actuator_force="xml")
    env.reset(seed=0)

    rng = np.random.default_rng(0)
    R.random_fallen_state(model, d, theirs, rng)
    mujoco.mj_forward(model, d)
    gyro = grav = joints = 0.0
    for _k in range(60):
        theirs.step(d)
        R._physics(model, d, theirs, decim, None, None)
        env.data.qpos[:] = d.qpos
        env.data.qvel[:] = d.qvel
        env.last_action = theirs.last.astype(np.float32).copy()
        mujoco.mj_forward(env.model, env.data)
        env._refresh_derived()
        o = env._get_obs()
        gyro = max(gyro, float(np.abs(
            o[0:3] - theirs._term("base_ang_vel", d) * 0.25).max()))
        grav = max(grav, float(np.abs(
            o[3:6] - theirs._term("projected_gravity", d)).max()))
        joints = max(joints, float(np.abs(
            o[9:32] - theirs._term("joint_pos", d)).max()))

    # The joint blocks come from qpos/qvel, which both harnesses read at the
    # same instant: identical to float precision.
    assert joints < 1e-6, joints
    # The IMU channels differ, and by enough to matter -- so this is stated
    # rather than assumed negligible. ("Before writing 'negligible': name the
    # case it is negligible FOR.")
    assert gyro > 1e-3, (
        "the two harnesses now agree on the gyro — if their runner started "
        "refreshing, this test is the thing that is out of date")
    assert gyro < 0.5 and grav < 0.2, (gyro, grav)


def test_the_adapter_declares_what_it_CONSUMES_not_what_the_graph_takes():
    """78, so the lab's contract check compares the right two numbers.

    `policy_refusal` compares `infer.obs_dim` against the env's observation
    width. Reporting the graph's 375 would have this policy refused by every
    Asimov env in the repo; reporting nothing would let it load against a
    61-float duck.

    Planted regression: deleting `obs_dim` makes `getattr(infer, 'obs_dim',
    None)` return None and the refusal check silently stops checking.
    """
    _skip_unless_fetched()
    p = OH.OpenHorizonPolicy("getup")
    assert p.obs_dim == 78
    with pytest.raises(ValueError, match="78-float"):
        p(np.zeros(375, np.float32))


def test_reset_drops_the_history_so_two_episodes_do_not_bleed():
    """A ring carried across a reset feeds frames from the previous episode.

    Planted regression: making `reset()` a no-op makes the two actions below
    differ by up to 0.43 — the policy answering a velocity transient that
    happened in an episode that is over.
    """
    _skip_unless_fetched()
    p = OH.OpenHorizonPolicy("getup")
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, 78).astype(np.float32)
    b = rng.normal(0, 0.3, 78).astype(np.float32)

    first = p(a)
    p(b), p(b), p(b)          # fill the ring with something else
    p.reset()
    assert np.allclose(p(a), first, atol=0), "reset did not clear the history"


def test_the_lab_seat_keeps_the_actuator_the_palette_ENTRY_declares():
    """A stage that disagrees with its own label is the fault to prevent.

    `Duck._make_env` assigned `actuator_force = "xml"` unconditionally for
    every non-duck body, which silently overrode this entry's `"dc"`: the
    roster row and the palette note both said "inside the published
    speed-torque envelope" while the physics on the stage was the flat clamp.

    It also has to take a PRIVATE mjModel. `DcMotorClip` rewrites
    `model.actuator_forcerange` every substep, and the lab shares one compiled
    model across the roster — `walk_env` refuses a shared model for exactly
    this, but it cannot SEE the sharing when the model is passed in
    explicitly, so the scope decision has to be made in the lab.

    Planted regressions:
      * restoring `kw["actuator_force"] = "xml"` unconditionally -> actuator
        is "xml" and `_motor` is None
      * dropping `or actuator == "dc"` from `writes_model` -> the env is built
        inside `shared_model_scope(exclusive=False)`; with two such ducks the
        second one's clamp overwrites the first's
    """
    _skip_unless_fetched()
    from microduck_local import viz_server as V
    from microduck_local.robots import spec as S

    entry = next(e for e in S.get("asimov1").shipped_policies()
                 if e.get("adapter") == "openhorizon:getup")
    duck = V.Duck("d0", entry["label"], OH.OpenHorizonPolicy("getup"),
                  seed=0, robot="asimov1",
                  env_kwargs=dict(entry["envKwargs"]))
    assert duck.env.actuator_model == "dc", (
        "the lab overrode the entry's actuator: the stage would show the flat "
        f"clamp while the label says otherwise (got {duck.env.actuator_model!r})")
    assert duck.env._motor is not None, "declared dc but built no motor model"
    # Private, not the roster's shared model: a second dc body must not be
    # able to clamp this one's torques.
    assert duck.env._model_shared is False
    other = V.Duck("d1", entry["label"], OH.OpenHorizonPolicy("getup"),
                   seed=1, robot="asimov1",
                   env_kwargs=dict(entry["envKwargs"]))
    assert duck.env.model is not other.env.model, (
        "two dc bodies share one mjModel — each rewrites actuator_forcerange "
        "from its own qvel every substep, so they would clamp each other")


def test_a_FETCHED_policys_env_survives_a_lab_restart():
    """`restore_ducks` must resolve a palette entry for ANY policy id.

    It looked one up only when the id started with `run:`, so a fetched
    policy — which has no `onnxPath` and no run directory — resolved to
    `run_path = None`, `env_kwargs_for_policy_path(None)` returned `{}`, and
    `slot_env` fell back to the body's `default_task` with the default
    actuator. openhorizon's get-up was spawned as `dc` and came back as `xml`
    after a restart. A FRESH spawn was always correct, which is precisely what
    made it invisible — the bug needed a restart to appear.

    Planted regression: restoring the `startswith("run:")` guard makes
    `env_kwargs_for_policy_path` return `{}` here instead of the entry's
    kwargs, and the assertion on `actuator_force` fails.
    """
    _skip_unless_fetched()
    from microduck_local import viz_server as V
    from microduck_local.robots import spec as S

    entry = next(e for e in S.get("asimov1").shipped_policies()
                 if e.get("adapter") == "openhorizon:getup")
    # EXACTLY what `save_lab_state` writes for a spawned shipped duck: the
    # policy id, and no `onnxPath` because there is no run artifact. Going
    # through `_restore_run_path` is the point — handing the test
    # `entry["path"]` directly would pass with the bug still in place, which
    # is how the first version of this test proved nothing.
    saved = {"id": "d0", "policy": entry["id"], "robot": "asimov1"}
    run_path = V._restore_run_path(saved)
    assert run_path, "a saved fetched policy resolved to no file at all"
    kw = V.env_kwargs_for_policy_path(run_path)
    assert kw.get("actuator_force") == "dc", (
        "a fetched policy's declared env did not survive a restart: "
        f"got {kw!r}")
    assert kw.get("task") == "getup", kw


def test_a_STACKED_arm_can_be_watched_but_still_cannot_hot_swap():
    """`policy_refusal` must admit a 78*N policy into a 78*N env, and nothing
    else.

    It compared the policy's contract against the BODY's canonical one, so
    every N-frame policy was refused everywhere — the Asimov get-up that
    finally rises inside the published actuator envelope reads 390 floats, and
    the lab silently declined to put it on the stage. That breaks the rule this
    repo cares most about ("if it moves the robot, Jonathan can see it") in the
    one case where there was finally something to see.

    It is an exception, not a loosening: the width check still has to pass, so
    it cannot admit a policy the env cannot feed.

    Planted regression (confirmed to fail this): dropping the `stacked`
    exception -> the first assertion fails, because the policy is refused in
    an env that fits it exactly.

    Deliberately NOT claimed: removing `int(want.obs_dim) == int(have_obs)`
    from the `stacked` condition was planted and this test still passed, so
    that term is redundant — the function's second rung (the plain width
    check) is what stops a 390-input policy entering a 78-float env. The
    second assertion below therefore guards the FUNCTION's behaviour, not that
    particular clause.
    """
    from microduck_local.lab import robots as lab_robots
    from microduck_local.robots import policy_contract as PC

    base = lab_robots._registry.get("asimov1").contract()
    # Built the way `export_onnx` builds it, slot table included: the contract
    # validates that its slots cover `obs_dim`, so a bare `replace` of the
    # width raises and would make this test about the dataclass.
    n = 5
    wide_slots = tuple(
        PC.Slot(f"t-{n - 1 - f}/{sl.name}",
                f * base.obs_dim + sl.start, f * base.obs_dim + sl.stop)
        for f in range(n) for sl in base.slots)
    stacked = dataclasses.replace(
        base, id=f"asimov1-{base.obs_dim * n}-hist{n}-control",
        obs_dim=base.obs_dim * n, slots=wide_slots)

    # Patched rather than exported: this is about the COMPARISON, and minting a
    # real 390-input ONNX here would make the test about the exporter.
    orig = lab_robots.recorded
    lab_robots.recorded = lambda _p: stacked
    try:
        assert lab_robots.policy_refusal(
            "x.onnx", "asimov1", stacked.obs_dim, stacked.obs_dim) is None, (
            "a stacked arm was refused in an env that produces exactly its "
            "width — it can then never be watched")
        why = lab_robots.policy_refusal(
            "x.onnx", "asimov1", stacked.obs_dim, base.obs_dim)
        assert why, "a 390-input policy was admitted into a 78-float env"
        # ...and a genuinely foreign contract is still refused.
        foreign = dataclasses.replace(
            base, id="microduck-61-v0", obs_dim=61,
            slots=(PC.Slot("everything", 0, 61),))
        lab_robots.recorded = lambda _p: foreign
        assert lab_robots.policy_refusal("x.onnx", "asimov1", 61, 61), (
            "a duck contract was admitted onto an Asimov")
    finally:
        lab_robots.recorded = orig
