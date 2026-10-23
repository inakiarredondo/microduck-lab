"""The mirror in `robots/asimov_walk.py` still matches Menlo's own env.yaml.

Every constant in that module is a COPY of something published in
`Menlo/asimov1-locomotion-0818`'s `env.yaml`, mirrored rather than parsed at
import for `contract.py`'s reason: a deployment contract has to be readable in
one place, and a 62 kB YAML full of `!!python/object/apply` tags is not it.

This is what keeps the copy honest. It re-reads the fetched artifact and
compares, so a revision of the policy that moved a gain, re-ordered a slot or
changed the action scale fails HERE — loudly, next to the numbers — instead of
silently making every measurement in this repo describe a different robot.

Skipped, not failed, when the assets are not downloaded: `AGENTS.md`, "A test
gated on optional assets SKIPS exactly where it would fail".
"""

from __future__ import annotations

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

from microduck_local.robots import asimov_walk as W  # noqa: E402

pytestmark = pytest.mark.skipif(
    not W.policy_ready(),
    reason="Asimov's shipped policy is not fetched "
           "(uv run python -m microduck_local.robots.asimov_walk --fetch)")


class _Loader(yaml.SafeLoader):
    """Isaac writes `!!python/object/apply:builtins.slice` into its dumps, and
    a safe loader refuses those. Unknown tags become None — none of them are
    values this contract reads."""


_Loader.add_multi_constructor("", lambda loader, suffix, node: None)


@pytest.fixture(scope="module")
def env_yaml() -> dict:
    with open(W.policy_env_yaml()) as f:
        return yaml.load(f, Loader=_Loader)


def test_the_clock_is_the_one_the_policy_was_trained_at(env_yaml):
    assert float(env_yaml["sim"]["dt"]) == W.PHYSICS_DT
    assert int(env_yaml["decimation"]) == W.DECIMATION
    assert W.CTRL_DT == pytest.approx(0.02)


def test_the_action_is_still_a_default_offset_at_one_scale(env_yaml):
    """`use_default_offset` is what makes the action an OFFSET from the
    default pose rather than an absolute target — a silent flip there would
    drive every joint to `0.25 * action` and look like a policy that collapses."""
    act = env_yaml["actions"]["joint_pos"]
    assert act["use_default_offset"] is True
    assert float(act["scale"]) == W.ACTION_SCALE


def test_every_servo_gain_matches_the_published_actuator_block(env_yaml):
    """kp, kd, torque limit, armature and joint friction, per group.

    The armature especially: Menlo's own asset README says a mismatch between
    the training and deployment models makes the robot fall immediately,
    because the dynamics are simply different.
    """
    published = env_yaml["scene"]["robot"]["actuators"]
    seen = set()
    for group in published.values():
        for expr in group["joint_names_expr"]:
            frag = expr.replace(".*_", "").replace("_joint", "")
            for name in W.JOINT_NAMES:
                if frag not in name:
                    continue
                seen.add(name)
                kp, kd, effort, armature, friction = W.gains_for(name)
                assert kp == float(group["stiffness"]), name
                assert kd == float(group["damping"]), name
                assert effort == float(group["effort_limit"]), name
                assert armature == pytest.approx(float(group["armature"])), name
                assert friction == pytest.approx(float(group["friction"])), name
    assert seen == set(W.JOINT_NAMES), (
        f"not every joint was covered by the published actuator block: "
        f"{sorted(set(W.JOINT_NAMES) - seen)}")


def test_the_default_pose_matches_the_published_init_state(env_yaml):
    """Resolved through the regexes the YAML writes it with (`.*_hip_roll_joint`)."""
    import re

    published = env_yaml["scene"]["robot"]["init_state"]["joint_pos"]
    for i, name in enumerate(W.JOINT_NAMES):
        hit = [v for pattern, v in published.items()
               if re.fullmatch(pattern.replace(".*", ".*"), name)]
        assert hit, f"{name} has no published default"
        assert float(W.DEFAULT_POSE[i]) == pytest.approx(float(hit[0])), name


def test_the_observation_order_and_every_slot_match_the_published_group(env_yaml):
    """The one that matters most: 78 floats in their order, and the three
    CAN-bus slots joint for joint. A permutation here still produces actions
    and still moves the robot — it just falls over."""
    policy = env_yaml["observations"]["policy"]
    terms = [k for k, v in policy.items()
             if isinstance(v, dict) and "func" in v]
    assert terms == ["base_ang_vel", "projected_gravity", "command",
                     "joint_pos_slot01", "joint_pos_slot23", "joint_pos_slot45",
                     "joint_vel_slot01", "joint_vel_slot23", "joint_vel_slot45",
                     "actions"], terms

    assert float(policy["base_ang_vel"]["scale"]) == W.ANG_VEL_SCALE
    for term in ("joint_vel_slot01", "joint_vel_slot23", "joint_vel_slot45"):
        assert float(policy[term]["scale"]) == W.JOINT_VEL_SCALE, term

    for term, mine in (("joint_pos_slot01", W.SLOT_01),
                       ("joint_pos_slot23", W.SLOT_23),
                       ("joint_pos_slot45", W.SLOT_45)):
        published = tuple(policy[term]["params"]["asset_cfg"]["joint_names"])
        assert published == mine, term
        # The velocity half must be the SAME order, or the two halves of one
        # joint's state land in different positions.
        vel = tuple(policy[term.replace("pos", "vel")]
                    ["params"]["asset_cfg"]["joint_names"])
        assert vel == mine, term

    # NOT `widths.get(t, len(...))`: a dict default is evaluated eagerly, so
    # the fallback would index `asset_cfg` on the three terms that have none.
    widths = {"base_ang_vel": 3, "projected_gravity": 3, "command": 3,
              "actions": W.NUM_JOINTS}
    total = sum(
        widths[t] if t in widths
        else len(policy[t]["params"]["asset_cfg"]["joint_names"])
        for t in terms)
    assert total == W.OBS_DIM


def test_the_slot_order_covers_every_joint_exactly_once():
    """Not from the YAML — the arithmetic our `_get_obs` depends on."""
    order = W.slot_order()
    assert sorted(order) == list(range(W.NUM_JOINTS))
    assert len(order) == W.NUM_JOINTS


def test_the_joint_order_is_the_models_own():
    """The action order is Isaac's articulation order and the model's is the
    MJCF tree's. They agree today; they are not required to, so it is checked
    rather than assumed."""
    model = W.walk_model(True, True)
    tree = tuple(model.joint(j).name for j in range(model.njnt)
                 if model.jnt_type[j] == 3)       # mjJNT_HINGE
    assert tree == W.JOINT_NAMES


def test_the_contract_the_body_publishes_adds_up():
    spec = W.spec()
    contract = spec.contract()
    assert contract.obs_dim == W.OBS_DIM
    assert contract.act_dim == W.NUM_JOINTS
    assert contract.rate_hz == pytest.approx(50.0)
    assert [s.width for s in contract.slots] == [3, 3, 3, 23, 23, 23]
    assert sum(s.width for s in contract.slots) == W.OBS_DIM


def test_the_shipped_policy_takes_exactly_this_contracts_observation():
    """The graph itself, not a record of it."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(W.policy_onnx()),
                                providers=["CPUExecutionProvider"])
    assert int(sess.get_inputs()[0].shape[-1]) == W.OBS_DIM
    out = sess.run(None, {sess.get_inputs()[0].name:
                          np.zeros((1, W.OBS_DIM), np.float32)})[0]
    assert out.reshape(-1).shape == (W.NUM_JOINTS,)
