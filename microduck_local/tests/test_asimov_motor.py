"""The speed-torque envelope is openhorizon's, to the newton-metre.

Every assertion here was A/B'd against a planted regression first (see
`AGENTS.md`, "A/B new tests against planted regressions"); the planted
versions and what each one broke are named in the test bodies.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from microduck_local.robots import asimov_motor as M

pytest.importorskip("mujoco")


def _theirs():
    """openhorizon's own `Policy`, built on our own fetched MJCF."""
    pytest.importorskip("onnxruntime")
    d = M.cache_dir()
    if not (d / "run_policy_mujoco.py").is_file():
        pytest.skip(f"{M.REPO} not fetched")
    sys.path.insert(0, str(d))
    import mujoco

    import run_policy_mujoco as R

    from microduck_local.robots import asimov
    spec = json.loads(M.spec_path().read_text())
    model = R.build_model(str(asimov.xml_path()), spec["physics_dt"])
    return R, R.Policy(str(d / "policies/getup.onnx"), spec, model), \
        model, mujoco.MjData(model)


def test_the_envelope_matches_openhorizons_own_torque_function():
    """Their `Policy.torque` and our `torque_window` are the same clamp.

    Driven at speeds well past `vmax` in both directions, which is the only
    region the two could differ in.

    Planted regression: dropping the `np.clip(qd, -vmax, vmax)` line from
    `torque_window` fails this at |err| up to 143.6 N*m, because above the
    rated speed their `hi` stops falling and ours keeps going negative.
    """
    from microduck_local.robots import asimov_walk as W

    R, pol, model, data = _theirs()
    lim = M.limits("openhorizon")
    # Their arrays are in THEIR action order, ours in JOINT_NAMES order.
    order = np.array([pol.names.index(n) for n in W.JOINT_NAMES])
    inv = np.empty(order.size, int)
    inv[order] = np.arange(order.size)
    assert np.allclose(pol.effort_limit[order], lim.effort)
    assert np.allclose(pol.peak_torque[order], lim.peak)
    assert np.allclose(pol.velocity_limit[order], lim.vlim)

    kp = np.array([W.gains_for(n)[0] for n in W.JOINT_NAMES])
    kd = np.array([W.gains_for(n)[1] for n in W.JOINT_NAMES])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        qd = rng.uniform(-25.0, 25.0, order.size)
        q = rng.uniform(-1.0, 1.0, order.size)
        target = rng.uniform(-1.0, 1.0, order.size)
        data.qpos[pol.qadr] = q[inv]
        data.qvel[pol.vadr] = qd[inv]
        pol.target = target[inv]
        theirs = pol.torque(data)[order]

        lo, hi = M.torque_window(qd, lim)
        ours = np.clip(kp * (target - q) - kd * qd, lo, hi)
        worst = max(worst, float(np.abs(ours - theirs).max()))
    assert worst == pytest.approx(0.0, abs=1e-9), f"worst |err| {worst} N*m"


def test_the_window_never_inverts_at_any_speed():
    """`lo <= hi` everywhere, including past `vmax` where both collapse.

    Planted regression: raising `vmax` to `vlim * (2 + effort/peak)` inverts
    the window above the rating and fails here; MuJoCo would then silently
    clamp a torque to a range whose low end exceeds its high end.
    """
    for src in M.SOURCES:
        lim = M.limits(src)
        for v in np.linspace(-40.0, 40.0, 801):
            lo, hi = M.torque_window(np.full(lim.effort.shape, v), lim)
            assert (lo <= hi + 1e-12).all(), (src, v)


def test_zero_speed_grants_the_peak_and_the_rating_grants_nothing():
    """The two ends of the curve, which is the whole reason it exists.

    At rest the clamp is the CONTINUOUS effort (peak only matters once kd*qd
    is large); at the rated speed there is no forward torque left at all.

    Planted regression: a `1 + v/vlim` typo in `hi` makes the rated speed pay
    2x the peak instead of zero and fails the second half by ~240 N*m.
    """
    lim = M.limits("urdf")
    lo, hi = M.torque_window(np.zeros_like(lim.effort), lim)
    assert np.allclose(hi, lim.effort) and np.allclose(lo, -lim.effort)

    lo, hi = M.torque_window(lim.vlim.copy(), lim)
    assert np.allclose(hi, 0.0, atol=1e-9), hi
    # Braking is still available while overspeeding -- that is the point.
    assert (lo <= -lim.effort + 1e-9).all()


def test_our_gains_agree_with_openhorizons_published_table():
    """Two independent publications of the same actuator, cross-checked.

    Planted regression: changing `GAIN_GROUPS`' elbow kp from 40 to 41 lists
    `left_elbow_joint.kp` and `right_elbow_joint.kp` here.
    """
    if not M.spec_path().is_file():
        pytest.skip(f"{M.REPO} not fetched")
    assert M.gains_cross_check() == {}


def test_the_urdf_rating_is_what_our_servos_already_clamp_at():
    """Our `forcerange` IS the URDF effort — so "dc" changes the SPEED
    dependence and nothing else, which is what makes the A/B clean.

    Planted regression: setting hip_pitch's GAIN_GROUPS effort to 40 (the
    openhorizon number) fails this with 45.0 != 40.0.
    """
    from microduck_local.robots import asimov_walk as W
    urdf = M.urdf_limits()
    for n in W.JOINT_NAMES:
        assert W.gains_for(n)[2] == pytest.approx(urdf[n][0]), n
