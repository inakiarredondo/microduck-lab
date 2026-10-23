"""record-world: the world-mode recorder runs a builtin scenario headlessly,
under the lab's own WorldState, and leaves the three files an agent reads."""

import os
import platform
import sys

import pytest

from microduck_local import record_world


def _offscreen_gl() -> str | None:
    """Why this machine cannot render offscreen, or None if it can. The
    recorder builds a `mujoco.Renderer`, and on a headless runner that is
    not a test failure: macOS raises CGLError("invalid pixel format"), and
    Linux with no display and no MUJOCO_GL ABORTS the interpreter from
    inside GLFW (2026-09-10, both CI runners) - so the Linux check must
    run before mujoco is asked for a context at all."""
    if platform.system() == "Linux" and not os.environ.get("DISPLAY") \
            and os.environ.get("MUJOCO_GL", "") not in ("egl", "osmesa"):
        return "Linux with no DISPLAY and MUJOCO_GL unset (set MUJOCO_GL=egl for offscreen)"
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>')
        r = mujoco.Renderer(m, 16, 16)
        r.update_scene(mujoco.MjData(m))
        r.render()
        r.close()
    except Exception as e:  # noqa: BLE001 - any GL failure means "no offscreen context here"
        return f"no offscreen GL context: {type(e).__name__}: {e}"
    return None


_NO_GL = _offscreen_gl()
pytestmark = pytest.mark.skipif(_NO_GL is not None, reason=f"record-world renders video; {_NO_GL}")


@pytest.mark.parametrize("scenario,camera,brain", [
    ("living-room", "top", []),
    ("playroom", "follow:d0", ["d0=tidy"]),
])
def test_record_world_writes_video_sheet_and_events(tmp_path, monkeypatch, scenario, camera, brain):
    out = tmp_path / "rw"
    argv = ["record-world", scenario, "--seconds", "1.0", "--out", str(out), "--camera", camera,
            "--width", "160", "--height", "120", "--sheet-frames", "4", "--stride", "10"]
    for b in brain:
        argv += ["--brain", b]
    monkeypatch.setattr(sys, "argv", argv)
    record_world.main()
    assert (out / "world.mp4").stat().st_size > 0
    assert (out / "sheet.png").stat().st_size > 0
    events = (out / "events.txt").read_text()
    assert events.startswith(f"# {scenario} seed 0")
    # the first tick is a transition into each brain's initial state
    assert "d0 ->" in events
    if brain:
        assert "d0=tidy" in events.splitlines()[0]


def test_a_follow_shot_is_framed_on_WHAT_IT_FOLLOWS():
    """`FOLLOW_DISTANCE` is a 0.277 m duck's 1.1 m, and at it a 1.296 m Asimov
    fills six tiles with its shins — every non-duck body this recorder has
    ever followed was filmed that way, and this is the tool the `record-world`
    skill says to read when a robot "does something weird".

    Measured off the SUBJECT, so the duck's shot is unchanged to the float and
    a body that cannot be measured still gets it.
    """
    from microduck_local.record_world import follow_scale
    from microduck_local.world_server import resolve_scenario

    duck = resolve_scenario("living-room")                  # d0 is a duck
    assert follow_scale(duck, "d0") == 1.0

    walkabout = resolve_scenario("asimov-walkabout")        # p0 is an Asimov
    assert follow_scale(walkabout, "p0") == pytest.approx(1.296 / 0.2772, abs=0.02)
    assert record_world.FOLLOW_DISTANCE * follow_scale(walkabout, "p0") > 1.296, (
        "the camera is closer to the robot than the robot is tall")

    mars = resolve_scenario("mars-follow")                  # d0 is a MARS
    assert 1.4 < follow_scale(mars, "d0") < 1.8

    # A mocap capsule has no model to measure; the scenario declares its size.
    capsule = next((sc for sc in (resolve_scenario(n) for n in ("follow-me",))
                    for p in sc.persons if p.kind == "capsule"), None)
    if capsule is not None:
        p = next(p for p in capsule.persons if p.kind == "capsule")
        assert follow_scale(capsule, p.id) == pytest.approx(p.height / 0.2772, abs=0.02)

    # Nothing to follow by that name: the shot it always had, not a crash —
    # `make_camera` is what refuses an unknown id, with the list of real ones.
    assert follow_scale(duck, "nobody") == 1.0


def test_record_world_rejects_unknown_duck_and_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["record-world", "living-room", "--seconds", "0.2",
                                      "--out", str(tmp_path), "--brain", "d9=wander"])
    with pytest.raises(SystemExit, match="d9"):
        record_world.main()
    monkeypatch.setattr(sys, "argv", ["record-world", "living-room", "--seconds", "0.2",
                                      "--out", str(tmp_path), "--camera", "follow:d9"])
    with pytest.raises(SystemExit, match="follow:d9"):
        record_world.main()
