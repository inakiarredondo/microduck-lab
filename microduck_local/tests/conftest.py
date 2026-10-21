"""Fixtures every test in this directory gets.

Deliberately almost empty: anything autouse here is paid for by all ~1400
tests, so only things that are WRONG to leave undone belong in it.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


def _clear_moss_scene_cache() -> None:
    """Drop `robots/moss_env._RECURRING_SCENES`, if that module is loaded.

    Asked of `sys.modules` rather than imported, so a session that never
    touches MOSS never pays for mujoco's import just to clear an empty dict.
    """
    mod = sys.modules.get("microduck_local.robots.moss_env")
    if mod is not None:
        mod.clear_scene_cache()


@pytest.fixture(autouse=True)
def moss_scene_cache():
    """Compiled MOSS scenes must not cross a test boundary.

    The cache keys on the inputs that vary at RUNTIME, not on every `moss`
    constant a scene reads — `_scene_inputs` says why that is the right
    trade and what it leaves open. What it leaves open is exactly a test:
    monkeypatch `moss.FINGER_FRICTION` or `moss.BIN_FLOOR_PRIORITY`, take
    the handover path, and `_bind_model` hands back the model compiled
    BEFORE the patch — so the test measures the unpatched physics and passes
    green. Clearing on both sides means neither the test that patches nor
    the one after it can be served a stale model.
    """
    _clear_moss_scene_cache()
    yield
    _clear_moss_scene_cache()


def _is_trainer_launch(cmd: object) -> bool:
    """Is this argv the LAB launching a trainer? `-m microduck_local.train*`.

    Anchored on the `-m` and the module name together, deliberately: the
    restart-guard tests plant sleepers whose FILE is named
    `microduck_local.train_behavior.py` precisely so they match the guard's
    `pgrep`, and a looser substring test would block those plants and make
    that suite untestable. A module launch is what the lab does and what a
    test must never do for real.
    """
    argv = cmd if isinstance(cmd, (list, tuple)) else [cmd]
    argv = [str(a) for a in argv]
    return any(a == "-m" and argv[i + 1].startswith("microduck_local.train")
               for i, a in enumerate(argv[:-1]))


def _called_by_the_lab() -> bool:
    """Did this `Popen` come from `viz_server`, or from a test itself?

    `viz_server.TrainingJob._launch` calls `subprocess.Popen` directly (two
    sites), so the frame that called the patched function is the lab's own.
    Walks a few frames rather than taking exactly one, so a helper added
    between them later does not silently disarm this.
    """
    frame = sys._getframe(2)             # 0 = here, 1 = the guard, 2 = caller
    for _ in range(4):
        if frame is None:
            return False
        if frame.f_globals.get("__name__") == "microduck_local.viz_server":
            return True
        frame = frame.f_back
    return False


@pytest.fixture(autouse=True)
def no_real_trainer(monkeypatch):
    """The LAB must never launch a real trainer during a test.

    Not "no test may launch a trainer" — that was this fixture's first shape
    and it broke 24 of them. `tests/test_train_resume.py` and
    `tests/test_train_brain_hparams.py` run small real trainings on purpose
    (`--envs 2 --steps 2000`); that IS the test. So the thing to forbid is
    narrower: `viz_server` spawning one behind a `/teach` request, which is a
    test that forgot `fake_popen`.

    The caller is read off the stack rather than guessed from the argv,
    because the argv is identical either way.

    MEASURED 2026-10-03: `test_lab_robots.test_a_duck_trick_is_never_matched
    _for_a_g1_trainee` POSTed `/teach` without `fake_popen`, matched a G1
    task, and the lab launched `-m microduck_local.train --robot g1 --task
    stand --envs 32 --steps 700000` for real — 33 processes, ~1.5 GB, 266 s,
    writing `model.zip` and `live.onnx` into the REAL `runs/` because
    `monkeypatch.setattr(V, "RUNS_DIR", ...)` cannot reach a child that
    computes its own path. It left five `teach-g1_stand-*` directories in one
    day, and the test passed in 1.66 s throughout, which is why it survived.
    What found it was `restart.sh`'s trainer guard refusing to restart the lab
    at random moments — a cost paid three layers away from the cause.

    Cheap enough for the rule in this module's docstring: a `sys.modules`
    lookup for a test that never touched the lab, and one `setattr` for one
    that did. `fake_popen` is requested explicitly, so pytest builds it AFTER
    this autouse one and its patch wins — tests that opt in are untouched.
    """
    # Armed UNCONDITIONALLY. The first shape armed only when `viz_server` was
    # already in `sys.modules`, which made it fire or not depending on TEST
    # ORDER — the resume tests passed alone and in every targeted run, and
    # failed only in the full suite after some earlier test imported the lab.
    # An order-dependent guard is worse than none: it reports a problem that
    # moves when you look at it.
    real = subprocess.Popen

    def guard(cmd, *a, **kw):
        if _is_trainer_launch(cmd) and _called_by_the_lab():
            raise AssertionError(
                "the LAB launched a REAL trainer for this test: "
                f"{' '.join(str(x) for x in cmd)[:120]}...\n"
                "Request the `fake_popen` fixture (tests/test_lab.py) — it "
                "records the launch and keeps runs/ and lab-state.json in "
                "tmp_path. A real one writes into the repo's runs/ and trips "
                "the restart guard for everybody.")
        return real(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", guard)
    yield
