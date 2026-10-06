"""The restart guard must refuse for WORK and proceed for TESTS.

`.claude/skills/restart-servers/restart.sh` refuses to bounce the lab while a
trainer is running, because a restart kills the lab's children with it. It used
to refuse for ANY process matching `microduck_local.train_behavior` /
`microduck_local.train ` — and this repo's own tests spawn real (short) training
runs (test_net_arch, test_select_run, test_distill, test_train_resume), so a
`pytest tests/` in another window tripped it at random moments. On 2026-10-03
that cost a confused restart loop, and the documented escape hatch
`MICRODUCK_RESTART_FORCE=1` would have killed the TEST SUITE rather than a teach
job.

The guard now partitions the trainers it finds: lab-owned (refuse, named),
positively a test (proceed), unknown (refuse). These tests drive
`restart.sh --check`, which runs ONLY the guard and exits 0 for "would proceed"
/ 1 for "would refuse", against real planted process trees.

Two things this file is built around, both of which cost time:

* **A plant that starts nothing passes everything.** The first attempt started
  no process whose command line matched the guard's pattern, and every
  assertion was vacuously green. Every plant here therefore asserts its process
  is visible to the guard's own `pgrep -fl` before a test may use it.
* **A test may not touch a process it did not create.** Every planted pid is
  killed by pid at teardown, and only after `ps` confirms the command line is
  still the one this file wrote. Foreign trainers are left alone, and tolerated
  when the guard can place them — see `_require_no_foreign_work`. The live lab
  on :8788 is never contacted: the guard's lab reading is pointed at a `file://`
  JSON fixture through `MICRODUCK_LAB_STATUS_URL`, so the lab-owned case is
  exercised without starting real training.

Shown to fail before the fix (11 of 14), and shown to have teeth after it — see
the "planted regressions" note at the bottom of this file.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2]
          / ".claude" / "skills" / "restart-servers" / "restart.sh")

#: The sleeper's FILE NAME is what makes its command line match the guard's
#: pattern — `python3 .../microduck_local.train_behavior.py`. Naming the module
#: is the whole trick: nothing has to import, let alone train.
TRAINER_NAME = "microduck_local.train_behavior.py"
TRAINER_RE = r"microduck_local\.train_behavio[r]|microduck_local\.trai[n] "
#: Long enough for one test (each finishes in a few seconds), short enough that
#: a hard-killed suite cannot leave a decoy behind for long. While an
#: "unidentifiable" plant is up, a `restart.sh` in another window WILL refuse —
#: correctly, since that is what the plant is pretending to be — so the window
#: is kept small rather than comfortable.
SLEEP_S = 45

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists() or not shutil.which("pgrep") or not shutil.which("bash"),
    reason="needs restart.sh, pgrep and bash")


# --------------------------------------------------------------- the plant

def _pgrep_trainers() -> str:
    return subprocess.run(["pgrep", "-fl", TRAINER_RE],
                          capture_output=True, text=True).stdout


class Planted:
    """Processes this file started, and nothing else."""

    def __init__(self) -> None:
        self.owned: dict[int, str] = {}      # pid -> a path its argv must hold

    def add(self, pid: int, fingerprint: Path) -> int:
        self.owned[pid] = str(fingerprint)
        return pid

    def visible(self, pid: int) -> bool:
        """Does the guard's own `pgrep -fl` see this pid? Asked before every
        assertion, because a plant that silently started nothing reads exactly
        like a guard that correctly ignored it."""
        return any(line.split(" ", 1)[0] == str(pid)
                   for line in _pgrep_trainers().splitlines())

    def wait_visible(self, pid: int) -> int:
        for _ in range(200):
            if self.visible(pid):
                return pid
            time.sleep(0.05)
        raise AssertionError(
            f"planted pid {pid} never appeared in the guard's own pgrep — the "
            "plant started nothing that matches, so every assertion would be "
            f"vacuous. table:\n{_pgrep_trainers()}")

    def kill_all(self) -> None:
        for pid, fingerprint in self.owned.items():
            cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                                 capture_output=True, text=True).stdout
            if fingerprint not in cmd:
                continue         # already gone, or the pid was recycled: not ours
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        for pid in self.owned:
            for _ in range(200):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)


@pytest.fixture
def planted():
    p = Planted()
    yield p
    p.kill_all()


def _sleeper(where: Path) -> Path:
    """A script named after the trainer module that only sleeps."""
    where.mkdir(parents=True, exist_ok=True)
    f = where / TRAINER_NAME
    f.write_text(f"import time\ntime.sleep({SLEEP_S})\n")
    return f


def _ppid(pid: int) -> str:
    return subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)],
                          capture_output=True, text=True).stdout.strip()


def _start_child(planted: Planted, where: Path, env: dict) -> int:
    """A trainer whose parent IS this pytest process — how the real ones run."""
    f = _sleeper(where)
    p = subprocess.Popen([sys.executable, str(f)], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return planted.wait_visible(planted.add(p.pid, f))


def _detach(cmd: list[str], env: dict) -> int:
    """Start `cmd` in its own session and return its pid after the launcher has
    exited, so the kernel has reparented it to init."""
    launcher = textwrap.dedent(f"""
        import subprocess
        p = subprocess.Popen({cmd!r}, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        print(p.pid)
    """)
    out = subprocess.run([sys.executable, "-c", launcher], env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return int(out.stdout.strip())


def _start_orphan(planted: Planted, where: Path, env: dict) -> int:
    """A trainer reparented to init: NO pytest anywhere in its parent chain.

    This is the case ancestry alone cannot see, and it is not hypothetical — a
    vec-env worker outlives its trainer by design here (viz_server's
    TrainingJob._snapshot_workers says why).
    """
    f = _sleeper(where)
    pid = planted.add(_detach([sys.executable, str(f)], env), f)
    planted.wait_visible(pid)
    assert _ppid(pid) in ("1", "0"), f"pid {pid} has parent {_ppid(pid)}, not init"
    return pid


def _start_under_fake_pytest(planted: Planted, where: Path, env: dict) -> int:
    """A trainer whose only pytest is its PARENT's command line.

    The wrapper is a shell script named `pytest_runner.sh` — `*pytest*` on a
    parent's command line is all the ancestry reading looks for — and the
    wrapper itself is detached, so the real pytest running this test is NOT in
    the chain. Without that the chain would contain pytest either way and the
    test would not isolate the reading it is for. It must NOT `exec`: an exec'd
    wrapper is replaced by python and the pytest name disappears.
    """
    f = _sleeper(where)
    runner = where / "pytest_runner.sh"
    runner.write_text(f'"{sys.executable}" "{f}"\nwait\n')
    wrapper = planted.add(_detach(["bash", str(runner)], env), runner)
    trainer = None
    for _ in range(200):
        kids = subprocess.run(["pgrep", "-P", str(wrapper)],
                              capture_output=True, text=True).stdout.split()
        if kids:
            trainer = int(kids[0])
            break
        time.sleep(0.05)
    assert trainer is not None, f"wrapper {wrapper} never spawned the trainer"
    planted.add(trainer, f)
    planted.wait_visible(trainer)
    assert _ppid(trainer) == str(wrapper)
    assert _ppid(wrapper) in ("1", "0"), "the wrapper must be the only pytest"
    return trainer


# ------------------------------------------------------------ environments

def _clean_env() -> dict:
    """This process's environment with every tell of pytest removed.

    `PYTEST_CURRENT_TEST` is in `os.environ` while a test runs, so a child
    spawned with `dict(os.environ)` inherits it — which is exactly how the real
    test trainers are identified, and exactly what must not leak into the
    "unknown trainer" plant.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
    for k in ("MICRODUCK_RUNS_DIR", "MICRODUCK_RESTART_FORCE",
              "MICRODUCK_LAB_STATUS_URL"):
        env.pop(k, None)
    return env


def _pytest_env() -> dict:
    """What a real test trainer is launched with: `dict(os.environ, ...)`."""
    env = dict(os.environ)
    env.pop("MICRODUCK_RESTART_FORCE", None)
    env.pop("MICRODUCK_LAB_STATUS_URL", None)
    assert "PYTEST_CURRENT_TEST" in env, "pytest no longer marks its own env"
    return env


# --------------------------------------------------------- driving --check

IDLE = {"running": False, "status": "idle", "job": None, "jobs": [],
        "envBudget": {"used": 0, "total": 84}}


def _owning(pids: list[int], running: bool = True) -> dict:
    job = {"behavior": "squat", "title": "Squat", "runName": "teach-squat",
           "status": "training" if running else "done",
           "steps": 120_000, "total": 2_000_000, "pids": pids}
    return {"running": running, "status": job["status"], "job": job,
            "jobs": [job], "envBudget": {"used": 8, "total": 84}}


def _check(tmp_path: Path, status: dict) -> subprocess.CompletedProcess:
    """Run ONLY the guard, with the lab's answer coming from a file."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    f = tmp_path / "teach-status.json"
    f.write_text(json.dumps(status))
    env = _clean_env()
    env["MICRODUCK_LAB_STATUS_URL"] = f.as_uri()
    return subprocess.run(["bash", str(SCRIPT), "--check"], env=env,
                          capture_output=True, text=True, timeout=180)


def _require_no_foreign_work(tmp_path: Path) -> None:
    """Skip if anything is running that this file did not plant AND the guard
    cannot place.

    The precondition is NOT an empty process table. On this machine the table is
    rarely empty: the repo's own suite spawns real G1 trainers that live ~4.5
    minutes each (test_lab_robots.py's `/teach` cases run without `fake_popen`),
    and an earlier version of this helper skipped eleven of twelve tests for
    that reason. A foreign trainer the guard sorts into the TEST bucket changes
    no verdict here — it only adds lines — so it is tolerated. A foreign trainer
    it cannot place would decide every verdict, and if it is somebody's real job
    then refusing is correct and the plant is untestable; skip.

    Asked of the guard itself rather than re-implemented, so the two cannot
    drift apart.
    """
    r = _check(tmp_path / "precondition", IDLE)
    if r.returncode != 0:
        pytest.skip("work the guard will not restart past is already running:\n"
                    + r.stdout)


# ------------------------------------------------------------------ cases

def test_nothing_to_protect_proceeds(tmp_path):
    """(a) An idle lab and nothing the guard will not restart past: it must not
    be in the way. (Foreign TEST trainers may be in the table — see
    `_require_no_foreign_work` — and must not change this.)"""
    _require_no_foreign_work(tmp_path)
    r = _check(tmp_path, IDLE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "would PROCEED" in r.stdout
    assert "REFUSING" not in r.stdout


def test_a_pytest_env_trainer_lets_the_restart_through(tmp_path, planted):
    """(b) The case that cost the restart loop: a trainer spawned from inside a
    pytest test, exactly as tests/test_net_arch.py spawns one.

    Identified by `PYTEST_CURRENT_TEST` in its own environment — the signal that
    survives its parent dying. The evidence string is asserted, not just the
    exit code, because the ancestry reading would also fire here and the point
    is which one did.
    """
    _require_no_foreign_work(tmp_path)
    pid = _start_child(planted, tmp_path / "plant", _pytest_env())
    r = _check(tmp_path, IDLE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "POSITIVELY a" in r.stdout
    assert "PYTEST_CURRENT_TEST" in r.stdout
    assert str(pid) in r.stdout, "the skipped trainer must be named"
    assert "REFUSING" not in r.stdout


def test_an_orphaned_test_trainer_is_still_identified(tmp_path, planted):
    """(b) The planted regression for the ancestry-only version: a test trainer
    whose pytest parent is GONE.

    No `PYTEST_CURRENT_TEST` in its env and nothing above it is pytest — the
    only evidence left is `MICRODUCK_RUNS_DIR` inside pytest's tmp root. A guard
    that only walked the parent chain refuses here.
    """
    _require_no_foreign_work(tmp_path)
    assert "/pytest-of-" in str(tmp_path), "pytest's tmp root was renamed"
    env = _clean_env()
    env["MICRODUCK_RUNS_DIR"] = str(tmp_path)
    pid = _start_orphan(planted, tmp_path / "plant", env)
    r = _check(tmp_path, IDLE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "pytest-of-" in r.stdout
    assert str(pid) in r.stdout


def test_a_pytest_parent_alone_is_enough(tmp_path, planted):
    """(b) The third reading on its own: nothing in the environment, but the
    parent's command line says pytest."""
    _require_no_foreign_work(tmp_path)
    pid = _start_under_fake_pytest(planted, tmp_path / "plant", _clean_env())
    r = _check(tmp_path, IDLE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "pytest parent" in r.stdout
    assert str(pid) in r.stdout


def test_an_unidentifiable_trainer_still_refuses(tmp_path, planted):
    """(c) THE safety property. A trainer with no pytest in its environment and
    no pytest above it is somebody's work, and the guard may not proceed.

    The sleeper itself lives under a `pytest-of-` path (it is written into
    `tmp_path`), so this also pins that the runs-dir reading is anchored on
    `MICRODUCK_RUNS_DIR=` and does not fire on any mention of pytest's tmp root
    elsewhere in the command line.
    """
    _require_no_foreign_work(tmp_path)
    pid = _start_orphan(planted, tmp_path / "plant", _clean_env())
    r = _check(tmp_path, IDLE)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stdout
    assert "could not be identified" in r.stdout
    assert str(pid) in r.stdout


def test_a_lab_owned_job_refuses_even_when_it_looks_like_a_test(tmp_path, planted):
    """(d) Ownership beats evidence of testhood, with `running` out of the way.

    The lab launches a trainer with `env={**os.environ, ...}`, so a lab started
    from a pytest-tainted shell would hand its teach job the very markers that
    identify a test. The pids from `/teach/status` are read FIRST for that
    reason, and this case is the only thing that says so: the planted trainer
    carries `PYTEST_CURRENT_TEST` (it would proceed on its own — the test above
    proves that) and the lab's answer claims it while reporting
    `running: false`.
    """
    _require_no_foreign_work(tmp_path)
    pid = _start_child(planted, tmp_path / "plant", _pytest_env())
    r = _check(tmp_path, _owning([pid], running=False))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "owns these trainer processes" in r.stdout
    assert str(pid) in r.stdout
    assert "POSITIVELY a" not in r.stdout


def test_a_live_teach_job_refuses(tmp_path, planted):
    """(d) The ordinary teach job: owned AND `running: true`."""
    _require_no_foreign_work(tmp_path)
    pid = _start_child(planted, tmp_path / "plant", _pytest_env())
    r = _check(tmp_path, _owning([pid]))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "owns these trainer processes" in r.stdout
    assert '"running":true' in r.stdout


def test_the_labs_running_flag_still_refuses_on_its_own(tmp_path):
    """The second, INDEPENDENT reading. A job whose trainer the pattern never
    matches (a `train_brain`, say) shows up only here, so `running: true` has to
    refuse with no process evidence at all."""
    _require_no_foreign_work(tmp_path)
    r = _check(tmp_path, _owning([], running=True))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "a teach job is training" in r.stdout


def test_a_lab_that_predates_the_pids_field_still_refuses(tmp_path, planted):
    """Graceful degradation. An older lab answers `/teach/status` without
    `pids`; its live trainer then has no claim on it and must fall through to
    "unknown", which refuses. The failure mode to avoid is a missing field
    reading as "the lab owns nothing, so nobody does"."""
    _require_no_foreign_work(tmp_path)
    pid = _start_orphan(planted, tmp_path / "plant", _clean_env())
    old = {"running": False, "status": "idle", "job": None,
           "jobs": [{"runName": "teach-old", "status": "done"}]}
    r = _check(tmp_path, old)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "could not be identified" in r.stdout
    assert str(pid) in r.stdout


def test_one_unknown_trainer_outvotes_a_crowd_of_tests(tmp_path, planted):
    """Mixed table: proceeding needs EVERY trainer identified, not most.

    The refusal must name the unknown one and say the tests were not the reason
    — the old message said "a trainer process is running" and left the reader to
    guess which of them it meant.
    """
    _require_no_foreign_work(tmp_path)
    test_pid = _start_child(planted, tmp_path / "t", _pytest_env())
    unknown_pid = _start_orphan(planted, tmp_path / "u", _clean_env())
    r = _check(tmp_path, IDLE)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "could not be identified" in r.stdout
    assert str(unknown_pid) in r.stdout
    assert "are this repo's TESTS and are not the reason" in r.stdout
    assert str(test_pid) in r.stdout


def test_check_is_a_read_and_never_restarts_anything(tmp_path):
    """`--check` must not touch the stack. It is the "check first, as a separate
    command" the skill doc asks for, so it may not print a word of the restart
    it is deciding about."""
    _require_no_foreign_work(tmp_path)
    r = _check(tmp_path, IDLE)
    for noise in ("stopping the lab", "starting backend", "nothing listening",
                  "backend UP"):
        assert noise not in r.stdout, f"--check reached the restart: {r.stdout}"


# ----------------------------------------------------- the FORCE preamble

#: Where the FORCE preamble ends. The test splices an `exit 0` after it so the
#: preamble can be driven without the restart it precedes.
FORCE_PREAMBLE_END = ('    echo "  Continuing in 5s (Ctrl-C to abort)..."\n'
                      "    sleep 5\n  fi\nfi\n")


def test_a_force_lists_the_jobs_it_is_about_to_kill(tmp_path):
    """A hook that had never fired. `MICRODUCK_RESTART_FORCE=1` is supposed to
    print the live jobs and their spent steps before it destroys them — "the
    cost is visible at the moment of the decision rather than in an apology
    afterwards", says the comment. It never printed anything: the step line was
    an f-string containing `{j[\\"runName\\"]}`, a backslash inside an f-string
    expression is a SyntaxError on every Python this repo runs, and `2>/dev/null`
    swallowed it. An empty list is indistinguishable from "nothing is training",
    so a FORCE that went through looked correct every time.

    Driven on a spliced COPY: the preamble is followed by the restart itself,
    and no test may run that.
    """
    _require_no_foreign_work(tmp_path)
    src = SCRIPT.read_text()
    assert FORCE_PREAMBLE_END in src, (
        "the FORCE preamble's tail moved — update FORCE_PREAMBLE_END; this test "
        "must stop before the restart, never splice blindly")
    copy = tmp_path / "force-probe.sh"
    copy.write_text(src.replace(FORCE_PREAMBLE_END,
                                FORCE_PREAMBLE_END + "exit 0\n", 1))

    status = tmp_path / "doomed.json"
    status.write_text(json.dumps({
        "running": True, "status": "training",
        "jobs": [
            {"runName": "teach-dribble-s3", "status": "training",
             "steps": 2_930_000, "total": 6_000_000, "pids": [999999]},
            {"runName": "teach-squat-s1", "status": "done",
             "steps": 500_000, "total": 500_000, "pids": []},
        ]}))
    env = _clean_env()
    env["MICRODUCK_RESTART_FORCE"] = "1"
    env["MICRODUCK_LAB_STATUS_URL"] = status.as_uri()
    r = subprocess.run(["bash", str(copy)], env=env, capture_output=True,
                       text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "will KILL these live training jobs" in r.stdout
    assert "teach-dribble-s3  2,930,000/6,000,000 steps" in r.stdout, (
        f"the doomed list is empty again:\n{r.stdout}\n{r.stderr}")
    assert "teach-squat-s1" not in r.stdout, "a finished job is not doomed"


def test_check_does_not_trip_the_force_preamble(tmp_path):
    """`--check` is a question, not a decision: it must not announce a kill, and
    it must not spend the preamble's 5-second grace period."""
    _require_no_foreign_work(tmp_path)
    status = tmp_path / "doomed.json"
    status.write_text(json.dumps(_owning([999999], running=True)))
    env = _clean_env()
    env["MICRODUCK_RESTART_FORCE"] = "1"
    env["MICRODUCK_LAB_STATUS_URL"] = status.as_uri()
    t0 = time.monotonic()
    r = subprocess.run(["bash", str(SCRIPT), "--check"], env=env,
                       capture_output=True, text=True, timeout=180)
    assert "will KILL" not in r.stdout
    assert time.monotonic() - t0 < 5.0, "it slept through the FORCE grace period"
    # and it reports the GUARD's verdict, saying the FORCE would override it
    assert r.returncode == 1
    assert "MICRODUCK_RESTART_FORCE=1 is set" in r.stdout


# ------------------------------------------- the field the guard reads

def test_teach_status_publishes_the_pids_a_job_owns(monkeypatch, tmp_path):
    """The guard's strongest reading is only as good as this field.

    `/teach/status` grew `pids` per job so something outside the lab can tell a
    teach job apart from any other trainer on the machine. Additive: every key a
    waiter already reads is untouched.
    """
    from microduck_local import viz_server as V

    class FakeProc:
        def __init__(self, cmd, **kw):
            self.pid = 424242
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(V.subprocess, "Popen", lambda cmd, **kw: FakeProc(cmd))
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))

    app = V.make_app([])

    def endpoint(path, method):
        for r in app.routes:
            if getattr(r, "path", None) == path and method in (
                    getattr(r, "methods", None) or ()):
                return r.endpoint
        raise AssertionError(f"no {method} {path}")

    idle = endpoint("/teach/status", "GET")()
    assert idle["jobs"] == [] and idle["job"] is None

    asyncio.run(endpoint("/teach", "POST")(V.TeachReq(text="stand still")))
    status = endpoint("/teach/status", "GET")
    live = status()
    assert live["job"]["pids"] == [424242]
    assert live["jobs"][0]["pids"] == [424242]
    # the keys a waiter already reads are all still there
    for k in ("running", "status", "job", "jobs", "envBudget"):
        assert k in live

    # A STOPPED job owns nothing. The guard must not refuse for a pid the lab is
    # merely remembering — that number belongs to whatever process reuses it.
    asyncio.run(endpoint("/teach/stop", "POST")())
    assert status()["job"]["pids"] == []


# ======================================================================
# PLANTED REGRESSIONS — what was done to show these tests can fail.
#
# A check that has never been shown to fire proves nothing, so every claim here
# was broken on purpose and the file re-run. Each arm is this same file with
# SCRIPT pointed at a copy in a scratch dir; 14/14 pass against the real one.
#
# * BEFORE the fix — HEAD's restart.sh (refuses for ANY trainer), with one
#   `exit 0` spliced after its guard's closing `fi` so it can be driven at all:
#   11 failed, 3 passed. Among them the three cases that are the user-visible
#   bug (test_a_pytest_env_trainer_lets_the_restart_through,
#   test_an_orphaned_test_trainer_is_still_identified,
#   test_a_pytest_parent_alone_is_enough) and
#   test_a_force_lists_the_jobs_it_is_about_to_kill. The refusal cases also
#   fail, but only on the message: the old guard refuses too, it just cannot
#   say which trainer it means.
#       git show HEAD:.claude/skills/restart-servers/restart.sh > /tmp/old.sh
#
# * `_test_evidence` patched to always print "test", i.e. everything reads as a
#   test — 5 failed: test_an_unidentifiable_trainer_still_refuses (the safety
#   property), test_a_lab_that_predates_the_pids_field_still_refuses,
#   test_one_unknown_trainer_outvotes_a_crowd_of_tests, and the two (b) cases
#   that assert WHICH signal fired rather than just the exit code.
#   test_a_lab_owned_job_refuses_even_when_it_looks_like_a_test survives it,
#   because ownership is checked before any evidence — which is the point.
#
# * `owned=" "` in _classify_trainers, i.e. the lab's pids are never consulted
#   — 2 failed, exactly the two lab-owned cases and nothing else. That is the
#   whole justification for the `pids` field.
# ======================================================================
