"""What a run SAYS it is: run_record.py, describe-run, and the palette.

These exist because of a measured failure, and the tests are written against
it: five G1 imitation runs trained through the lab showed up in the policy
palette as five rows of `teach-g1_imitate-g1_front_kick-<hash>-sN`, and the
chain's own button pointed at the FINAL stage — which was the worst one
(3/8 seeds against 8/8 two stages earlier). So the load-bearing cases here
are "a record reaches the palette", "the pick is unique within a chain", and
"a trainer never writes a judgement about its own run".
"""

from __future__ import annotations

import json

import pytest

from microduck_local import run_record as R
from microduck_local import viz_server as V
from microduck_local.describe_run import main as describe_main


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """An isolated runs/ that describe-run and the palette both look at."""
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setattr(V, "RUNS_DIR", d)
    import microduck_local.describe_run as DR
    monkeypatch.setattr(DR, "RUNS_DIR", d)
    return d


@pytest.fixture
def fake_popen(monkeypatch, tmp_path):
    """Record every trainer launch instead of spawning one (the same trick as
    tests/test_lab.py's fixture of this name, kept local so this file does not
    import another test module)."""
    launches: list[dict] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            self.cmd, self.kwargs, self.returncode = cmd, kwargs, None

        def poll(self):
            return self.returncode

    def popen(cmd, **kwargs):
        proc = FakeProc(cmd, **kwargs)
        launches.append(proc)
        return proc

    monkeypatch.setattr(V.subprocess, "Popen", popen)
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    return launches


def _run(runs, name, **run_json):
    d = runs / name
    d.mkdir()
    (d / "policy.onnx").write_bytes(b"not a real onnx, but the palette lists it")
    meta = {"run_name": name, "robot": "g1", "task": "imitate", "steps": 1_000_000,
            "env_kwargs": {"clip_name": "g1-front-kick"}, "init_from": None}
    meta.update(run_json)
    (d / "run.json").write_text(json.dumps(meta))
    return d


# ------------------------------------------------------------ the record

def test_a_run_with_no_record_reads_as_empty_not_an_error(runs):
    d = _run(runs, "plain")
    assert R.read_record(d) == {}


def test_an_unreadable_record_never_takes_the_run_down_with_it(runs):
    """A label must not be able to hide the policy it labels — the same
    reasoning as load_clips skipping an unparseable clip."""
    d = _run(runs, "broken")
    R.record_path(d).write_text("{ this is not json")
    assert R.read_record(d) == {}
    entry = _entry(V.discover_policies(), "broken")
    assert entry is not None and "title" not in entry


def test_write_record_merges_and_none_leaves_a_field_alone(runs):
    d = _run(runs, "merge")
    R.write_record(d, title="first", description="why")
    R.write_record(d, title=None, note="measured")
    got = R.read_record(d)
    assert got == {"title": "first", "description": "why", "note": "measured"}
    # An empty string is a deliberate clear, not "leave alone".
    R.write_record(d, note="")
    assert R.read_record(d)["note"] == ""


def test_write_record_leaves_no_temp_file_behind(runs):
    d = _run(runs, "atomic")
    R.write_record(d, title="t")
    assert [p.name for p in d.iterdir() if p.name.startswith("record.json.")] == []
    assert json.loads(R.record_path(d).read_text())["title"] == "t"


def test_a_second_measurement_by_the_same_tool_replaces_the_first(runs):
    """The honest current reading is the last one taken: a growing list is
    how a run ends up quoted at its best ever rather than its present."""
    d = _run(runs, "measured")
    R.write_measurement(d, "eval_imitate.py --seeds 8", {"apex_m": 0.10})
    R.write_measurement(d, "eval_imitate.py --seeds 8", {"apex_m": 0.62})
    R.write_measurement(d, "eval_imitate.py --seeds 8 --noise", {"apex_m": 0.65})
    measured = R.read_record(d)["measured"]
    assert measured["eval_imitate.py --seeds 8"] == {"apex_m": 0.62}
    assert len(measured) == 2


# ------------------------------------------------- what a trainer writes

def test_the_default_record_states_facts_and_never_a_verdict(runs):
    rec = R.default_record("teach-g1_imitate-x-s2", json.loads(
        (_run(runs, "facts", init_from="/abs/path/teach-g1_imitate-x-s1") / "run.json").read_text()),
        behavior_title="Perform “g1-front-kick”",
        stage_env={"MICRODUCK_G1_LIFT_MIN": "0.15", "MICRODUCK_CLIP": "g1-front-kick"})
    assert rec["title"] == "Perform “g1-front-kick”"     # not doubled
    assert rec["clip"] == "g1-front-kick"
    assert rec["init_from"] == "teach-g1_imitate-x-s1"   # basename, not an abs path
    # The clip is named on its own; it is not a curriculum rung.
    assert rec["stage_env"] == {"MICRODUCK_G1_LIFT_MIN": "0.15"}
    assert "MICRODUCK_G1_LIFT_MIN=0.15" in rec["description"]
    assert "warm-started from teach-g1_imitate-x-s1" in rec["description"]
    # A trainer does not get to say its own run was good.
    for judgement in ("note", "pick", "measured"):
        assert judgement not in rec


def test_a_title_that_does_not_name_the_clip_gains_it(runs):
    rec = R.default_record("r", json.loads((_run(runs, "clipname") / "run.json").read_text()))
    assert rec["title"] == "imitate (g1) · “g1-front-kick”"


def test_the_lab_hands_the_trainer_the_title_the_watcher_saw(fake_popen):
    """MICRODUCK_RUN_TITLE: the trainer writes the record and cannot know what
    the lab called the job, so the job passes its own display title down."""
    job = V.TrainingJob("g1_stand", steps=1000)
    env = fake_popen[0].kwargs["env"]
    assert env["MICRODUCK_RUN_TITLE"] == job.display_title()


# ------------------------------------------------------------ describe-run

def test_describe_run_writes_the_fields_and_refuses_an_empty_call(runs, capsys):
    _run(runs, "solo")
    assert describe_main(["solo", "--title", "T", "--note", "8/8 seeds"]) == 0
    got = R.read_record(runs / "solo")
    assert got["title"] == "T" and got["note"] == "8/8 seeds"
    capsys.readouterr()
    assert describe_main(["solo"]) == 2            # nothing to write
    assert describe_main(["nosuchrun", "--title", "T"]) == 2


def test_only_one_stage_of_a_chain_can_be_the_pick(runs, capsys):
    """Two picks is the same 'which one do I use?' question the record
    exists to answer — setting one clears the others."""
    for n in ("teach-k-abc-s1", "teach-k-abc-s2", "teach-k-abc-s3"):
        _run(runs, n)
    _run(runs, "teach-k-OTHER-s2")                 # a different chain
    describe_main(["teach-k-abc-s1", "--pick"])
    describe_main(["teach-k-OTHER-s2", "--pick"])
    describe_main(["teach-k-abc-s3", "--pick"])
    capsys.readouterr()
    picks = {n: bool(R.read_record(runs / n).get("pick"))
             for n in ("teach-k-abc-s1", "teach-k-abc-s2", "teach-k-abc-s3")}
    assert picks == {"teach-k-abc-s1": False, "teach-k-abc-s2": False,
                     "teach-k-abc-s3": True}
    # ...and a sibling chain is untouched.
    assert R.read_record(runs / "teach-k-OTHER-s2")["pick"] is True
    describe_main(["teach-k-abc-s3", "--no-pick"])
    assert R.read_record(runs / "teach-k-abc-s3")["pick"] is False


def test_backfill_fills_only_what_is_missing(runs, capsys):
    """Run over hundreds of old directories at once: a hand-written title is
    worth more than a derived one and must survive."""
    _run(runs, "old")
    describe_main(["old", "--title", "Mine", "--note", "measured"])
    capsys.readouterr()
    assert describe_main(["old", "--backfill"]) == 0
    got = R.read_record(runs / "old")
    assert got["title"] == "Mine"                  # not overwritten
    assert got["note"] == "measured"
    assert "imitate on the g1" in got["description"]   # filled, was missing


def test_backfill_reports_a_run_it_cannot_name(runs, capsys):
    d = runs / "norunjson"
    d.mkdir()
    assert describe_main(["norunjson", "--backfill"]) == 2
    assert "nothing to derive" in capsys.readouterr().err


# ------------------------------- the OTHER trainer's file (behavior.json)

def _trick(runs, name, **behavior_json):
    """A duck trick run: train_behavior writes behavior.json, not run.json."""
    d = runs / name
    d.mkdir()
    (d / "policy.onnx").write_bytes(b"onnx")
    meta = {"behavior": "imitate", "steps": 3_000_000, "weights": {}}
    meta.update(behavior_json)
    (d / "behavior.json").write_text(json.dumps(meta))
    return d


def test_a_trick_runs_own_title_reaches_the_palette(runs):
    """behavior.json has carried title/description since the teach panel
    existed, but only the /train page read it — 647 trick runs sat in the
    palette as bare directory names with their titles already on disk."""
    _trick(runs, "teach-backflip-abc", title="Backflip — the pick",
           description="the one that lands")
    assert R.read_label(runs / "teach-backflip-abc")["title"] == "Backflip — the pick"
    assert _entry(V.discover_policies(), "teach-backflip-abc")["title"] == "Backflip — the pick"


def test_an_untitled_trick_run_is_named_by_its_recipe_and_clip(runs):
    _trick(runs, "teach-imitate-xyz", clip="sprint-cycle")
    assert R.read_label(runs / "teach-imitate-xyz")["title"] == "imitate · “sprint-cycle”"
    _trick(runs, "teach-spin-xyz", behavior="spin")
    assert R.read_label(runs / "teach-spin-xyz")["title"] == "spin"


def test_the_record_wins_over_behavior_json(runs):
    """record.json is this module's file and the one describe-run writes, so
    a hand-written title must not be shadowed by the trainer's."""
    d = _trick(runs, "teach-both-abc", title="from behavior.json")
    R.write_record(d, title="from record.json", note="8/8", pick=True)
    label = R.read_label(d)
    assert label == {"title": "from record.json", "note": "8/8", "pick": True}


def test_a_run_with_neither_file_is_simply_unnamed(runs):
    d = runs / "bare"
    d.mkdir()
    (d / "policy.onnx").write_bytes(b"onnx")
    assert R.read_label(d) == {}
    assert "title" not in _entry(V.discover_policies(), "bare")


def test_backfill_names_a_trick_run_from_its_behavior_json(runs, capsys):
    _trick(runs, "teach-crouch-abc", behavior="crouch")
    assert describe_main(["teach-crouch-abc", "--backfill"]) == 0
    capsys.readouterr()
    assert R.read_record(runs / "teach-crouch-abc")["title"] == "crouch"


def test_backfill_still_keeps_the_pick_unique_on_a_trick_chain(runs, capsys):
    """--backfill takes one path through main() whatever file it read from:
    its first draft returned early for trick runs and skipped this."""
    _trick(runs, "teach-flip-abc-s1")
    _trick(runs, "teach-flip-abc-s2")
    describe_main(["teach-flip-abc-s1", "--pick"])
    describe_main(["teach-flip-abc-s2", "--backfill", "--pick"])
    capsys.readouterr()
    assert R.read_record(runs / "teach-flip-abc-s1")["pick"] is False
    assert R.read_record(runs / "teach-flip-abc-s2")["pick"] is True


def test_backfill_says_so_when_there_is_nothing_to_derive_from(runs, capsys):
    d = runs / "nothing"
    d.mkdir()
    assert describe_main(["nothing", "--backfill"]) == 2
    assert "nothing to derive" in capsys.readouterr().err


# ------------------------------------------------------- the roster row

def test_assigning_a_described_run_puts_its_title_on_the_roster(runs, monkeypatch):
    """The roster's policy column is where the user looked first and saw a
    list of hashes; a described run says its name there too, while the event
    line keeps the identifier so the log stays greppable."""
    import asyncio

    R.write_record(_run(runs, "teach-k-xyz-s2"), title="Front kick (G1)")
    monkeypatch.setenv("LAB_STATE_PATH", str(runs.parent / "lab-state.json"))
    monkeypatch.setattr(V, "load_policy_infer", lambda pid: V._zero_infer)
    monkeypatch.setattr(V, "policy_robot", lambda p: "microduck")
    monkeypatch.setattr(V, "env_kwargs_for_policy_path", lambda p: {})
    app = V.make_app([])
    duck = V.Duck("d0", "before", V._zero_infer, seed=1, env_kwargs={})
    st = app.state.lab
    st.ducks[:] = [duck]
    asyncio.run(app.state.do_assign("d0", "run:teach-k-xyz-s2"))
    assert duck.label == "Front kick (G1)"
    assert duck.policy_id == "run:teach-k-xyz-s2"
    assert any("teach-k-xyz-s2" in e for e in st.events), st.events


def test_an_undescribed_run_still_shows_its_name_on_the_roster(runs, monkeypatch):
    import asyncio

    _run(runs, "teach-k-plain-s1")
    monkeypatch.setenv("LAB_STATE_PATH", str(runs.parent / "lab-state.json"))
    monkeypatch.setattr(V, "load_policy_infer", lambda pid: V._zero_infer)
    monkeypatch.setattr(V, "policy_robot", lambda p: "microduck")
    monkeypatch.setattr(V, "env_kwargs_for_policy_path", lambda p: {})
    app = V.make_app([])
    duck = V.Duck("d0", "before", V._zero_infer, seed=1, env_kwargs={})
    app.state.lab.ducks[:] = [duck]
    asyncio.run(app.state.do_assign("d0", "run:teach-k-plain-s1"))
    assert duck.label == "teach-k-plain-s1"


# --------------------------------------------------------- the palette

def _entry(policies, run_name):
    return next((p for p in policies if p.get("id") == f"run:{run_name}"), None)


def test_the_palette_carries_the_title_note_and_pick(runs):
    d = _run(runs, "teach-k-xyz-s2")
    R.write_record(d, title="Front kick (G1)", note="8/8 seeds hold 20 s", pick=True)
    entry = _entry(V.discover_policies(), "teach-k-xyz-s2")
    assert entry["title"] == "Front kick (G1)"
    assert entry["note"] == "8/8 seeds hold 20 s"
    assert entry["pick"] is True
    # The LABEL stays the run name whatever the title says: it is the
    # identifier --init-from, the docs and the delete endpoint address.
    assert entry["label"] == "teach-k-xyz-s2"
    assert entry["chain"] == "teach-k-xyz" and entry["stage"] == 2


def test_a_described_run_without_a_pick_says_nothing_about_one(runs):
    d = _run(runs, "nopick")
    R.write_record(d, title="T")
    entry = _entry(V.discover_policies(), "nopick")
    assert entry["title"] == "T"
    assert "pick" not in entry and "note" not in entry


def test_a_cleared_pick_is_not_served_as_one(runs):
    d = _run(runs, "cleared")
    R.write_record(d, pick=True)
    R.write_record(d, pick=False)
    assert "pick" not in _entry(V.discover_policies(), "cleared")


def test_a_stopped_run_does_not_keep_its_half_written_snapshot(tmp_path):
    """`POST /teach/stop` SIGTERMs the trainer, and Python runs NO cleanup on
    SIGTERM — not `finally`, not `except`, not `atexit`. `_snapshot` writes
    each artefact to a temp name and `os.replace`s it, so a stage killed
    between the two keeps the temp forever. Measured 2026-10-06: stopping one
    stage left `teach-asimov_getup-07942a-s4/model.tmp` at 5,018,382 bytes
    with no `model.zip` beside it, and every stopped job would leave one.

    The writer cannot fix this however it is written, which is why the sweep
    is the fix and lives where something outlives the kill.
    """
    run = tmp_path / "teach-stopped-s4"
    run.mkdir()
    for name in R.SNAPSHOT_TMPS:
        (run / name).write_bytes(b"half a checkpoint")
    keep = run / "live.onnx"            # the last GOOD snapshot
    keep.write_bytes(b"a real policy")

    gone = R.sweep_snapshot_tmps(run)

    assert sorted(gone) == sorted(R.SNAPSHOT_TMPS)
    assert not any((run / n).exists() for n in R.SNAPSHOT_TMPS)
    assert keep.read_bytes() == b"a real policy", (
        "the sweep must not touch the finished artefacts beside the temps"
    )


def test_the_sweep_leaves_a_temp_ANOTHER_process_may_be_writing(tmp_path):
    """The complement of the filter, which is the half worth checking.

    `write_record` makes its temp with `tempfile.mkstemp`, so the name carries
    a random infix and a concurrent writer may have one open RIGHT NOW — in
    this repo the lab and a trainer write the same run's record. A `*.tmp`
    glob in the sweep would therefore be a race that destroys a record
    mid-write, so the sweep names its three files explicitly.
    """
    run = tmp_path / "run"
    run.mkdir()
    (run / "model.tmp").write_bytes(b"stale")
    live = run / f"{R.RECORD_NAME}.a1b2c3.tmp"      # a mkstemp name, in flight
    live.write_bytes(b"being written")

    gone = R.sweep_snapshot_tmps(run)

    assert gone == ["model.tmp"]
    assert live.exists(), (
        "a record temp carries a random infix and may be open in another "
        "process — the sweep must not glob *.tmp"
    )


def test_the_sweep_covers_EVERY_temp_the_trainers_WRITE(tmp_path):
    """One name list, or the sweep silently stops covering one.

    Three trainers write snapshot temps (train.py, train_behavior.py,
    train_brain.py). If any of them adds or renames one and
    `SNAPSHOT_TMPS` is not updated, nothing fails — the stale file just
    accumulates again. So the names are compared against the source.
    """
    import re
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1] / "src" / "microduck_local"
    written: set[str] = set()
    for mod in ("train.py", "train_behavior.py", "train_brain.py"):
        src = (root / mod).read_text()
        # the literal temp names each trainer saves to, e.g. "model.tmp"
        written |= {m for m in re.findall(r'"([A-Za-z0-9_.]+\.tmp)"', src)}
    assert written, "no temp names found — did the writers change shape?"
    missing = written - set(R.SNAPSHOT_TMPS)
    assert not missing, (
        f"these temps are written but never swept: {sorted(missing)} — add "
        f"them to run_record.SNAPSHOT_TMPS"
    )


def test_the_lab_sweeps_when_the_TRAINER_EXITS_not_when_it_finishes_cleanly():
    """The stale temp is left by a STOP, so a sweep gated on success misses
    exactly the case it exists for. It must also run before
    `_advance_stage()` rebinds `self.dir` to the next stage's directory.
    """
    import ast
    import inspect

    src = inspect.getsource(V.TrainingJob)
    tree = ast.parse("class _C:\n" + "\n".join(
        "    " + ln for ln in src.splitlines()[1:]))
    sweeps = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "sweep_snapshot_tmps"]
    assert sweeps, "the lab must clear a dead trainer's temps"
    advance = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") == "_advance_stage"]
    assert min(n.lineno for n in sweeps) < min(n.lineno for n in advance), (
        "sweep before _advance_stage, or it cleans the NEXT stage's directory"
    )
    # ...and it must not sit INSIDE the clean-exit guard. Checked on the
    # tree, not on the text: the first spelling of "sweep_snapshot_tmps" in
    # this method is in a COMMENT, so a `src.index` comparison found the
    # prose, read as passing, and let a planted regression through — the same
    # dead-guard shape as the `_mde` units mismatch in verify_run.py.
    def mentions_done_record(node: ast.expr) -> bool:
        return any(isinstance(n, ast.Name) and n.id == "done_record"
                   for n in ast.walk(node))

    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not mentions_done_record(node.test):
            continue
        inside = [n for b in node.body for n in ast.walk(b)
                  if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", "") == "sweep_snapshot_tmps"]
        assert not inside, (
            "the sweep is inside a `done_record` guard: a sweep gated on a "
            "clean exit misses the STOP, which is the only case that leaves "
            "a temp behind"
        )


def test_every_trainer_clears_stale_temps_at_the_START_of_a_run():
    """A CLI run has no lab watching it exit, and a resumed directory would
    carry a previous kill's temp forward. All three trainers sweep at start.
    """
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1] / "src" / "microduck_local"
    for mod in ("train.py", "train_behavior.py", "train_brain.py"):
        src = (root / mod).read_text()
        assert "sweep_snapshot_tmps(out)" in src, (
            f"{mod} must clear a previous interrupted run's temps before it "
            f"starts writing into the same directory"
        )


def test_the_lab_sweeps_a_STOPPED_stage_whose_status_is_no_longer_training(tmp_path):
    """The sweep must not be gated on `status == "training"`.

    This is the mistake, caught live on 2026-10-06 after it had already been
    written, tested and reported as working. The obvious home for the sweep is
    the `poll()` branch that handles a trainer exiting — and that branch reads
    `if self.proc.poll() is not None and self.status == "training"`, while
    `stop()` sets `"stopped"` BEFORE the process exits, deliberately, because
    that is what keeps poll() from chaining the next stage. So the sweep sat
    in a branch a stop can never enter, and a stop is the ONLY thing that
    leaves a temp behind: planted `model.tmp` and `live.onnx.tmp` survived a
    real `/teach/stop` untouched, in a run with no `model.zip`, so nothing
    else could have taken them either.

    Driven through `poll()` on a stub rather than live, because live the
    trainer's own 25 k-step snapshot keeps winning the race and
    `os.replace`ing the planted temps away — three live attempts in a row
    were inconclusive for that reason, and two of them looked like passes.
    """
    import subprocess
    from types import SimpleNamespace

    run = tmp_path / "teach-stopped-s1"
    run.mkdir()
    for name in R.SNAPSHOT_TMPS:
        (run / name).write_bytes(b"half a checkpoint")

    dead = subprocess.Popen(["true"])
    dead.wait()
    job = SimpleNamespace(
        dir=run, proc=dead, _swept=set(), run_name="teach-stopped-s1",
        status="stopped",          # <- what stop() set, before the exit
        restarting=False, progress={}, _offset=0, _live_mtime=0.0,
        _fps_points=[],
        # poll() falls through to _snapshot_workers() for a job whose exit
        # branch it skips — which is the whole point here, so stub it.
        _snapshot_workers=lambda: [],
    )

    V.TrainingJob.poll(job)

    assert not any((run / n).exists() for n in R.SNAPSHOT_TMPS), (
        "a stopped stage's half-written snapshot must be cleared — a sweep "
        "reachable only while status == 'training' never runs on a stop"
    )
    assert job._swept == {str(run)}, "and it must record that it swept"

    # Idempotent: a stopped job keeps being polled every tick.
    (run / "model.tmp").write_bytes(b"written again, somehow")
    V.TrainingJob.poll(job)
    assert (run / "model.tmp").exists(), (
        "keyed by directory, so a dir is swept once — a second sweep would "
        "race any later writer"
    )
