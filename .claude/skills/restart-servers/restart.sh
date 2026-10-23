#!/bin/bash
# Restart the microduck backend (duck-lab farm on :8788). Cwd-proof, rename-proof.
# Usage: restart.sh [--check] [--backend-only] [--fresh] [policy.onnx ...]
#   --check         run ONLY the guard and exit 0 if a restart would proceed,
#                   1 if it would refuse — printing the same reasons. This is
#                   the "check first, as a separate command" the skill doc asks
#                   for, except it is the guard's own verdict instead of a
#                   human reading /teach/status and deciding.
#   --backend-only  leave the viewer dev server (:63317) running — a lab-only
#                   code change has no reason to bounce it
#   --fresh         drop the saved roster (passed through to duck-lab)
#
# Only the process LISTENING on :8788 is stopped. This used to `pkill -f
# duck-lab`, which matches every lab on the machine — on 2026-09-17 another
# session had a scratch lab up on :8799 (`--world pitch-3v3 --fresh`) while
# the :8788 lab needed a restart, and the old script would have taken it
# down too. A port names exactly one lab; a process name does not.
set -u
# Repo-relative: works from any clone location.
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ML=$ROOT/microduck_local
LOG=$ML/lab-server.log
PORT=8788

BACKEND_ONLY=0
CHECK_ONLY=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --backend-only) BACKEND_ONLY=1 ;;
    --check) CHECK_ONLY=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

# ============================================================== THE GUARD
# A farm restart kills any live teach job with it. That happened twice by
# accident (2026-08-31, both times a check combined into the same command as
# the restart), so the script refuses on its own.
#
# WHAT it refuses for is the part that took two rounds to get right. The
# principle: refuse for anything that might be somebody's WORK; proceed only
# for what is POSITIVELY identified as a test. Anything unidentifiable still
# refuses — a hand-started CLI run is real work, and "I could not tell" is not
# a licence.
#
# Why "proceed" exists at all: this repo's own TESTS spawn real (short)
# training runs — `train_behavior run` / `imitate` with tiny `--steps`, in
# tests/test_net_arch.py, test_select_run.py, test_distill.py,
# test_train_resume.py. A `pytest tests/` in another window therefore put a
# matching trainer in the process table at random moments, the guard refused,
# and the refusal read "a trainer process is running" when nobody was
# training. On 2026-10-03 that cost a confused restart loop — and the
# documented escape hatch MICRODUCK_RESTART_FORCE=1 would have killed the TEST
# SUITE, the exact opposite of what that flag is for. A test trainer is also
# the one trainer a restart provably cannot hurt: `stop_port` signals only the
# pid LISTENING on :8788, and a pytest subprocess is not that lab's child.
#
# Every matching pid lands in exactly one of three buckets.
#
# (1) LAB-OWNED -> refuse, named. `/teach/status` publishes each live job's
#     pids (the trainer plus its vec-env workers — `TrainingJob.owned_pids`),
#     because the lab is the only thing that knows which trainer is its own.
#     Checked FIRST on purpose: the lab launches a trainer with
#     `env={**os.environ, ...}`, so a lab started from a pytest-tainted shell
#     would hand its teach job the pytest markers in (2) and the job would
#     read as a test. Ownership beats any evidence of testhood. An older lab
#     that predates the field publishes no pids and contributes nothing — its
#     trainer then falls through to (3) and still refuses.
#
# (2) POSITIVELY A TEST -> proceed, printing what was skipped and why. The
#     evidence is read off the trainer's OWN exec-time environment. The first
#     version of this walked the parent chain, and ancestry is the one signal
#     that evaporates exactly when it is needed: an ORPHANED test trainer (its
#     pytest already gone — vec-env workers outlive their parent by design
#     here, see TrainingJob._snapshot_workers) has no pytest above it at all.
#       * PYTEST_CURRENT_TEST in the environment. pytest sets it per test, and
#         every one of the spawn sites above builds its child env as
#         `dict(os.environ, ...)`, so it rides in — measured on this machine:
#         the child's env carries `PYTEST_CURRENT_TEST=<nodeid>` during both
#         the setup and the call phase. Nothing else sets that name, and an
#         exec'd environment never changes afterwards.
#       * MICRODUCK_RUNS_DIR under a `pytest-of-<user>/` directory. Every one
#         of those spawn sites points the trainer at `tmp_path`, and pytest's
#         tmp root is the only thing that builds that path component.
#         Deliberately NOT `/tmp` or `/private/tmp`: a human may legitimately
#         train into `MICRODUCK_RUNS_DIR=/tmp/scratch`, and that must keep
#         refusing.
#       * a pytest ancestor in the parent chain — kept as a third reading, for
#         a test that builds its child env by hand, but never the only one.
#     The environment comes from /proc/<pid>/environ on Linux and `ps -Ewww`
#     on macOS (no /proc there; `ps -E` appends the environment to the command
#     line). No answer = no evidence = bucket (3).
#
# (3) UNKNOWN -> refuse. A trainer nobody claims is assumed to be work.
#
# The lab's own `running` flag is still read INDEPENDENTLY of all three: it is
# what the lab polls off its subprocess, and it refuses on its own even if the
# pid bookkeeping ever disagrees — a job whose trainer the pattern below does
# not match at all would show up only there.
#
# A FORCE is not a free pass: it is the bypass that has killed another
# session's run FOUR times (2026-09-24, a dribble chain at 2.93M of 6M among
# them). The guard below is exactly what a FORCE skips, so a FORCE while
# someone else's job is up IS the kill. It stays available — editing lab
# Python is a real reason, and the running process holds stale code — but it
# now PRINTS WHAT IT IS ABOUT TO DESTROY, with the steps already spent, so
# the cost is visible at the moment of the decision rather than in an
# apology afterwards.

# BOTH trainers: the duck's `train_behavior` and a task run (`python -m
# microduck_local.train --robot g1 ...`), which the first pattern here missed.
# The bracket keeps the pattern from matching the pgrep's own command line;
# `grep -i python` keeps monitor/grep shells that merely MENTION the name in
# their command text from tripping it.
TRAINER_RE='microduck_local\.train_behavio[r]|microduck_local\.trai[n] '

# The lab's answer, fetched at most once per run. MICRODUCK_LAB_STATUS_URL
# overrides the URL and ONLY the guard's own test sets it (to a file:// JSON
# fixture) — the lab-owned case must be verifiable without starting real
# training, and a test that starts real training to check a guard against real
# training is a test nobody will run.
_STATUS_FETCHED=0
_STATUS_JSON=""
_teach_status() {
  if [ "$_STATUS_FETCHED" = 0 ]; then
    _STATUS_JSON="$(curl -s -m 2 \
      "${MICRODUCK_LAB_STATUS_URL:-http://127.0.0.1:$PORT/teach/status}" \
      2>/dev/null)"
    _STATUS_FETCHED=1
  fi
  printf '%s' "$_STATUS_JSON"
}

if [ "${MICRODUCK_RESTART_FORCE:-0}" = "1" ] && [ "$CHECK_ONLY" != 1 ]; then
  # `.format`, not an f-string: this was `f"{j[\"runName\"]} ..."`, and a
  # backslash inside an f-string expression is a SyntaxError on every Python
  # this repo runs — so with `2>/dev/null` swallowing it, `_doomed` was ALWAYS
  # empty and the "what this FORCE is about to destroy" warning had never once
  # printed. Found 2026-10-03 only because the guard's new test drove the
  # preamble with a fixture that had live jobs in it; there is no way to notice
  # it from a FORCE that goes through, because an empty list looks exactly like
  # "nothing is training".
  _doomed="$(_teach_status | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
for j in d.get("jobs", []):
    if j.get("status") in ("training", "restarting"):
        st, tot = j.get("steps") or 0, j.get("total") or 0
        print("    {}  {:,}/{:,} steps".format(j.get("runName"), st, tot))
' 2>/dev/null)"
  if [ -n "$_doomed" ]; then
    echo "FORCE: this restart will KILL these live training jobs:"
    echo "$_doomed"
    echo "  The lab runs several jobs at once — if you only need to START one,"
    echo "  POST /teach instead and leave these alone. If a job is not yours,"
    echo "  say so in the shared channel first and let it checkpoint."
    echo "  Continuing in 5s (Ctrl-C to abort)..."
    sleep 5
  fi
fi

# "pid argv" per running trainer, python processes only.
_trainer_lines() { pgrep -fl "$TRAINER_RE" 2>/dev/null | grep -i python; }
_trainer_pids() { _trainer_lines | awk '{print $1}'; }

# One pid per line: every pid the lab on :PORT says it owns. Silent (and empty)
# when the lab is down, answers something unparseable, or predates the field.
_lab_owned_pids() {
  _teach_status | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
for j in d.get("jobs") or []:
    for p in j.get("pids") or []:
        print(int(p))
' 2>/dev/null
}

# A process environment as text. Empty when it cannot be read — which is
# "no evidence", never "it is a test".
_proc_env() {
  if [ -r "/proc/$1/environ" ]; then          # Linux
    tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null
  else                                        # macOS: ps appends the env
    ps -Ewww -o command= -p "$1" 2>/dev/null
  fi
}

# Walk a pid's parent chain looking for pytest; stops at init, bounded depth.
_has_pytest_ancestor() {
  local parent=$1 depth=0
  while [ "$depth" -lt 8 ]; do
    parent=$(ps -o ppid= -p "$parent" 2>/dev/null | tr -d ' ')
    case "$parent" in ''|0|1) return 1 ;; esac
    case "$(ps -o command= -p "$parent" 2>/dev/null)" in
      *pytest*) return 0 ;;
    esac
    depth=$((depth + 1))
  done
  return 1
}

# Prints the evidence that pid $1 is a test trainer, or nothing at all.
_test_evidence() {
  local blob why=""
  blob="$(_proc_env "$1")"
  case "$blob" in *PYTEST_CURRENT_TEST=*) why="PYTEST_CURRENT_TEST" ;; esac
  # grep, not a glob: a glob's `*` spans spaces, so `MICRODUCK_RUNS_DIR=/x ...
  # SOMETHING=pytest-of-y` would pass a `*MICRODUCK_RUNS_DIR=*pytest-of-*`
  # case. The path has to be pytest's, not merely near something that is.
  if printf '%s\n' "$blob" \
     | grep -q 'MICRODUCK_RUNS_DIR=[^ ]*/pytest-of-'; then
    why="${why:+$why+}runs-dir in pytest-of-"
  fi
  if [ -z "$why" ] && _has_pytest_ancestor "$1"; then why="pytest parent"; fi
  printf '%s' "$why"
}

# "pid python <argv tail>" — the interpreter path alone is 100+ characters, so
# it is dropped. Far enough to reach `--run-name <name>`: the run name is what
# identifies a trainer to a human, and at the old width it was cut off exactly
# one token early on every G1/MARS/MOSS task run.
_argv_line() {
  ps -o command= -p "$1" 2>/dev/null | awk -v p="$1" \
    '{ out = p " python"; for (i = 2; i <= NF && i <= 10; i++) out = out " " $i; print out }'
}

_LAB_TRAINERS=""
_TEST_TRAINERS=""
_UNKNOWN_TRAINERS=""
_classify_trainers() {
  local owned pid why
  _LAB_TRAINERS=""; _TEST_TRAINERS=""; _UNKNOWN_TRAINERS=""
  # Padded with spaces at both ends so `*" $pid "*` cannot match a substring
  # of a longer pid (1234 inside 12345).
  owned=" $(_lab_owned_pids | tr '\n' ' ')"
  for pid in $(_trainer_pids); do
    case "$owned" in
      *" $pid "*)
        _LAB_TRAINERS="$_LAB_TRAINERS$(_argv_line "$pid")"$'\n'
        continue ;;
    esac
    why="$(_test_evidence "$pid")"
    if [ -n "$why" ]; then
      _TEST_TRAINERS="$_TEST_TRAINERS$(_argv_line "$pid")  [$why]"$'\n'
    else
      _UNKNOWN_TRAINERS="$_UNKNOWN_TRAINERS$(_argv_line "$pid")"$'\n'
    fi
  done
}

if [ "$CHECK_ONLY" = 1 ] || [ "${MICRODUCK_RESTART_FORCE:-0}" != "1" ]; then
  _classify_trainers
  _lab_training=0
  case "$(_teach_status)" in
    *'"running":true'*|*'"running": true'*) _lab_training=1 ;;
  esac
  if [ "$_lab_training" = 1 ] || [ -n "$_LAB_TRAINERS$_UNKNOWN_TRAINERS" ]; then
    echo "REFUSING to restart:"
    if [ -n "$_LAB_TRAINERS" ]; then
      echo "  the lab on :$PORT owns these trainer processes — its teach jobs,"
      echo "  which this restart would kill:"
      printf '%s' "$_LAB_TRAINERS" | sed 's/^/    /'
    fi
    if [ "$_lab_training" = 1 ]; then
      echo "  the lab on :$PORT says a teach job is training (\"running\":true)."
    fi
    if [ -n "$_UNKNOWN_TRAINERS" ]; then
      echo "  these trainer processes could not be identified, so they are"
      echo "  assumed to be somebody's work — which is what this guard is for:"
      printf '%s' "$_UNKNOWN_TRAINERS" | sed 's/^/    /'
    fi
    if [ -n "$_TEST_TRAINERS" ]; then
      echo "  (these others are this repo's TESTS and are not the reason:"
      printf '%s' "$_TEST_TRAINERS" | sed 's/^/    /'
      echo "  )"
    fi
    if [ -n "$_LAB_TRAINERS" ] || [ "$_lab_training" = 1 ]; then
      echo "Stop the job first:  curl -s -X POST http://127.0.0.1:$PORT/teach/stop"
    fi
    if [ -n "$_UNKNOWN_TRAINERS" ]; then
      echo "An unidentified trainer is not the lab's to stop — find out what it"
      echo "is (ps -Ewww -o command= -p <pid> shows who spawned it and where it"
      echo "writes), or wait for it."
    fi
    if [ "${MICRODUCK_RESTART_FORCE:-0}" = "1" ]; then
      echo "NOTE: MICRODUCK_RESTART_FORCE=1 is set, so a real restart would go"
      echo "      ahead and kill the above. --check reports the GUARD's verdict."
    else
      echo "or, to kill it deliberately:  MICRODUCK_RESTART_FORCE=1 restart.sh ..."
    fi
    exit 1
  fi
  if [ -n "$_TEST_TRAINERS" ]; then
    echo "trainer processes are running, and every one of them is POSITIVELY a"
    echo "test — this repo's suite spawns real short training runs:"
    printf '%s' "$_TEST_TRAINERS" | sed 's/^/    /'
    echo "  Proceeding. A restart stops only the pid LISTENING on :$PORT, and a"
    echo "  pytest subprocess is not that lab's child, so the suite survives it."
    echo "  (This is the case a FORCE used to be reached for, and a FORCE here"
    echo "  killed the tests instead of a teach job. Nothing to force now.)"
  fi
fi
if [ "$CHECK_ONLY" = 1 ]; then
  echo "--check: a restart would PROCEED (nothing running that it would harm)."
  exit 0
fi

# Stop whatever LISTENS on a port — and nothing else. TERM first; a listener
# still holding the port after 10 s gets KILL, by pid, never by name.
stop_port() {
  local port=$1 pids
  pids=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)
  if [ -z "$pids" ]; then
    echo "nothing listening on :$port"
    return 0
  fi
  for pid in $pids; do
    # argv without the interpreter path, which is 100+ characters on its own
    echo "stopping pid $pid: $(ps -o command= -p "$pid" 2>/dev/null | awk '{ for (i = 2; i <= NF; i++) printf "%s ", $i }' | cut -c1-90)"
    kill "$pid" 2>/dev/null
  done
  for _ in $(seq 1 20); do
    sleep 0.5
    lsof -nP -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 || return 0
  done
  for pid in $pids; do
    echo "pid $pid still holds :$port after 10 s — kill -9"
    kill -9 "$pid" 2>/dev/null
  done
  sleep 1
}

# Launch a server DETACHED IN ITS OWN SESSION. `nohup … &` is not enough: the
# child stays in this shell's process group, and when the Claude harness
# reaps a tool call it has backgrounded it kills that whole group — on
# 2026-09-10 that took the lab down hours after a bring-up had printed
# "lab: up". macOS has no `setsid`, so the new session comes from Python
# (start_new_session). Usage: detach LOGFILE cmd args…  (env prefixes pass
# through; the log is truncated first, as the old `> LOG` did).
detach() {
  local log=$1; shift
  : > "$log"
  python3 - "$log" "$@" <<'PY'
import os, subprocess, sys
log, cmd = sys.argv[1], sys.argv[2:]
# PYTHONUNBUFFERED: stdout to a FILE is block-buffered, so a detached server
# writes nothing to its log until ~8 KB accumulates. On 2026-10-04 a teach
# chain failed at stage 3 and `lab-server.log` was 0 bytes — the reason was
# sitting in a buffer in a process that had not crashed, so there was nothing
# to read and nothing to debug. A log you cannot read during the failure is not
# a log.
env = {**os.environ, "PYTHONUNBUFFERED": "1"}
with open(log, "ab") as fh:
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh,
                         stderr=subprocess.STDOUT, start_new_session=True,
                         env=env)
print(f"detached pid {p.pid} (its own session)")
PY
}

echo "[1/3] stopping the lab on :$PORT..."
stop_port "$PORT"
if [ "$BACKEND_ONLY" != 1 ]; then
  pkill -f "duck-viewer/node_modules/.bin/next" 2>/dev/null
  pkill -f "next dev -p 63317" 2>/dev/null
  sleep 2
fi

echo "[2/3] starting backend (duck-lab :$PORT)..."
# Entry-point name has churned (duck-farm -> duck-lab); resolve from pyproject.
ENTRY=$(grep -oE '^(duck-[a-z]+) *= *"microduck_local.viz_server:main"' "$ML/pyproject.toml" | cut -d' ' -f1)
ENTRY=${ENTRY:-duck-lab}
# ${ARGS[@]+...}: bash 3.2 (macOS) trips `set -u` on an empty array otherwise.
MICRODUCK_ACTUATOR=bam detach "$LOG" uv run --directory "$ML" "$ENTRY" --port "$PORT" ${ARGS[@]+"${ARGS[@]}"}

for i in $(seq 1 40); do
  sleep 2
  curl -s -m 2 "http://127.0.0.1:$PORT/joints" >/dev/null 2>&1 && { echo "backend UP on :$PORT"; ok=1; break; }
done
if [ "${ok:-0}" != 1 ]; then
  if grep -q "no ducks" "$LOG" 2>/dev/null; then
    echo "empty roster — retrying with the default walker..."
    MICRODUCK_ACTUATOR=bam detach "$LOG" uv run --directory "$ML" "$ENTRY" --port "$PORT" ../microduck/policies/alpha_walking.onnx
    for i in $(seq 1 40); do
      sleep 2
      curl -s -m 2 "http://127.0.0.1:$PORT/joints" >/dev/null 2>&1 && { echo "backend UP on :$PORT"; ok=1; break; }
    done
  fi
fi
if [ "${ok:-0}" != 1 ]; then
  echo "backend FAILED — last log lines:"; tail -8 "$LOG"; exit 1
fi
if [ "$BACKEND_ONLY" = 1 ]; then
  echo "[3/3] --backend-only: viewer (:63317) left as it was"
  exit 0
fi
echo "[3/3] starting viewer (:63317) detached..."
# Detached on purpose: the managed preview (preview_start) ties the dev server
# to the session's tool runtime and it is reaped on a session/model switch.
bash "$(dirname "$0")/viewer.sh" --force
