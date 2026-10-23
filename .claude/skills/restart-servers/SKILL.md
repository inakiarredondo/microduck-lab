---
name: restart-servers
description: Restart the microduck dev stack — the duck-lab backend (farm, :8788) and the duck-viewer dev server (:63317). Use when the viewer shows "offline", after editing viz_server/behaviors (the farm holds stale code), or whenever the user asks to restart the dev/backend servers.
---

Restart both microduck servers, in this order:

1. **WARNING — training dies with the farm.** A farm restart kills any live
   teach job. Its checkpoint survives; resume after with
   `POST /teach {"text": ..., "initFrom": "<run>"}`. The script refuses on its
   own when there is something to protect. Check first anyway, **as a separate
   command**, never combined with the restart — and ask the guard itself rather
   than reading the lab and deciding for it:

   ```
   bash .claude/skills/restart-servers/restart.sh --check
   ```

   `--check` runs only the guard and exits 0 ("a restart would PROCEED") or 1,
   printing the same reasons the restart would. It touches nothing.

   **What it refuses for.** It sorts every running trainer — `train_behavior`
   (duck) and `microduck_local.train` (G1 / MARS / MOSS task) — into three
   buckets, and refuses unless all of them are clear:

   | bucket | how it is decided | verdict |
   |---|---|---|
   | the lab's own teach jobs | their pids, from `GET /teach/status` (each job's `pids`: trainer + vec-env workers) | **refuse**, named |
   | positively a test | `PYTEST_CURRENT_TEST` in the trainer's own environment, or `MICRODUCK_RUNS_DIR` inside a `pytest-of-…` dir, or a pytest ancestor | proceed, listed with the evidence |
   | anything else | — | **refuse** |

   Plus one independent reading: the lab's `GET /teach/status` saying
   `"running": true` refuses on its own, whatever the process table says.

   **What it now proceeds through.** This repo's TESTS spawn real short
   training runs (`train_behavior run` / `imitate` with tiny `--steps` — see
   `tests/test_net_arch.py`, `test_select_run.py`, `test_distill.py`,
   `test_train_resume.py`), so a `pytest tests/` in another window used to trip
   the guard at random moments; on 2026-10-03 that cost a confused restart
   loop, and a FORCE there would have killed the test suite rather than a teach
   job. A test trainer is also the one trainer a restart provably cannot
   touch — only the pid LISTENING on :8788 is stopped, and a pytest subprocess
   is not that lab's child. So the guard now identifies them and goes ahead,
   printing what it skipped. **A trainer it cannot identify still refuses**:
   that is the safety property, and a hand-started CLI run looks exactly like
   it.

   **To tell them apart by hand**, ask the process, not the lab — its
   environment names the test that spawned it:

   ```
   ps -Ewww -o command= -p <pid> | tr ' ' '\n' | grep -E '^(PYTEST_CURRENT_TEST|MICRODUCK_RUNS_DIR)='
   ```

   A `PYTEST_CURRENT_TEST=tests/test_x.py::test_y` is a test, full stop, even
   when its pytest is long gone — which happens, so do not reach for the parent
   chain first. Nothing there means nothing is claiming it; treat it as work.

   Only `MICRODUCK_RESTART_FORCE=1` overrides the refusal, and it should now be
   needed only for a real job you mean to kill. `tests/test_restart_guard.py`
   locks all of this against planted process trees.

2. Run the script (stops the lab, restarts it cwd-proof and
   entry-point-rename-proof, verifies :8788 health, then bounces the viewer):

   ```
   bash .claude/skills/restart-servers/restart.sh
   ```

   After a lab-only code change (viz_server, behaviors, world, brain…) leave
   the viewer dev server alone — it hot-reloads on its own:

   ```
   bash .claude/skills/restart-servers/restart.sh --backend-only
   ```

   Pass `--fresh` plus `.onnx` paths to reseed the duck roster; with no args
   the saved roster (lab-state.json) is kept.

   **Only the process LISTENING on :8788 is stopped** (`lsof … -sTCP:LISTEN`,
   TERM, then KILL by pid after 10 s). It used to `pkill -f duck-lab`, which
   matched every lab on the machine: on 2026-09-17 another session had a
   scratch lab on :8799 (`--world pitch-3v3`) that the old script would have
   killed along with the real one. A scratch lab on another port — with its
   own `LAB_STATE_PATH=…` so it never rewrites the real `lab-state.json` —
   survives a restart of :8788 now.

3. The script now starts the viewer too, detached, by calling `viewer.sh`.
   Nothing more to do. To (re)start only the viewer:

   ```
   bash .claude/skills/restart-servers/viewer.sh          # no-op if already up
   bash .claude/skills/restart-servers/viewer.sh --force  # restart regardless
   ```

   **Do NOT use `preview_start` for the viewer.** The managed preview makes the
   dev server a child of the session's tool runtime, so it is reaped on a
   session or model switch — that killed it on 2026-09-02 and looked like a
   crash. `viewer.sh` reparents it to launchd (PPID 1), like the backend, so it
   survives. Open the viewer with `navigate` to a normal browser tab instead.

   **Use `localhost`, not `127.0.0.1`, for the viewer.** Next 16 serves the
   HTML to either host but 403s its own `/_next` dev resources when the host is
   not in `allowedDevOrigins`, so `127.0.0.1:63317` renders a blank page with
   an empty `<body>` and console 403s — it looks like the app is broken when
   only the origin is wrong. `http://localhost:63317/sim` renders fine. (The
   backend is the opposite: curl it at 127.0.0.1:8788, and note `/` is a 404
   there because it has no root route — use `/joints` for a health check.)

4. Confirm frames flow: one WS read from ws://127.0.0.1:8788/ws should return
   a frame with a non-empty `ducks` list. Tell the user to refresh the page.

Logs: backend `microduck_local/lab-server.log`, viewer
`duck-viewer/viewer-server.log` (both gitignored).

Known gotchas these scripts already handle: the shell cwd resets to the parent
repo (breaks bare `uv run`); the backend entry point was renamed
(duck-farm → duck-lab) and may churn again; a stale Next lock file can block
viewer spawns (`viewer.sh` kills the port holder with `lsof` first); a managed
preview would not survive a session restart (hence the detached launch). And — since 2026-09-10 — **each
server starts in its own session** (`detach` in both scripts: Python's
`start_new_session`, because macOS has no `setsid`). `nohup … &` alone left
the server in the tool call's process group, and when the Claude harness
reaped a tool call it had backgrounded, the lab went down with it hours after
"backend UP". The `sim-smoke` bring-up delegates every launch here for that
reason; from an agent shell, never start these servers any other way.
