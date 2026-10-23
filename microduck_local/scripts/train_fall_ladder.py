"""Train the fall-derived ladder BACKWARDS from `fall_7`, cumulatively.

    uv run python scripts/train_fall_ladder.py [--steps-per-stage N]

Each stage KEEPS every rung it has already learned and adds the next one down,
because training a stage only on its newest start made the robot forget: across
one 2.7 M-step reference chain, time-at-stand went 7.8 s, 3.3, 0.4, 0.3, 0.2 —
a 94 % collapse by the third stage that never came back.

The rungs are states recorded from the robot FALLING (`asimov_getup.fall_ladder`),
which is what makes this ladder different from the two that failed. The blend
ladder INTERPOLATES toward standing and its rungs are not poses the body passes
through: at alpha 0.8 the robot is one-footed, lands both feet only at height
0.885 / upright 0.83 — under the 0.90/0.90 gate — and has collapsed to 0.292 by
t = 1.0 s even carrying half its weight. The authored ladder routes through the
pike, whose trunk reaches 98 deg against the 50 deg the trained stand recovers
from. A fall state is reachable by construction, and a get-up is a fall
backwards.

From `fall_7` the standing gate already FIRES for five control steps
(height 0.918, upright +0.98, both feet, ~14 reward a step) before the robot
tips. So stage 1 is not asking the policy to discover standing — it is asking it
to hold a state it is handed.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

LAB = "http://127.0.0.1:8788"
RUNGS = [f"fall_{i}" for i in range(7, -1, -1)]      # fall_7 down to fall_0


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        LAB + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def status() -> dict:
    for _ in range(5):
        try:
            with urllib.request.urlopen(LAB + "/teach/status", timeout=10) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(2)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps-per-stage", type=int, default=1_500_000)
    ap.add_argument("--init-from", default="teach-asimov_getup_ref-1039ef-s8")
    a = ap.parse_args()

    donor = a.init_from
    for i in range(len(RUNGS)):
        label = ",".join(RUNGS[:i + 1])              # CUMULATIVE
        out = post("/teach", {
            "text": "get up off the floor", "robot": "asimov1",
            "initFrom": donor,
            "env": {"MICRODUCK_ASIMOV_GETUP_POSE": label},
            "steps": a.steps_per_stage, "weights": {}})
        job = out.get("job") or {}
        run = job.get("runName")
        if not run:
            print(f"[ladder] stage {i+1} refused: {out.get('message')}", flush=True)
            return 1
        print(f"[ladder] stage {i+1}/{len(RUNGS)}  {run}  starts={label}  "
              f"from {donor}", flush=True)

        last = -1
        while True:
            time.sleep(20)
            st = status()
            if not st:                                # a failed poll is not a result
                continue
            j = st.get("job") or {}
            if j.get("runName") != run:
                continue
            steps = j.get("steps") or 0
            if steps != last:
                last = steps
            if not st.get("running"):
                break
        # what the stage actually bought, from the run's own log
        try:
            rows = [json.loads(l) for l in
                    open(f"runs/{run}/progress.jsonl") if '"terms"' in l]
            t = rows[-1]["terms"]
            print(f"[ladder] stage {i+1} done: ep_rew {rows[-1]['ep_rew']:.1f}  "
                  f"standing {t.get('standing', 0):.1f}  stand {t.get('stand', 0):.1f}",
                  flush=True)
        except Exception as e:
            print(f"[ladder] stage {i+1} done (no log: {e})", flush=True)
        donor = run
    print(f"[ladder] FINISHED; final run {donor}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
