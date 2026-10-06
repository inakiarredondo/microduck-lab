"""Score an Asimov get-up run — the report `pick_report.py` cannot give.

`pick_report.py` and `reward_budget.py` are MOSS's: they build a 32-float
observation and feed it to the policy, so on a 78-float Asimov export they do
not produce a wrong answer, they raise. This is the Asimov equivalent and it
reports the only number that matters for `getup_ref`: **which waypoint the
policy reaches**, over seeds, from the deterministic export.

The run's OWN flags, read off `run.json`, which is the whole reason Step 0
existed — the curriculum rung used to live in `os.environ` and every evaluation
read it from the ambient shell. `train-loop`'s rule 1: evaluating a policy under
flags it did not train with does not look like an error, it looks like a result.

    uv run python scripts/getup_report.py runs/teach-asimov_getup_ref-69021a-s4
    uv run python scripts/getup_report.py runs/<a> runs/<b> --episodes 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: Every knob an evaluation must match the run's training on. `history` is in
#: here for the same reason as the rest and it bit immediately: without it the
#: env hands a 78-float frame to a policy trained on 390 and onnxruntime
#: raises — which is the LUCKY failure. The three cases `train-loop` records
#: all reported a plausible NUMBER instead.
EVAL_KEYS = ("start_phase", "track_tol", "ref_speed", "getup_label",
             "getup_blend", "getup_assist", "history")


def run_flags(run: Path) -> dict:
    """The env kwargs this run TRAINED with, as recorded."""
    meta = json.loads((run / "run.json").read_text())
    kw = meta.get("env_kwargs") or {}
    out = {k: kw[k] for k in EVAL_KEYS if k in kw}
    if not out:
        raise SystemExit(
            f"{run.name} records no get-up knobs in env_kwargs — it predates "
            "AsimovBody.train_env_kwargs, so there is no way to know which "
            "rung it trained at and nothing here can score it honestly")
    return out, meta.get("task") or "getup_ref"


def episode(env, sess, name, seed):
    obs, _ = env.reset(seed=seed)
    info: dict = {}
    for _ in range(env.max_steps + 1):
        a = sess.run(None, {name: obs.reshape(1, -1).astype(np.float32)})[0][0]
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return info


def report(run: Path, episodes: int) -> dict:
    import onnxruntime as ort

    from microduck_local.robots import asimov_getup_ref as REF
    from microduck_local.robots import asimov_walk as W

    flags, task = run_flags(run)
    sess = ort.InferenceSession(str(run / "policy.onnx"),
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    env = W.spec().env_class(task)(domain_rand=False, obs_noise=False, **flags)

    names = [w.name for w in REF.WAYPOINTS]
    hist = {n: 0 for n in names}
    held, stood, steps = [], 0, []
    for s in range(episodes):
        info = episode(env, sess, name, 1000 + 100 * s)
        hist[info.get("phase_name", names[0])] += 1
        held.append(float(info.get("held_s", 0.0)))
        stood += int(bool(info.get("reached_stand")))
        steps.append(int(info.get("phase", 0)))
    return {"run": run.name, "flags": flags, "hist": hist, "held": held,
            "stood": stood, "episodes": episodes,
            "start": flags.get("start_phase", "?")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--episodes", type=int, default=16)
    args = ap.parse_args()

    from microduck_local.robots import asimov_getup_ref as REF
    order = [w.name for w in REF.WAYPOINTS]

    for run in args.runs:
        d = report(run, args.episodes)
        start_i = order.index(d["start"]) if d["start"] in order else 0
        print(f"\n{d['run']}   start={d['start']}  "
              f"tol={d['flags'].get('track_tol')}")
        print(f"  reached `stand`: {d['stood']}/{d['episodes']}"
              f"   median held {np.median(d['held']):.2f} s"
              f"   best held {max(d['held']):.2f} s")
        print("  waypoint reached (the > marks where it started):")
        for i, n in enumerate(order):
            c = d["hist"][n]
            mark = ">" if i == start_i else " "
            bar = "#" * c
            tag = ""
            if n == "fold":
                tag = "   <- THE LEG THE SERVOS CANNOT DO"
            print(f"   {mark} {n:<16} {c:>3} {bar}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
