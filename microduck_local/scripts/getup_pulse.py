"""IS THIS CHAIN ALIVE? Score the LIVE checkpoint mid-training, in seconds.

    uv run python scripts/getup_pulse.py [host:port] [episodes]

`train-loop` blocks until a run ENDS, and `watch-training` renders frames for
a human to read. Neither answers the question that actually matters while
23 M steps are grinding: *is this working, or am I about to spend half an hour
on a dead chain?*

On 2026-10-10 a from-scratch get-up chain ran 12 M steps while Jonathan
watched it roll around, and the only check available was "wait for the end".
It rose 0/24 at stage 4 — which turned out to be exactly where the previous
chain had been, but nobody could know that without stopping to measure. This
is that measurement, cheap enough to run every few million steps.

It scores `live.onnx` — the trainer's own rolling export, normalizer baked —
on the three numbers that decide whether a get-up chain is going anywhere:

    rises      does it reach a stand from the floor at all
    stays      is it still standing at the buzzer
    best hold  the longest unbroken stand

and prints them beside the ZERO-ACTION floor for the same env, because a
number that does not beat doing nothing is not progress.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"


def live_job(addr: str) -> dict:
    with urllib.request.urlopen(f"http://{addr}/teach/status", timeout=8) as r:
        return (json.loads(r.read().decode()).get("job") or {})


def score(onnx: Path | None, run: Path, episodes: int) -> dict:
    import onnxruntime as ort

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs

    kw = dict(replay_kwargs(run))
    kw.pop("task", None)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    sess = (ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            if onnx is not None else None)
    rose = stayed = 0
    holds, rets = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        cur = best = 0
        total = 0.0
        up = False
        for _ in range(int(env.max_steps)):
            a = (np.zeros(env.action_space.shape, np.float32) if sess is None
                 else sess.run(None, {"obs": obs.reshape(1, -1)
                                      .astype(np.float32)})[0][0])
            obs, r, t, tr, _ = env.step(np.asarray(a, np.float32))
            total += float(r)
            up = R.is_standing(env)
            cur = cur + 1 if up else 0
            best = max(best, cur)
            if t or tr:
                break
        rose += best > 0
        stayed += up
        holds.append(best / 50.0)
        rets.append(total)
    return dict(rises=rose, stays=stayed, hold=max(holds),
                ret=float(np.mean(rets)), n=episodes)


def main() -> int:
    addr = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8788"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    job = live_job(addr)
    run = job.get("runName")
    if not run:
        print("[pulse] nothing is training")
        return 1
    d = RUNS / str(run)
    live = d / "live.onnx"
    if not live.is_file():
        print(f"[pulse] {run}: no live.onnx yet (no snapshot has landed)")
        return 1
    print(f"[pulse] {run}  {job.get('steps'):,}/{job.get('total'):,}")
    floor = score(None, d, max(4, episodes // 3))
    got = score(live, d, episodes)
    print(f"   zero action : rises {floor['rises']}/{floor['n']}   "
          f"stays {floor['stays']}/{floor['n']}   return {floor['ret']:8.1f}")
    print(f"   LIVE        : rises {got['rises']}/{got['n']}   "
          f"stays {got['stays']}/{got['n']}   return {got['ret']:8.1f}   "
          f"best hold {got['hold']:.2f}s")
    verdict = ("RISING" if got["rises"] else
               ("above the floor, but not standing yet"
                if got["ret"] > floor["ret"] else "NOT BEATING ZERO ACTION"))
    print(f"   -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
