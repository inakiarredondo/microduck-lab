"""Score a NARROW stand from fold: holds, narrowness, span — paired.

The deliverable for the narrow track is not "did it stand" but "did it stand
WITHOUT the splits". `getup_acceptance` reports a span but gates on the
get-up; this scores the narrow stand directly and pairs two runs on the same
seeds with an exact McNemar, because a 4-episode margin on 48 is not a result
(this track has already had one, at p = 0.125, that needed 144 episodes to
resolve).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

#: Knobs that belong to a different task and would either raise or quietly
#: change what is being measured. `history` is NOT here: it is part of the
#: policy's own observation, and dropping it builds a 78-float env for a
#: 390-input policy.
DROP = ("task", "clip_file", "track_tol", "ref_speed", "free", "start_phase",
        "handoff_pay", "force_lag", "getup_label")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--pose", default="fold")
    ap.add_argument("--gate", type=float, default=0.40)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    print(f"narrow stand from {a.pose!r}, {a.episodes} episodes, "
          f"obs noise + action delay, gate {a.gate:g} m\n")
    out = {}
    for run in a.runs:
        kw = dict(replay_kwargs(Path("runs") / run))
        for k in DROP:
            kw.pop(k, None)
        kw.update(getup_label=a.pose, fell_cut=False, max_episode_s=8.0)
        env = S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
        pol = GA.load(run)
        holds, narrow, spans = [], [], []
        for s in range(a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            mx, up = 0.0, False
            for _k in range(int(env.max_steps)):
                o, _r, t, tr, _i = env.step(pol(o))
                mx = max(mx, float(env._foot_span()))
                up = R.is_standing(env)
                if t or tr:
                    break
            holds.append(up)
            narrow.append(bool(up and mx <= a.gate))
            spans.append(mx)
        out[run] = (holds, narrow)
        print(f"  {run[-30:]:32s} holds {sum(holds):3d}/{a.episodes}  "
              f"NARROW {sum(narrow):3d}/{a.episodes}  "
              f"median span {np.median(spans):.3f} m")

    if len(a.runs) > 1:
        base = a.runs[0]
        print()
        for run in a.runs[1:]:
            for lbl, i in (("holds", 0), ("narrow", 1)):
                x, y = out[base][i], out[run][i]
                bb, cc, pv = GA.mcnemar(x, y)
                verdict = ("real at p<=0.05" if pv <= 0.05
                           else "NULL — inside the noise")
                print(f"  {lbl:7s} {base[-12:]} {sum(x):3d} vs "
                      f"{run[-12:]} {sum(y):3d} | discordant {bb}-{cc} | "
                      f"p = {pv:.4f} | {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
