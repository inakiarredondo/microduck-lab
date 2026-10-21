"""Score a get-up in the DEPLOYMENT env under the published actuator envelope.

    uv run --with onnxruntime python scripts/score_getup_dc.py RUN [RUN ...] \
        --episodes 48 --pose supine,prone

The tracking env is where an imitation run TRAINS; it is not where the
deliverable lives. "Score at the horizon the deliverable is defined at, with
only the deployment's own cuts" -- so this builds the plain `getup` env, keeps
the run's `action_clip` and `actuator_force` (a policy judged under a clip it
cannot command, or an actuator it never saw, is a different controller), drops
the imitation knobs, and asks `asimov_getup_ref.is_standing`.

Runs are PAIRED by seed with an exact McNemar whenever more than one is given,
because both arms then see the same episodes.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: Knobs that belong to the TRACKING task and have no meaning in the
#: deployment env. Dropped by name rather than by a try/except on TypeError,
#: so a new one has to be classified instead of silently surviving.
TRACKING_ONLY = ("clip_file", "track_tol", "ref_speed", "free", "start_phase",
                 "handoff_pay", "force_lag")
#: NOT in the list above, deliberately: `history` is part of the POLICY's own
#: observation, not of the tracking task. Dropping it built a 78-float
#: deployment env for a 390-input policy and the normalizer died on
#: `(78,) vs (390,)` — which is the honest failure; the dangerous version
#: would have been an env that silently fitted.


def _acceptance():
    spec = importlib.util.spec_from_file_location(
        "_ga", HERE / "getup_acceptance.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--pose", default="supine,prone")
    ap.add_argument("--actuator", default="dc", choices=("xml", "dc"))
    a = ap.parse_args(argv)

    A = _acceptance()
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs

    print(f"deployment env (task=getup), actuator={a.actuator!r}, "
          f"{a.episodes} episodes per pose, paired by seed\n")
    out: dict[str, dict[str, list[bool]]] = {}
    for run in a.runs:
        rec = dict(replay_kwargs(Path("runs") / run))
        trained = rec.get("actuator_force", "xml")
        for k in ("task", "getup_label", *TRACKING_ONLY):
            rec.pop(k, None)
        rec["actuator_force"] = a.actuator
        rec.setdefault("limit_cost", 1.0)
        act = A.load(run)
        out[run] = {}
        for pose in [p.strip() for p in a.pose.split(",") if p.strip()]:
            env = S.get("asimov1").env_class("getup")(
                domain_rand=False, obs_noise=True, action_delay=True, seed=0,
                getup_label=pose, **rec)
            rises, stays = [], []
            for s in range(a.episodes):
                o, _ = env.reset(seed=s)
                if hasattr(act, "reset"):
                    act.reset()
                rose = last = False
                for _k in range(int(env.max_steps)):
                    o, _r, t, tr, _i = env.step(act(o))
                    last = R.is_standing(env)
                    rose = rose or last
                    if t or tr:
                        break
                rises.append(rose)
                stays.append(last)
            out[run][pose] = {"rises": rises, "stays": stays}
            print(f"  {run[-30:]:30s} {pose:7s} trained_on={trained:4s} "
                  f"rises {sum(rises):3d}/{a.episodes}  "
                  f"stays {sum(stays):3d}/{a.episodes}")
    if len(a.runs) > 1:
        base = a.runs[0]
        print()
        for run in a.runs[1:]:
            for pose in out[run]:
                for metric in ("rises", "stays"):
                    b = out[base][pose][metric]
                    c = out[run][pose][metric]
                    bb, cc, pv = A.mcnemar(b, c)
                    verdict = ("real at p<=0.05" if pv <= 0.05
                               else "NULL — inside the noise")
                    print(f"  {pose:7s} {metric:6s} {base[-12:]} "
                          f"{sum(b):3d} vs {run[-12:]} {sum(c):3d} | "
                          f"discordant {bb}-{cc} | p = {pv:.4f} | {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
