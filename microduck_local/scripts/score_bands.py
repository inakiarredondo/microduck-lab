"""SCORE A GET-UP PER BAND — with uprightness beside every height.

THE CANONICAL BAND SCORER. `asimov_getup_ref.standing_report` is the one
definition of "did it stand" and a test already fails the build on a private
copy of it — but nothing stopped a REPORT from quoting a height on its own,
and on 2026-10-19/20 that happened twice:

* a five-run series was reported as progress on median peak trunk z
  (0.255 -> 0.422) while uprightness at that same frame went -0.43 -> -0.66.
  The policy was lifting its trunk while going further FACE DOWN, and
  `standing` was 0/24 in every one of those runs.
* a later run's "peak 0.616" at 0.26-0.30 was a climb to standing HEIGHT at
  uprightness -0.33, i.e. inverted.

So this prints, for every band, in this order: **STANDS** (the deliverable,
scored on the pose that SURVIVES), **ever** (did it stand at any instant —
separates a hold failure from a rise failure), and only then a height, which
is never shown without `up@peak` next to it. A height column alone is how
both mistakes were made.

    uv run python scripts/score_bands.py <run> [--episodes 20] [--floor]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: Trunk-height bands, top-down. Fine enough to see a frontier MOVE: a coarse
#: 0.30-0.38 band straddled a real physical boundary (both feet above it, a
#: loaded forearm below) and read as a flat wall for four interventions.
_DEFAULT = None

#: STILLNESS is a 50 Hz phenomenon and a contact sheet samples at ~1.4 Hz, so
#: a render CANNOT show it — the frames of a jiggling stand look identical to
#: a still one. It took a human watching the mp4 to notice. These two columns
#: exist because every table printed during the get-up work tracked stands,
#: hold, height, uprightness and span, and NONE of them covered the one thing
#: that made the result look bad: 2.56 rad/s of joint velocity while standing
#: and 0.183 rad off the idle pose, against 0.021 and 0.036 for the policy
#: that holds still.
BANDS = ((0.40, 0.46), (0.36, 0.40), (0.33, 0.36), (0.30, 0.33),
         (0.26, 0.30), (0.20, 0.26), (0.12, 0.20), (0.00, 0.12))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--floor", action="store_true",
                    help="also score prone and supine, the deliverable itself")
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as _W
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs

    global _DEFAULT
    _DEFAULT = np.asarray(_W.DEFAULT_POSE, float)

    sp = importlib.util.spec_from_file_location("_ga", HERE / "getup_acceptance.py")
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    base = dict(replay_kwargs(Path("runs") / a.run))
    base.pop("task", None)
    base.pop("start_rung", None)
    for k in ("start_phase", "free", "track_tol", "ref_speed", "force_lag",
              "arrival", "cut_back"):
        base.pop(k, None)
    pol = GA.load(a.run)

    def score(kw0, label):
        kw = dict(base)
        kw.update(kw0)
        kw.update(fell_cut=False, max_episode_s=8.0)
        try:
            env = S.get("asimov1").env_class("getup")(
                domain_rand=False, obs_noise=True, action_delay=True,
                seed=0, **kw)
        except Exception as e:                       # a band the cache lacks
            print(f"  {label:<14s}  refused: {str(e)[:48]}")
            return None
        st, ever, pk, upk, hold = [], [], [], [], []
        jig, offi = [], []          # stillness, measured only WHILE standing
        for s in range(a.seed0, a.seed0 + a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            m, ua, seen, h, best = -1.0, float("nan"), False, 0, 0
            for _k in range(int(env.max_steps)):
                o, _r, t, tr, _i = env.step(pol(o))
                z = float(env.data.xpos[env.trunk_body_id][2])
                rep = R.standing_report(env)
                if z > m:
                    m, ua = z, float(rep.get("upright", float("nan")))
                if rep.get("standing"):
                    seen = True
                    h += 1
                    best = max(best, h)
                    if h > 25:           # settled: 0.5 s after it first stood
                        jig.append(float(np.abs(env._joint_vel()).mean()))
                        q = env.data.qpos[-len(_DEFAULT):]
                        offi.append(float(np.abs(q - _DEFAULT).mean()))
                else:
                    h = 0
                if t or tr:
                    break
            st.append(bool(R.standing_report(env).get("standing")))
            ever.append(seen); pk.append(m); upk.append(ua); hold.append(best * 0.02)
        n = a.episodes
        print(f"  {label:<14s}{sum(st):5d}/{n:<3d}{sum(ever):5d}/{n:<3d}"
              f"{np.median(hold):9.2f}{np.median(pk):9.3f}"
              f"{np.median(upk):+9.2f}"
              f"{(np.mean(jig) if jig else float('nan')):9.2f}"
              f"{(np.mean(offi) if offi else float('nan')):9.3f}")
        return sum(st), sum(ever)

    print(f"{a.run}   {a.episodes} eps/band, seeds {a.seed0}-{a.seed0 + a.episodes - 1}")
    print("  a STAND is trunk up AND upright AND feet-only, on the pose that "
          "SURVIVES.\n  'ever' separates a HOLD failure from a RISE failure. "
          "A height is never\n  shown without the uprightness at that same frame.")
    print(f"\n  {'band':<14s}{'STANDS':>8s}{'ever':>8s}{'hold s':>9s}"
          f"{'peak z':>9s}{'up@peak':>9s}{'jiggle':>9s}{'off-idle':>9s}")
    for lo, hi in BANDS:
        score({"start_rung": f"route:{lo},{hi}"}, f"{lo:.2f}-{hi:.2f}")
    if a.floor:
        print()
        for lbl in ("prone", "supine"):
            score({"getup_label": lbl}, f"FLOOR {lbl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
