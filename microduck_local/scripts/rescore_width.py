"""Re-score every Asimov policy that LEAVES THE FLOOR by how WIDE it gets.

The 0.40 m acceptance gate has no derivation, so it was never the right way to
rank these. The deliverable is "narrow enough to work", which is a judgement
about a number rather than a threshold to pass — so this reports the number,
for every policy that actually rises, and ranks by it.

Three widths, because a peak over the episode hides where the width is spent:

  * PEAK   — the worst instant anywhere in the episode (what the old gate used)
  * BAND   — median span while the pelvis is inside 0.30-0.52 m, the crossing
  * STAND  — median span once up (above 0.52 m), i.e. the stance it ends in

The crossing is the one that matters for hardware: `fold` is 0.215 m and the
vendor's own working rise spends 0.84-0.95 m there while standing at 0.239 m.
A policy that crosses wide and stands narrow is doing the splits on the way up,
whatever its final stance looks like.

Scored on the pose that SURVIVES (`is_standing`), from `prone`, under the real
`dc` envelope, with each run's own recorded kwargs.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

DROP = ("task", "clip_file", "track_tol", "ref_speed", "free", "start_phase",
        "handoff_pay", "force_lag", "getup_label", "prologue")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--pose", default="prone")
    ap.add_argument("--openhorizon", action="store_true", default=True)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    runs = list(a.runs)
    if not runs:
        runs = sorted(p.name for p in Path("runs").glob("asimov-getup-h5*")
                      if (p / "policy.onnx").is_file())

    rows = []

    def measure(label, pol, env):
        m, d = env.model, env.data
        pel = m.body("pelvis_link").id
        rise = 0
        peaks, band, stand = [], [], []
        for s in range(a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            mx, up = 0.0, False
            for _k in range(int(env.max_steps)):
                o, _r, t, tr, _i = env.step(np.asarray(pol(o), np.float32))
                z = float(d.xpos[pel][2])
                spn = float(env._foot_span())
                mx = max(mx, spn)
                if 0.30 <= z < 0.52:
                    band.append(spn)
                elif z >= 0.52:
                    stand.append(spn)
                up = R.is_standing(env)
                if t or tr:
                    break
            rise += int(up)
            peaks.append(mx)
        rows.append((label, rise, a.episodes, float(np.median(peaks)),
                     float(np.median(band)) if band else float("nan"),
                     float(np.median(stand)) if stand else float("nan")))
        print(f"  ...{label[-34:]:36s} rises {rise}/{a.episodes}", flush=True)

    for run in runs:
        try:
            kw = dict(replay_kwargs(Path("runs") / run))
            for k in DROP:
                kw.pop(k, None)
            kw.update(getup_label=a.pose, fell_cut=False, max_episode_s=10.0)
            env = S.get("asimov1").env_class("getup")(
                domain_rand=False, obs_noise=True, action_delay=True,
                seed=0, **kw)
            measure(run, GA.load(run), env)
        except Exception as e:
            print(f"  ...{run[-34:]:36s} SKIPPED ({type(e).__name__})",
                  flush=True)

    if a.openhorizon:
        from microduck_local.robots import asimov_openhorizon as OH
        env = S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=True, action_delay=True, seed=0,
            getup_label=a.pose, actuator_force="dc", action_clip=4.0,
            history=1, fell_cut=False, max_episode_s=10.0)
        measure("openhorizon (vendor)", OH.OpenHorizonPolicy(), env)

    up = [r for r in rows if r[1] > 0]
    print(f"\n  POLICIES THAT LEAVE THE FLOOR, ranked by CROSSING width "
          f"(from '{a.pose}', {a.episodes} eps each)\n")
    print(f"  {'policy':<34s}{'rises':>8s}{'PEAK':>8s}{'BAND':>8s}"
          f"{'STAND':>8s}")
    for r in sorted(up, key=lambda r: (np.nan_to_num(r[4], nan=9e9), r[3])):
        print(f"  {r[0][-32:]:<34s}{f'{r[1]}/{r[2]}':>8s}{r[3]:8.3f}"
              f"{r[4]:8.3f}{r[5]:8.3f}")
    if not up:
        print("  NONE of them left the floor.")
    print(f"\n  reference: `fold` 0.215 m, the robot is 1.2 m tall.")
    print(f"  BAND = median span crossing pelvis 0.30-0.52 m (the splits live "
          f"here)")
    print(f"  {len(rows) - len(up)} of {len(rows)} policies never rose and are "
          f"not ranked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
