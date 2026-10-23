"""Paired comparison of two get-ups on the SAME episodes, with McNemar.

8/12 against 10/12 is 4-vs-2 discordant pairs and nowhere near significant,
but this track has repeatedly read a marginal count as an effect. `AGENTS.md`:
compare PAIRED when both arms see the same episodes, and read the MDE, not the
p — a null here means "no instrument", not "no difference".
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
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--episodes", type=int, default=48)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    def score(run):
        kw = dict(replay_kwargs(Path("runs") / run))
        for k in DROP:
            kw.pop(k, None)
        kw.update(getup_label="prone", fell_cut=False, max_episode_s=10.0)
        env = S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
        m, d = env.model, env.data
        pel = m.body("pelvis_link").id
        ids = tuple(int(m.geom_bodyid[env.foot_geoms[s]])
                    for s in ("left", "right"))
        pol = GA.load(run)
        out, spans = [], []
        for s in range(a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            ok, mx = False, 0.0
            for _k in range(int(env.max_steps)):
                o, _r, t, tr, _i = env.step(np.asarray(pol(o), np.float32))
                z = float(d.xpos[pel][2])
                if 0.30 <= z < 0.52:
                    v = np.asarray(d.xpos[ids[0]], float) - np.asarray(
                        d.xpos[ids[1]], float)
                    mx = max(mx, float(np.hypot(v[0], v[1])))
                ok = R.is_standing(env)
                if t or tr:
                    break
            out.append(ok)
            spans.append(mx)
        return out, spans

    xa, sa = score(a.a)
    xb, sb = score(a.b)
    bb = sum(1 for p, q in zip(xa, xb) if p and not q)
    cc = sum(1 for p, q in zip(xa, xb) if q and not p)
    from math import comb
    n = bb + cc
    pv = (1.0 if n == 0 else
          min(1.0, 2.0 * sum(comb(n, k) for k in range(min(bb, cc) + 1))
              / 2 ** n))
    print(f"  {a.a[-30:]:32s} {sum(xa):3d}/{a.episodes}  "
          f"crossing span {np.median(sa):.3f} m")
    print(f"  {a.b[-30:]:32s} {sum(xb):3d}/{a.episodes}  "
          f"crossing span {np.median(sb):.3f} m")
    print(f"\n  paired on the SAME {a.episodes} episodes: discordant "
          f"{bb}-{cc}, exact McNemar p = {pv:.4f}")
    print(f"  {'NULL — the two are indistinguishable on rises' if pv > 0.05 else 'a real difference at p<=0.05'}")
    d_span = np.median(sa) - np.median(sb)
    print(f"  span difference: {d_span:+.3f} m "
          f"({'tighter' if d_span < 0 else 'wider'} for {a.a[-16:]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
