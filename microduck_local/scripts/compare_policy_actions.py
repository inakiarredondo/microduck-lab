"""Are these DIFFERENT controllers, or the same one with drift?

Everything on the Asimov stage descends from OpenHorizon's demonstrations:
`h5pd2` is a clone of them, and today's `809b71` is a fine-tune of a fine-tune
of that clone. Two policies can look different in one rollout and still be the
same function — rollouts diverge from tiny differences — so the honest test is
to feed them the SAME observations and compare what they OUTPUT.

Driven by one reference policy so every policy sees an identical state
sequence. Reports per-action RMS difference against the reference's own action
scale, and the correlation. Near-zero difference means the fine-tune is
cosmetic; a large one means the controller really changed.
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
    ap.add_argument("--driver", default="asimov-getup-h5pd2")
    ap.add_argument("--others", default="teach-asimov_getup_clip-809b71,"
                                        "teach-asimov_getup_clip-1edbc6")
    ap.add_argument("--pose", default="supine")
    ap.add_argument("--episodes", type=int, default=6)
    a = ap.parse_args(argv)

    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    kw = dict(replay_kwargs(Path("runs") / a.driver))
    for k in DROP:
        kw.pop(k, None)
    kw.update(getup_label=a.pose, fell_cut=False, max_episode_s=8.0)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0, **kw)

    drv = GA.load(a.driver)
    names = [x for x in a.others.split(",") if x]
    pols = {n: GA.load(n) for n in names}

    ref, out = [], {n: [] for n in names}
    for s in range(a.episodes):
        o, _ = env.reset(seed=s)
        for p in (drv, *pols.values()):
            if hasattr(p, "reset"):
                p.reset()
        for _k in range(int(env.max_steps)):
            ad = np.asarray(drv(o), np.float32)
            ref.append(ad)
            for n, p in pols.items():
                out[n].append(np.asarray(p(o), np.float32))
            o, _r, t, tr, _i = env.step(ad)
            if t or tr:
                break
    R = np.array(ref)
    print(f"  driver: {a.driver}  ({len(R)} steps from '{a.pose}', all "
          f"policies seeing the SAME observations)\n")
    print(f"  {'policy':<34s}{'RMS diff':>10s}{'vs action sd':>14s}"
          f"{'corr':>8s}{'verdict':>22s}")
    sd = float(R.std())
    for n in names:
        A = np.array(out[n])
        rms = float(np.sqrt(((A - R) ** 2).mean()))
        cc = float(np.corrcoef(A.ravel(), R.ravel())[0, 1])
        rel = rms / max(sd, 1e-9)
        v = ("SAME controller" if rel < 0.10 else
             "drifted" if rel < 0.40 else "DIFFERENT controller")
        print(f"  {n[-32:]:<34s}{rms:10.4f}{rel:13.2f}x{cc:8.3f}{v:>22s}")
    print(f"\n  the driver's own action sd is {sd:.4f}; 'vs action sd' is the")
    print(f"  difference as a fraction of that. Rollouts diverge from tiny")
    print(f"  differences, so only this comparison can tell drift from change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
