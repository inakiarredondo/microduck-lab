"""Is the sprawl OUT OF SPEC, or just alarming to look at?

The whole narrow track assumes a wide crossing is unacceptable, and the 0.40 m
gate it was judged against has no derivation. Before treating the splits as a
defect, ask the engineering question: during the crossing, does the robot
exceed Menlo's published JOINT LIMITS or their published TORQUE envelope?

If it stays inside both, "it does the splits" is a statement about appearance
and floor space, not about damage — and the deliverable may already exist.
If it exceeds either, the sprawl is a real problem with a named cause.

Joint limits come from the MJCF (the vendor's own ranges, at the sha pinned
for the hardware). Torque headroom is `robots/asimov_motor.headroom`, priced
against the window actually in force at that instant under `dc`.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="asimov-getup-h5pd2")
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    kw = dict(replay_kwargs(Path("runs") / a.run))
    for k in ("task", "clip_file", "track_tol", "ref_speed", "free",
              "start_phase", "handoff_pay", "force_lag", "getup_label",
              "prologue"):
        kw.pop(k, None)
    kw.update(getup_label="prone", fell_cut=False, max_episode_s=10.0)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    m, d = env.model, env.data
    JN = list(W.JOINT_NAMES)
    qadr = np.array([int(m.joint(n).qposadr[0]) for n in JN])
    lo = np.array([m.jnt_range[m.joint(n).id][0] for n in JN])
    hi = np.array([m.jnt_range[m.joint(n).id][1] for n in JN])
    pel = m.body("pelvis_link").id
    pol = GA.load(a.run)

    worst_lim = np.zeros(len(JN))
    worst_head = 0.0
    band_lim, band_head = [], []
    rises = 0
    for s in range(a.episodes):
        o, _ = env.reset(seed=s)
        if hasattr(pol, "reset"):
            pol.reset()
        up = False
        for _k in range(int(env.max_steps)):
            o, _r, t, tr, _i = env.step(np.asarray(pol(o), np.float32))
            q = np.asarray(d.qpos[qadr], float)
            # how far OUTSIDE its own range is each joint, in rad
            over = np.maximum(np.maximum(lo - q, q - hi), 0.0)
            worst_lim = np.maximum(worst_lim, over)
            mot = getattr(env, "_motor", None)
            # headroom() is PER JOINT; the episode's number is the worst one.
            h = float(np.max(np.asarray(mot.headroom(), float))) \
                if mot is not None else float("nan")
            if np.isfinite(h):
                worst_head = max(worst_head, h)
            z = float(d.xpos[pel][2])
            if 0.30 <= z < 0.52:
                band_lim.append(float(over.max()))
                if np.isfinite(h):
                    band_head.append(h)
            if R.is_standing(env):
                up = True
            if t or tr:
                break
        rises += int(up)

    print(f"  {a.run}, {a.episodes} eps from 'prone' — rises {rises}/"
          f"{a.episodes}\n")
    k = int(np.argmax(worst_lim))
    print(f"  JOINT LIMITS (the vendor's own MJCF ranges)")
    print(f"    worst excursion past a limit: {worst_lim[k]:.4f} rad "
          f"({np.degrees(worst_lim[k]):.2f} deg) on {JN[k]}")
    n_over = int((worst_lim > 1e-4).sum())
    print(f"    joints that ever left their range: {n_over}/{len(JN)}")
    print(f"    inside the crossing, worst excursion "
          f"{(max(band_lim) if band_lim else 0.0):.4f} rad")
    print()
    print(f"  TORQUE (Menlo's published speed-torque envelope, `dc`)")
    print(f"    worst headroom used, whole episode: {worst_head:.3f}  "
          f"(1.000 = exactly at the envelope)")
    if band_head:
        print(f"    inside the crossing: median {np.median(band_head):.3f}, "
              f"max {max(band_head):.3f}")
    print()
    ok_lim = worst_lim.max() <= 1e-3
    ok_tq = worst_head <= 1.001
    print(f"  >> joints {'INSIDE spec' if ok_lim else 'OUT of spec'}; "
          f"torque {'INSIDE the envelope' if ok_tq else 'OVER the envelope'}")
    if ok_lim and ok_tq:
        print("     the sprawl is WIDE, not out of spec — it is a statement")
        print("     about floor space and appearance, not about damage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
