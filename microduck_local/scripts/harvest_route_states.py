"""Harvest the STATES a feasible get-up passes through, as a start distribution.

WHY THIS EXISTS. The authored reference never closes the centre-of-mass to
feet gap: per-foot distance is 0.577 at `pike` and 0.137 at `fold` with
nothing in between, and uprightness goes to **-0.37** at `pike` (face down).
Twelve reward economies, a 4x budget and two new env knobs all failed to
cross that, every one satisfied by a degenerate behaviour.

But the task is FEASIBLE in this exact env: the vendor's shipped policy rises
12/12 from prone under the real `dc` envelope, median 1.63 s. Traced, its
route looks nothing like the reference -- uprightness stays POSITIVE
throughout (+0.19 -> +1.00), both feet are down by 1.0 s, and the per-foot
distance closes GRADUALLY and monotonically (0.651 -> 0.115) through a WIDE
stance (span 0.89) that narrows only at the end (0.22).

So the continuum of intermediates exists; the authored waypoints simply miss
it. This records those states so a reverse curriculum can start from them.

WHAT IS AND IS NOT TAKEN. Only `qpos`/`qvel` -- geometry the body
demonstrably passes through. No actions, no observations, no distillation
loss, no policy weights. A policy trained from these starts learns its own
actions from scratch; the demonstration supplies only the question "can you
recover from here?".
"""
from __future__ import annotations

import argparse

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--per-band", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from microduck_local.lab import robots as lab_robots
    from microduck_local.robots import asimov as _asimov
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_openhorizon as OH
    from microduck_local.robots import spec as S

    entry = next(e for e in S.get("asimov1").shipped_policies()
                 if e.get("adapter") == "openhorizon:getup")
    kw = dict(entry["envKwargs"])
    kw["actuator_force"] = "dc"          # the REAL envelope, not the flat clamp
    kw["getup_label"] = "prone"
    env = lab_robots.slot_env("asimov1", seed=0, kwargs=kw)
    pol = OH.OpenHorizonPolicy("getup")

    qs, vs, zs, ups, pers = [], [], [], [], []
    rose = 0
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        pol.reset()
        traj = []
        for k in range(int(env.max_steps)):
            obs, _r, t, tr, _i = env.step(pol(obs))
            traj.append((env.data.qpos.copy(), env.data.qvel.copy(),
                         float(env.data.xpos[env.trunk_body_id][2]),
                         float(R.standing_report(env).get("upright", 0.0)),
                         float(env._com_foot_per())))
            if t or tr:
                break
        # ONLY episodes that actually ended standing contribute states: a
        # failed attempt's states are not known to be recoverable, which is
        # the whole property this cache is supposed to carry.
        if not R.standing_report(env).get("standing"):
            continue
        rose += 1
        for q, v, z, u, pf in traj:
            qs.append(q); vs.append(v); zs.append(z); ups.append(u); pers.append(pf)

    qs = np.asarray(qs); vs = np.asarray(vs)
    zs = np.asarray(zs); ups = np.asarray(ups); pers = np.asarray(pers)
    out = a.out or str(_asimov.CACHE_DIR / "route_states.npz")
    np.savez_compressed(out, qpos=qs, qvel=vs, trunk_z=zs, upright=ups,
                        per_foot=pers)
    print(f"{rose}/{a.episodes} episodes stood; {len(qs)} states -> {out}")
    for lo, hi in ((0.50, 0.60), (0.40, 0.50), (0.30, 0.40), (0.20, 0.30),
                   (0.10, 0.20), (0.00, 0.10)):
        m = (zs >= lo) & (zs < hi)
        if m.any():
            print(f"  trunk {lo:.2f}-{hi:.2f}: {int(m.sum()):5d} states | "
                  f"upright {np.median(ups[m]):+.2f} | "
                  f"per-foot {np.median(pers[m]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
