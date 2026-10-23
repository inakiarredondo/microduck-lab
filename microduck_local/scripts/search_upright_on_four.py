"""Can the robot get UPRIGHT while the forearms are still down — and hold it?

Uprightness is the single binding clause of the arrival: of the six conditions,
span passes 97.1 % of frames, joints 79.4 %, feet-only 35.3 %, height 33.3 %,
lift 30.4 % — and uprightness >= 0.70 only **2.9 %**, with all six never
coinciding. It is not a kinematic bound: `fold` is uprightness +0.83, feet
only, at pelvis 0.511 m, and it holds. So it is a TRAJECTORY problem — the
trunk cannot be got up on the way.

The four-contact search already run targeted HEIGHT and happened to return
poses at uprightness -0.19 to +0.07, which answered a different question. This
one maximises UPRIGHTNESS directly, with the forearms allowed down and the
1.5 s hold inside the objective. If the robot can be upright and stable on four
contacts, uprightness stops being the blocker and the remaining move is only to
unload the hands — which `arm_unload` already demonstrably does (feet-only
frames 0.0 % -> 35.3 %).

Scored on what SURVIVES the hold, not on what is touched: a pose that is
upright for one frame on the way over is the `a-stand-is-not-a-getup` failure.
"""
from __future__ import annotations

import argparse
import numpy as np

LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--span-max", type=float, default=0.30)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--from-fold", action="store_true",
                    help="seed the CEM at `fold`'s own configuration. Without "
                         "this the prior is the midpoint of the joint ranges, "
                         "which is nowhere near fold's crouch: unseeded, the "
                         "search returns 0.0 while fold's own config scores "
                         "105.0 through this very pipeline — an optimisation "
                         "failure that reads as impossibility.")
    ap.add_argument("--lowest", action="store_true",
                    help="objective becomes: the LOWEST pelvis that is still "
                         "upright (>=0.70), holding, and narrow. That is the "
                         "real question — how far down the unholdable band an "
                         "upright stable pose reaches — and it is only "
                         "answerable from a feasible seed.")
    ap.add_argument("--save", help="write the winning pose's 23 joint values "
                                   "and root pitch to this .npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=a.seed,
        getup_label="prone", actuator_force="dc", action_clip=4.0,
        fell_cut=False, max_episode_s=30.0)
    m, d = env.model, env.data
    JN = list(W.JOINT_NAMES)
    qs = {n: int(m.joint(n).qposadr[0]) for n in JN}
    lim = {n: m.jnt_range[m.joint(n).id].copy() for n in JN}
    free = [f"{s}_{k}_joint" for s in ("left", "right") for k in LEG
            if f"{s}_{k}_joint" in qs]
    free += [n for n in JN if any(k in n for k in
             ("shoulder", "elbow", "wrist", "waist"))]
    pel = m.body("pelvis_link").id
    dflt = np.asarray(env.robot.default_pose, float)
    sc = np.asarray(env.robot.action_scale, float)

    def put(x):
        d.qpos[:] = 0.0
        d.qpos[2] = 0.70
        pitch = float(x[-1])
        c, s = np.cos(pitch / 2), np.sin(pitch / 2)
        d.qpos[3:7] = [c, 0.0, s, 0.0]
        for n, v in zip(free, x[:-1]):
            lo, hi = lim[n]
            d.qpos[qs[n]] = float(np.clip(v, lo, hi))
        d.qvel[:] = 0.0
        R.settle_onto_floor(m, d, clearance=-0.001)
        mujoco.mj_forward(m, d)

    def state():
        parts = [str(v) for v in R.floor_parts(env)]
        feet = sum(1 for v in parts if "foot" in v)
        other = sum(1 for v in parts if "foot" not in v
                    and "forearm" not in v.lower())
        return (float(-env._projected_gravity()[2]),
                float(d.xpos[pel][2]), float(env._foot_span()), feet, other,
                sum(1 for v in parts if "forearm" in v.lower()))

    def score(x):
        put(x)
        # THE PRE-GATE MUST NOT BE READ BEFORE A STEP. Judged at the settled
        # state, `fold` itself reports feet 0 and "on floor: NOTHING" — the
        # settle leaves it a hair clear — so a `feet >= 2` pre-gate rejects a
        # pose that is perfectly good one step later. That is what made the
        # first run of this search return 0.0 while `fold`, scored the same
        # way, scores 105.0: an optimisation that never reached the region,
        # reported as impossibility. One step first.
        q_t0 = np.asarray([d.qpos[qs[n]] for n in JN], float)
        act0 = np.clip((q_t0 - dflt) / sc, -4.0, 4.0).astype(np.float32)
        env._obs_stack, env._obs_live = [], False
        env.step(act0)
        u0, z0, sp0, fe0, ot0, fa0 = state()
        if ot0 or fe0 < 2 or sp0 > a.span_max:
            return -100.0 - 10.0 * ot0, (u0, z0, sp0, fe0, fa0, 0.0)
        act = act0
        for _ in range(int(a.hold_s / 0.02)):
            env.step(act)
        u1, z1, sp1, fe1, ot1, fa1 = state()
        kept = (ot1 == 0 and fe1 >= 2 and sp1 <= a.span_max
                and z1 > z0 - 0.05)
        # UPRIGHTNESS is the objective; survival is a hard multiplier, because
        # an upright pose that collapses answers nothing.
        if a.lowest:
            # Pay for being LOW, but only while upright, holding and narrow.
            # Hard gates, so a low pose that is not upright scores nothing —
            # the whole point is how far down uprightness survives.
            if not kept or u1 < 0.70:
                return -100.0, (u1, z1, sp1, fe1, fa1, float(kept))
            return (100.0 * (0.62 - z1) + 20.0,
                    (u1, z1, sp1, fe1, fa1, float(kept)))
        return (100.0 * u1 * float(kept) + 20.0 * float(kept)
                - 20.0 * max(0.0, sp1 - a.span_max),
                (u1, z1, sp1, fe1, fa1, float(kept)))

    rng = np.random.default_rng(a.seed)
    lo = np.array([lim[n][0] for n in free] + [-np.pi / 2])
    hi = np.array([lim[n][1] for n in free] + [0.6])
    print(f"  maximise UPRIGHTNESS, forearms allowed, span <= {a.span_max:g}, "
          f"surviving {a.hold_s:g} s")
    print(f"  {'':6s}{'best':>9s}{'upright':>9s}{'pelvis':>8s}{'span':>7s}"
          f"{'feet':>6s}{'forearm':>9s}{'held':>6s}")
    best = None
    for r in range(a.restarts):
        if a.from_fold:
            fj = np.asarray(R.BY_NAME["fold"].joints, float)
            mu = np.array([fj[JN.index(n)] for n in free]
                          + [float(R.BY_NAME["fold"].root_pitch)])
            mu = mu + rng.normal(0, 0.05 + 0.10 * r, len(lo))
            sig = np.full(len(lo), 0.12)
        else:
            mu = (lo + hi) / 2 + rng.normal(0, 0.2, len(lo))
            sig = (hi - lo) / 4
        for it in range(a.iters):
            pop = np.clip(rng.normal(mu, sig, (a.pop, len(lo))), lo, hi)
            rows = [score(x) for x in pop]
            order = np.argsort([-v[0] for v in rows])[:max(4, a.pop // 10)]
            mu = pop[order].mean(0)
            sig = pop[order].std(0) + 1e-3
            if best is None or rows[order[0]][0] > best[0]:
                best = (rows[order[0]][0], rows[order[0]][1],
                        pop[order[0]].copy())
            u, z, sp, fe, fa, kept = best[1]
            print(f"  {r}.{it:<3d}{best[0]:9.1f}{u:+9.2f}{z:8.3f}{sp:7.3f}"
                  f"{int(fe):6d}{int(fa):9d}{int(kept):6d}", flush=True)
    u, z, sp, fe, fa, kept = best[1]
    print(f"\n  BEST held pose: upright {u:+.2f}, pelvis {z:.3f} m, span "
          f"{sp:.3f} m, {int(fe)} feet + {int(fa)} forearm(s) down")
    ok = kept and u >= 0.70
    if a.lowest:
        print(f"  {'the LOWEST upright holding pose found' if ok else 'no upright holding pose found'}"
              f" — fold is 0.511 m, the band bottom is 0.30 m")
    else:
        print(f"  {'UPRIGHT AND HOLDING' if ok else 'nothing upright survived (check the seed: unseeded this search cannot find fold)'}")
    print(f"  the arrival needs upright >= 0.70; `fold` is +0.83 on feet alone")
    if a.save and ok:
        x = best[2]
        put(x)
        q = np.asarray([d.qpos[qs[n]] for n in JN], float)
        np.savez(a.save, joints=q, root_pitch=float(x[-1]),
                 upright=u, pelvis=z, span=sp)
        print(f"  pose -> {a.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
