"""Search pike->fold on the JOINT-SPACE objective, which nothing has used.

Every previous search of this transition — `search_pike_fold`,
`search_narrow_kip`, `search_crawl_in` — scored trunk HEIGHT or UPRIGHTNESS.
Today's measurement says that is the wrong target: a state matching `fold`'s
trunk to 30 mm at uprightness +0.73, feet-only and narrow, is caught **0 of
78** times by the policy that stands 43/48 from `fold` itself, because it sits
**0.290 rad mean / 1.214 rad max** away in joint space (hips 0.186 against
fold's 1.400 — `fold` is a deep CROUCH and that state was straight-legged).
The catcher's basin is a neighbourhood of fold's CONFIGURATION.

So this scores mean joint error against the `fold` waypoint, over the whole
trajectory, and starts from the route's own arrival at `pike` — the last rung
the servos reach unaided (pelvis 0.289 m, forearms down, span 0.214 m). Under
`dc`, so the envelope is Menlo's.

Reaching ~0.1 rad would BE the missing demonstration: that is the distance the
catcher's own start poses sit at, and from there it stands 43/48. Reporting the
best distance honestly matters even if it is far — it bounds what a demonstration
could look like, which no previous search did on this axis.
"""
from __future__ import annotations

import argparse
import numpy as np

CLIP = 4.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--knots", type=int, default=4,
                    help="waypoints the trajectory is free to pass through")
    ap.add_argument("--leg-s", type=float, default=0.30)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=a.seed,
        getup_label="prone", actuator_force="dc", action_clip=CLIP,
        fell_cut=False, max_episode_s=20.0)
    m, d = env.model, env.data
    JN = list(W.JOINT_NAMES)
    qadr = [int(m.joint(n).qposadr[0]) for n in JN]
    pel = m.body("pelvis_link").id
    default = np.asarray(env.robot.default_pose, float)
    scale = np.asarray(env.robot.action_scale, float)
    q_fold = np.asarray(R.BY_NAME["fold"].joints, float)
    nj = len(JN)

    # --- the START: ramp the servos to `pike`, the last rung they reach.
    names = [w.name for w in R.WAYPOINTS]
    route = names[:names.index("pike") + 1]
    env.reset(seed=a.seed)
    w0 = R.BY_NAME[route[0]]
    env._place(np.asarray(w0.joints, float), w0.root_pitch)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    cur = np.asarray(w0.joints, float)
    for nm in route[1:]:
        tgt = np.asarray(R.BY_NAME[nm].joints, float)
        for k in range(13):
            q = cur + (tgt - cur) * min(1.0, (k + 1) / 12)
            env.step(np.clip((q - default) / scale, -CLIP, CLIP).astype(np.float32))
        cur = tgt
    q0, v0 = d.qpos.copy(), d.qvel.copy()
    q_start = np.asarray(d.qpos[qadr], float)
    print(f"  start = the route's arrival at `pike`: pelvis "
          f"{float(d.xpos[pel][2]):.3f} m, span {env._foot_span():.3f} m, "
          f"joint distance to fold {np.abs(q_start - q_fold).mean():.3f} rad")
    print(f"  on floor: "
          f"{', '.join(str(x) for x in R.floor_parts(env)) or 'NOTHING'}\n")

    n_leg = max(1, int(a.leg_s / 0.02))

    def roll(p):
        d.qpos[:], d.qvel[:] = q0, v0
        env._obs_stack, env._obs_live = [], False
        mujoco.mj_forward(m, d)
        knots = p.reshape(a.knots, nj)
        q = q_start.copy()
        best = (9e9, 0.0, 0.0, 0.0)
        for i in range(a.knots):
            # The knot is an OFFSET from fold, not an absolute pose, so the
            # search starts near the target rather than in a random corner.
            tgt = q_fold + knots[i]
            src = q.copy()
            for k in range(n_leg):
                f = (k + 1) / n_leg
                env.step(np.clip(((src + (tgt - src) * f) - default) / scale,
                                 -CLIP, CLIP).astype(np.float32))
                err = float(np.abs(
                    np.asarray(d.qpos[qadr], float) - q_fold).mean())
                if err < best[0]:
                    parts = [str(x) for x in R.floor_parts(env)]
                    feet = sum(1 for x in parts if "foot" in x)
                    best = (err, float(d.xpos[pel][2]),
                            float(env._foot_span()),
                            1.0 if (feet >= 2 and len(parts) == feet) else 0.0)
            q = tgt
        return best

    rng = np.random.default_rng(a.seed)
    overall = None
    for r in range(a.restarts):
        mu = np.zeros(a.knots * nj)
        sig = np.full(a.knots * nj, 0.35)
        best = None
        for it in range(a.iters):
            pop = rng.normal(mu, sig, (a.pop, a.knots * nj))
            pop[0] = mu
            rows = [roll(x) for x in pop]
            order = np.argsort([x[0] for x in rows])[:max(4, a.pop // 10)]
            mu = pop[order].mean(0)
            sig = pop[order].std(0) + 1e-3
            if best is None or rows[order[0]][0] < best[0][0]:
                best = (rows[order[0]], pop[order[0]].copy())
            e, z, sp, fo = best[0]
            print(f"  {r}.{it:<3d} joint err {e:.3f} rad   pelvis {z:.3f}   "
                  f"span {sp:.3f}   feet-only {fo:.0f}", flush=True)
        if overall is None or best[0][0] < overall[0][0]:
            overall = best
    e, z, sp, fo = overall[0]
    print(f"\n  BEST: joint distance to `fold` {e:.3f} rad, at pelvis "
          f"{z:.3f} m, span {sp:.3f} m, feet-only {int(fo)}")
    print(f"  the catcher's own start poses sit ~0.10 rad from fold and it "
          f"stands 43/48 from them.")
    print(f"  the route's arrival at pike is "
          f"{np.abs(q_start - q_fold).mean():.3f} rad away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
