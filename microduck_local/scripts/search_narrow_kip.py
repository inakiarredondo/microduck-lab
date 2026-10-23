"""Search for a DYNAMIC narrow rise — a kip — from the servo route's endpoint.

Everything tried so far is quasi-static: servo ramps between poses, or RL from
stable starts. Both are bounded by the same fact — no narrow static rung exists
below `fold`, and the CoM sits a leg length ahead of the feet, so support
cannot be put under mass without widening.

A kip is not bounded by that. It rotates the trunk with ANGULAR MOMENTUM
rather than static balance, and it is narrow by nature (feet together). Humans
rise from their backs this way. Nothing in this track has searched for one.

Starts from the end of the authored SERVO route, which is already narrow
(0.215 m) and costs nothing to reach: trunk ~0.20, upright ~-0.34. Searches a
long-horizon joint trajectory with enough knots and amplitude to express a
swing, and scores only the pose that SURVIVES a hold — mid-motion instability
is the POINT, so nothing is required of the intermediate poses.
"""
from __future__ import annotations

import argparse
import numpy as np

CLIP = 4.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--elites", type=int, default=10)
    ap.add_argument("--knots", type=int, default=8)
    ap.add_argument("--move-s", type=float, default=2.0)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--span-max", type=float, default=0.35)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=a.seed,
        getup_label="prone", actuator_force="dc", action_clip=CLIP,
        fell_cut=False, max_episode_s=30.0)
    m, d = env.model, env.data
    nj = len(W.JOINT_NAMES)
    default = np.asarray(env.robot.default_pose, float)
    scale = np.asarray(env.robot.action_scale, float)
    names = [w.name for w in R.WAYPOINTS]
    route = names[:names.index("fold") + 1]

    # --- the narrow servo route, played once; its end is the search's start
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
    print(f"servo route ends at trunk {float(d.xpos[env.trunk_body_id][2]):.3f}"
          f"  upright {float(-env._projected_gravity()[2]):+.2f}"
          f"  span {env._foot_span():.3f}  (narrow, and free to reach)\n")

    q_start = np.asarray(d.qpos[[int(m.joint(n).qposadr[0])
                                 for n in W.JOINT_NAMES]], float)
    n_move, n_hold = int(a.move_s / 0.02), int(a.hold_s / 0.02)
    t_knot = np.linspace(0, 1, a.knots + 1)
    t_step = np.linspace(0, 1, n_move)

    def roll(res):
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_forward(m, d)
        env._obs_stack = []
        env._obs_live = False
        knots = np.vstack([q_start, q_start + res.reshape(a.knots, nj)])
        path = np.stack([np.interp(t_step, t_knot, knots[:, j])
                         for j in range(nj)], 1)
        span = 0.0
        for k in range(n_move + n_hold):
            q = path[min(k, n_move - 1)]
            env.step(np.clip((q - default) / scale, -CLIP, CLIP).astype(np.float32))
            span = max(span, float(env._foot_span()))
        up = float(-env._projected_gravity()[2])
        z = float(d.xpos[env.trunk_body_id][2])
        parts = [str(p) for p in R.floor_parts(env)]
        feet = sum(1 for p in parts if "foot" in p)
        other = sum(1 for p in parts if "foot" not in p)
        return up, z, span, feet, other

    def score(res):
        up, z, span, feet, other = roll(res)
        if other > 0 or feet < 2:
            return -1000.0 + 100.0 * z + 20.0 * up, (up, z, span, feet, other)
        s = 400.0 * z + 150.0 * up
        s -= 300.0 * max(0.0, span - a.span_max)
        return s, (up, z, span, feet, other)

    rng = np.random.default_rng(a.seed)
    mu = np.zeros(a.knots * nj)
    sig = np.full(a.knots * nj, a.sigma)
    best = (*score(mu),)
    best = (best[0], mu.copy(), best[1])
    print(f"  {'iter':>5s}{'best':>9s}{'upright':>9s}{'trunk':>8s}"
          f"{'span':>7s}{'feet':>6s}{'other':>7s}")
    for it in range(a.iters):
        pop = rng.normal(mu, sig, (a.pop, mu.size))
        pop[0] = mu
        rows = [score(p) for p in pop]
        order = np.argsort([-r[0] for r in rows])[:a.elites]
        mu = pop[order].mean(0)
        sig = pop[order].std(0) + 1e-3
        top = rows[order[0]]
        if top[0] > best[0]:
            best = (top[0], pop[order[0]].copy(), top[1])
        print(f"  {it:5d}{best[0]:9.1f}{best[2][0]:+9.2f}{best[2][1]:8.3f}"
              f"{best[2][2]:7.3f}{best[2][3]:6d}{best[2][4]:7d}", flush=True)

    up, z, span, feet, other = roll(best[1])
    print(f"\nBEST after a {a.hold_s:g} s hold: upright {up:+.2f}, trunk {z:.3f}, "
          f"peak span {span:.3f} m, {feet} feet / {other} other parts down")
    print(f"  a STAND is trunk ~0.64 upright +1.00 on 2 feet and nothing else")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
