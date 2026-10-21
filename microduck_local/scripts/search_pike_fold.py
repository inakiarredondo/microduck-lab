"""Search for a dynamically feasible pike -> fold rotation, under the REAL envelope.

    uv run python scripts/search_pike_fold.py --iters 15 --pop 48

Why this exists, and why it is not a repeat. `pike -> fold` is the one
transition a narrow Asimov get-up needs and nothing performs: the authored
route collapses through it, the openhorizon teacher falls out of pike
entirely, and the best clone is 0/12 there. Six optimisers have failed on it
before — and every one of them ran under the FLAT actuator model, +/-45 N*m
with no speed dependence, which is a robot that is weak at rest. This move is
a slow braced rotation, i.e. a near-stall motion, and the published envelope
(`robots/asimov_motor`) grants up to **3x** the continuous torque there — 120
N*m at the hip against 45. That is the regime the motion lives in and it has
never been searched.

The search is CEM over a RESIDUAL on the straight-line joint path, so the
seed is the obvious motion and the optimiser only has to find what is wrong
with it. The start state is a pike the robot ARRIVED at, by playing the servo
prologue, not a placed pose — an arrived pike carries real velocity and a
placed one does not.

**It is scored on what SURVIVES, not on what is touched.** After the motion
the final targets are held for `--hold` seconds and the pose is read at the
END. Every scorer in this repo that read a peak instead has lied: a
best-so-far height scored a face-down hip-throw as a stand.
"""
from __future__ import annotations

import argparse
import numpy as np

CLIP = 4.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--elites", type=int, default=8)
    ap.add_argument("--knots", type=int, default=4)
    ap.add_argument("--move-s", type=float, default=1.2)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--span-max", type=float, default=0.30)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actuator", default="dc", choices=("xml", "dc"))
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=a.seed,
        getup_label="prone", actuator_force=a.actuator, action_clip=CLIP,
        fell_cut=False, prologue="1", max_episode_s=20.0)
    m, d = env.model, env.data
    nj = len(W.JOINT_NAMES)
    default = np.asarray(env.robot.default_pose, float)
    scale = np.asarray(env.robot.action_scale, float)

    # --- the arrived pike: play the servo prologue and snapshot the state
    env.reset(seed=a.seed)
    while env.in_prologue():
        env.step(np.zeros(nj, np.float32))      # the env substitutes the plan
    q0, v0 = d.qpos.copy(), d.qvel.copy()
    print(f"arrived at pike: trunk z {float(d.xpos[env.trunk_body_id][2]):.3f}  "
          f"span {env._foot_span():.3f}  upright "
          f"{float(-env._projected_gravity()[2]):+.2f}")

    qa = np.array([d.qpos[m.joint(n).qposadr[0]] for n in W.JOINT_NAMES])
    qb = np.asarray(R.BY_NAME["fold"].joints, float)
    n_move, n_hold = int(a.move_s / 0.02), int(a.hold_s / 0.02)

    def targets(res):
        """Knot residuals -> a per-step joint target path from here to fold."""
        knots = qa + (qb - qa) * np.linspace(0, 1, a.knots + 2)[:, None]
        knots[1:-1] += res.reshape(a.knots, nj)
        t = np.linspace(0, 1, n_move)
        kt = np.linspace(0, 1, a.knots + 2)
        return np.stack([np.interp(t, kt, knots[:, j]) for j in range(nj)], 1)

    def roll(res, record=False):
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_forward(m, d)
        env._obs_stack = []; env._obs_live = False
        path = targets(res)
        traj, span = [], 0.0
        for k in range(n_move + n_hold):
            q = path[min(k, n_move - 1)]
            act = np.clip((q - default) / scale, -CLIP, CLIP).astype(np.float32)
            env.step(act)
            span = max(span, float(env._foot_span()))
            if record:
                traj.append(d.qpos.copy())
        # READ THE POSE THAT SURVIVES, after the hold.
        up = float(-env._projected_gravity()[2])
        z = float(d.xpos[env.trunk_body_id][2])
        feet = sum(1 for p in R.floor_parts(env) if "foot" in str(p))
        other = sum(1 for p in R.floor_parts(env) if "foot" not in str(p))
        return up, z, span, feet, other, traj

    def score(res):
        """Fold is a POSE ON THE FEET, and the first version of this did not
        say so. It paid 100*uprightness + 60*trunk and charged 25 per non-foot
        contact, and the optimiser found that LYING ON ITS LEFT SIDE scores
        well: span 0.023 m (both feet together, so no width penalty) and
        uprightness +0.16 (better than pike's -0.52). It rolled over and
        collected. Trunk height collapsed 0.316 -> 0.158 and the score went UP.

        So the support is a GATE, not a tariff: anything but feet on the floor
        is disqualifying, and trunk height now dominates, because that is the
        quantity the hack destroyed while every other term improved.
        """
        up, z, span, feet, other, _ = roll(res)
        if other > 0 or feet < 2:
            # Not a fold. Ranked by how close it got, so the search still has
            # a gradient out of the disqualified region instead of a plateau.
            return -1000.0 + 20.0 * up + 100.0 * z, (up, z, span, feet, other)
        s = 300.0 * z + 100.0 * up                # HOLD HEIGHT, then rotate
        s -= 200.0 * max(0.0, span - a.span_max)
        return s, (up, z, span, feet, other)

    rng = np.random.default_rng(a.seed)
    mu = np.zeros(a.knots * nj)
    sig = np.full(a.knots * nj, a.sigma)
    base = score(mu)
    print(f"seed (straight line): score {base[0]:7.1f}  upright {base[1][0]:+.2f} "
          f" trunk {base[1][1]:.3f}  span {base[1][2]:.3f}  feet {base[1][3]} "
          f"other {base[1][4]}\n")
    best = (base[0], mu.copy(), base[1])
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
        # Prints the RUNNING best, which is what gets saved and replayed. It
        # used to print THIS iteration's top while saving the running one, so
        # the final replay disagreed with the last line for two whole runs —
        # two different trajectories with near-equal scores, which read as
        # non-determinism and cost a round of debugging to rule out.
        print(f"  iter {it:2d}  this {top[0]:7.1f} | best {best[0]:7.1f}  "
              f"upright {best[2][0]:+.2f}  trunk {best[2][1]:.3f}  "
              f"span {best[2][2]:.3f}  feet {best[2][3]}  other {best[2][4]}",
              flush=True)

    up, z, span, feet, other, traj = roll(best[1], record=True)
    print(f"\nBEST: upright {up:+.2f} (fold is +0.83), trunk {z:.3f} "
          f"(fold 0.511), peak span {span:.3f} m, {feet} feet / {other} other "
          f"parts down AFTER a {a.hold_s:g} s hold")
    out = "/private/tmp/claude-501/-Users-bone-dev-robotics-microduck/" \
          "76c4690c-a79a-47c2-9c55-a5bafae2f822/scratchpad/pike_fold_best.npz"
    np.savez(out, residual=best[1], qpos=np.asarray(traj),
             knots=a.knots, move_s=a.move_s)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
