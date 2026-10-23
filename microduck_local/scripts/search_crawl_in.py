"""A CRAWL-IN: arms carry the load while the feet step in, one at a time.

Every search so far optimised a joint trajectory with whatever contact set
physics happened to produce. None of them could decide to BREAK contact and
re-place a foot, which is the one thing a contact-implicit planner buys — and
the one thing the geometry calls for. The CoM sits ~a leg length ahead of the
feet, so the support cannot be stepped under the mass in a SINGLE move; but
with the ARMS taking the weight, each foot can be unloaded and brought in a
little at a time.

That is how a person gets up from their front without doing the splits: hands
and feet down, walk the feet in toward the hands, then push up. At the end of
the narrow servo route this robot is already in that posture — forearms on the
floor, span 0.215 m.

So the contact SCHEDULE is structured here rather than discovered (alternating
single-leg swings, arms supporting throughout) and CEM tunes only what each
swing does. Structure is what the unstructured searches lacked: a free 23-dim
trajectory has to invent the alternation before it can profit from it, and
1,600 samples never did.

Scored on the pose that survives a hold, with the span measured throughout —
stepping the feet IN should keep them together, where sprawling does not.
"""
from __future__ import annotations

import argparse
import numpy as np

CLIP = 4.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--elites", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4, help="alternating leg swings")
    ap.add_argument("--swing-s", type=float, default=0.30)
    ap.add_argument("--rise-s", type=float, default=0.80)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--sigma", type=float, default=0.30)
    ap.add_argument("--close-gap", action="store_true",
                    help="optimise the CoM-to-foot GAP (and narrowness) "
                         "instead of trunk height — the test that separates "
                         "a foot-placement bound from a trunk-rotation one")
    ap.add_argument("--span-max", type=float, default=0.30)
    ap.add_argument("--min-trunk", type=float, default=0.18,
                    help="--close-gap: the trunk must stay off the floor, or "
                         "the cheapest way to shorten the gap is to flatten")
    ap.add_argument("--hands", action="store_true",
                    help="also walk the HANDS back, contralateral to each leg "
                         "swing — the other end of the CoM-to-foot gap")
    ap.add_argument("--save", help="write the best params to this .npz so the "
                                   "candidate can be replayed and rendered")
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
    JN = list(W.JOINT_NAMES)
    names = [w.name for w in R.WAYPOINTS]
    route = names[:names.index("fold") + 1]

    def leg_idx(side):
        return {k: JN.index(f"{side}_{k}_joint")
                for k in ("hip_pitch", "hip_roll", "knee",
                          "ankle_pitch", "ankle_roll")}
    L, Rg = leg_idx("left"), leg_idx("right")
    FB = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]

    def arm_idx(side):
        return {k: JN.index(f"{side}_{k}_joint")
                for k in ("shoulder_pitch", "shoulder_roll", "elbow")}
    LA, RA = arm_idx("left"), arm_idx("right")

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
    q_start = np.asarray(d.qpos[[int(m.joint(n).qposadr[0]) for n in JN]], float)
    parts0 = sorted(str(p) for p in R.floor_parts(env))
    print(f"crawl starts at trunk {float(d.xpos[env.trunk_body_id][2]):.3f} "
          f"upright {float(-env._projected_gravity()[2]):+.2f} "
          f"span {env._foot_span():.3f}\n  on floor: {', '.join(parts0)[:60]}\n")

    # 4 numbers per swing (lift, draw-in, plant, roll) + 5 for the final rise.
    # --hands adds 3 more: the CONTRALATERAL hand lifts, walks BACK and plants
    # on the same beat. The gap the leg swings cannot close has two ends, and
    # only one of them was ever searched: with the hands planted far forward,
    # the trunk leans over them and the CoM stays forward no matter where the
    # feet go. A person getting up off their front walks the hands back under
    # the shoulders. Diagonal pairing (left leg with right arm) is used because
    # the ipsilateral pair unloads one whole side at once.
    PER = 7 if a.hands else 4
    dim = a.steps * PER + 5

    def roll(p, record=False):
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_forward(m, d)
        env._obs_stack = []
        env._obs_live = False
        q = q_start.copy()
        span = float(env._foot_span())
        n_sw = max(1, int(a.swing_s / 0.02))
        for i in range(a.steps):
            side = L if i % 2 == 0 else Rg
            blk = p[i * PER:(i + 1) * PER]
            lift, draw, plant, roll_ = blk[:4]
            tgt = q.copy()
            tgt[side["hip_pitch"]] += draw          # bring the thigh forward
            tgt[side["knee"]] += lift               # unload and bend
            tgt[side["ankle_pitch"]] += plant       # set the foot down
            tgt[side["hip_roll"]] += roll_ * 0.2    # small, keeps it narrow
            if a.hands:
                arm = RA if i % 2 == 0 else LA      # contralateral
                h_lift, h_draw, h_plant = blk[4:7]
                tgt[arm["elbow"]] += h_lift
                tgt[arm["shoulder_pitch"]] += h_draw
                tgt[arm["shoulder_roll"]] += h_plant * 0.2
            src = q.copy()
            for k in range(n_sw):
                f = (k + 1) / n_sw
                env.step(np.clip(((src + (tgt - src) * f) - default) / scale,
                                 -CLIP, CLIP).astype(np.float32))
                span = max(span, float(env._foot_span()))
            q = tgt
        # the rise: both legs extend, arms push off
        rise = p[a.steps * PER:]
        tgt = q.copy()
        for side in (L, Rg):
            tgt[side["hip_pitch"]] += rise[0]
            tgt[side["knee"]] += rise[1]
            tgt[side["ankle_pitch"]] += rise[2]
        for nm, j in (("left_shoulder_pitch_joint", 3),
                      ("right_shoulder_pitch_joint", 4)):
            tgt[JN.index(nm)] += rise[j]
        n_r = max(1, int(a.rise_s / 0.02))
        src = q.copy()
        for k in range(n_r + int(a.hold_s / 0.02)):
            f = min(1.0, (k + 1) / n_r)
            env.step(np.clip(((src + (tgt - src) * f) - default) / scale,
                             -CLIP, CLIP).astype(np.float32))
            span = max(span, float(env._foot_span()))
        parts = [str(x) for x in R.floor_parts(env)]
        com = float(np.sum(m.body_mass[:, None] * d.xipos, axis=0)[0]
                    / m.body_mass.sum())
        gap = com - float(np.mean([d.xpos[b][0] for b in FB]))
        return (float(-env._projected_gravity()[2]),
                float(d.xpos[env.trunk_body_id][2]), span,
                sum(1 for x in parts if "foot" in x),
                sum(1 for x in parts if "foot" not in x), gap)

    def score(p):
        up, z, span, feet, other, gap = roll(p)
        if a.close_gap:
            # THE DECISIVE OBJECTIVE. Scoring trunk height asks the crawl to
            # rise, which it cannot, so every run converges on the same stuck
            # pose and tells us nothing about WHY. Scoring the CoM-to-foot gap
            # instead asks only "can the feet get under the mass while the
            # stance stays narrow" — and that separates the two candidate
            # bounds. If the gap reaches ~0 and the rise is still refused, the
            # blocker is the TRUNK ROTATION (112 deg needed, 59 available, no
            # back pitch joint) and not foot placement.
            # The gate is NOT optional. Without it the optimiser lies flat —
            # 0 feet, 5 other parts, trunk 0.146 — because flattening shortens
            # the HORIZONTAL gap just as well as stepping in does, and costs
            # nothing. Scored that way the gap read +0.342 m and meant nothing.
            # A crawl posture is: both feet on the floor and the trunk still
            # carried off it.
            if feet < 2 or z < a.min_trunk:
                return (-1000.0 + 200.0 * z + 100.0 * feet,
                        (up, z, span, feet, other, gap))
            s = -400.0 * abs(gap)
            s -= 600.0 * max(0.0, span - a.span_max)
            s += 40.0 * z
            return s, (up, z, span, feet, other, gap)
        if other > 0 or feet < 2:
            return (-1000.0 + 120.0 * z + 25.0 * up,
                    (up, z, span, feet, other, gap))
        return (450.0 * z + 180.0 * up - 300.0 * max(0.0, span - 0.35),
                (up, z, span, feet, other, gap))

    rng = np.random.default_rng(a.seed)
    mu, sig = np.zeros(dim), np.full(dim, a.sigma)
    s0 = score(mu)
    best = (s0[0], mu.copy(), s0[1])
    # The baseline is the ROUTE'S OWN ENDPOINT with every swing at zero, so
    # every number below is read against what the servos already deliver.
    print(f"  START (zero swings): upright {s0[1][0]:+.2f}, trunk "
          f"{s0[1][1]:.3f}, span {s0[1][2]:.3f} m, {s0[1][3]} feet / "
          f"{s0[1][4]} other down, CoM gap {s0[1][5]:+.3f} m")
    print(f"  {'iter':>5s}{'best':>9s}{'upright':>9s}{'trunk':>8s}"
          f"{'span':>7s}{'feet':>6s}{'other':>7s}{'CoM gap':>9s}")
    for it in range(a.iters):
        pop = rng.normal(mu, sig, (a.pop, dim))
        pop[0] = mu
        rows = [score(x) for x in pop]
        order = np.argsort([-r[0] for r in rows])[:a.elites]
        mu = pop[order].mean(0)
        sig = pop[order].std(0) + 1e-3
        top = rows[order[0]]
        if top[0] > best[0]:
            best = (top[0], pop[order[0]].copy(), top[1])
        print(f"  {it:5d}{best[0]:9.1f}{best[2][0]:+9.2f}{best[2][1]:8.3f}"
              f"{best[2][2]:7.3f}{best[2][3]:6d}{best[2][4]:7d}"
              f"{best[2][5]:+9.3f}", flush=True)
    up, z, span, feet, other, gap = roll(best[1])
    print(f"\nBEST: upright {up:+.2f}, trunk {z:.3f}, peak span {span:.3f} m, "
          f"{feet} feet / {other} other down, CoM gap {gap:+.3f} m "
          f"after a {a.hold_s:g} s hold")
    print("  a STAND is trunk ~0.64, upright +1.00, feet only")
    if a.save:
        np.savez(a.save, params=best[1], steps=a.steps, swing_s=a.swing_s,
                 rise_s=a.rise_s, hold_s=a.hold_s)
        print(f"  params -> {a.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
