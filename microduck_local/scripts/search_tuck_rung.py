"""Is there a NEW RUNG between `pike` (pelvis 0.300 m) and `fold` (0.511 m)?

Every curriculum here jumps straight from pike/rock_back to fold, and nothing
has ever crossed it: `9702ba` is 43/48 from `fold` and 0/48 from both
`narrow_rock_back` and `narrow_pike`, and the reference recipe at its full 12 M
steps is 0/24 from `prone`. The gap is one move wide and no reward has bought
it.

Today's reachability sweep (`probe_tuck_reach.py`) says a feet-under-the-hips
pose EXISTS at pelvis 0.303 m once the arm joints are free — which is pike's
own height. If such a pose also HOLDS, it is an intermediate rung the ladder
never had: pike -> tuck -> fold, with each leg short enough to be a transition
rather than a leap.

So the hold is the OBJECTIVE here, not a post-hoc check. The previous version
of this measurement scored poses on the x-gap alone, found eight that zeroed
it, and reported that all eight fell — but it never asked for a narrow stance
or for lateral balance, so it was scoring arbitrary gap-zeroing contortions.
A pose is scored here only on what survives 1.5 s of being commanded:

  * nothing but the feet on the floor at the END of the hold,
  * the pelvis still within 50 mm of where it started,
  * foot span <= --span-max, because a wide tuck is the splits by another name,
  * the CoM inside the support polygon in BOTH axes (`support_margin`, built
    from MuJoCo's contact points, because a pad hovering 2 mm clear is not
    support).

The hold path is validated: run through it, `stand` keeps 0.614 -> 0.608 m
upright +1.00 and `fold` 0.511 -> 0.516 m upright +0.85, both on feet alone.
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
    ap.add_argument("--target-z", type=float, default=0.33,
                    help="pelvis height to aim the rung at; pike is 0.300 and "
                         "fold 0.511, so a rung here halves the one hard move")
    ap.add_argument("--span-max", type=float, default=0.30)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--hands", action="store_true",
                    help="allow the FOREARMS down too. The unholdable band "
                         "(pelvis 0.30-0.511 m) was measured with FEET ONLY "
                         "allowed, but the robot has forearms and the whole "
                         "authored route sits inside that band on them, "
                         "holding. So the unasked question is how HIGH a pose "
                         "can be and still hold on four contacts — because "
                         "the hands then only have to come off over the "
                         "REMAINDER, not over the whole band.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="write the best pose as an .npz")
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
             ("shoulder", "elbow", "wrist"))]
    pelvis = m.body("pelvis_link").id
    FB = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
    dflt = np.asarray(env.robot.default_pose, float)
    scale = np.asarray(env.robot.action_scale, float)

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
        other = len(parts) - feet
        return (float(d.xpos[pelvis][2]), feet, other,
                float(env._foot_span()),
                float(-env._projected_gravity()[2]))

    def margin():
        try:
            return float(R.support_margin(env))
        except Exception:
            return float("nan")

    def score(x):
        put(x)
        z0, feet0, other0, span0, _ = state()
        # Reject before paying for a 1.5 s hold: the wrong height, the wrong
        # contacts or a wide stance cannot become a rung, and the hold is by
        # far the most expensive part of this loop.
        # With --hands, forearm contact is free; everything else still is not
        # (a knee or a hip down is not a stance to rise from).
        if a.hands:
            parts0 = [str(x) for x in R.floor_parts(env)]
            other0 = sum(1 for x in parts0 if "foot" not in x
                         and "forearm" not in x.lower())
        pen = (20.0 * abs(z0 - a.target_z) + 10.0 * other0
               + 10.0 * max(0, 2 - feet0)
               + 20.0 * max(0.0, span0 - a.span_max))
        if pen > 2.0:
            return -100.0 - pen, (z0, z0, feet0, other0, span0, 0.0,
                                  float("nan"))
        q_t = np.asarray([d.qpos[qs[n]] for n in JN], float)
        env._obs_stack, env._obs_live = [], False
        act = np.clip((q_t - dflt) / scale, -4.0, 4.0).astype(np.float32)
        for _ in range(int(a.hold_s / 0.02)):
            env.step(act)
        z1, feet1, other1, span1, up1 = state()
        if a.hands:
            parts1 = [str(x) for x in R.floor_parts(env)]
            other1 = sum(1 for x in parts1 if "foot" not in x
                         and "forearm" not in x.lower())
        mg = margin()
        kept = (other1 == 0 and feet1 >= 2 and z1 > z0 - 0.05)
        s = (60.0 * float(kept) + 40.0 * up1 - 30.0 * abs(z1 - a.target_z)
             - 40.0 * max(0.0, span1 - a.span_max)
             + (20.0 * np.tanh(mg * 20.0) if np.isfinite(mg) else 0.0)
             - pen)
        return s, (z0, z1, feet1, other1, span1, up1, mg)

    rng = np.random.default_rng(a.seed)
    lo = np.array([lim[n][0] for n in free] + [-np.pi / 2])
    hi = np.array([lim[n][1] for n in free] + [0.3])
    print(f"  a rung at pelvis ~{a.target_z:.2f} m, span <= {a.span_max:.2f} m,"
          f" surviving a {a.hold_s:g} s commanded hold")
    print(f"  {'iter':>5s}{'best':>9s}{'z0':>7s}{'z1':>7s}{'feet':>6s}"
          f"{'other':>7s}{'span':>7s}{'upright':>9s}{'margin':>9s}")
    best = None
    for r in range(a.restarts):
        mu = (lo + hi) / 2 + rng.normal(0, 0.15, len(lo))
        sig = (hi - lo) / 4
        for it in range(a.iters):
            pop = np.clip(rng.normal(mu, sig, (a.pop, len(lo))), lo, hi)
            rows = [score(pop[i]) for i in range(a.pop)]
            order = np.argsort([-v[0] for v in rows])[:max(4, a.pop // 10)]
            mu = pop[order].mean(0)
            sig = pop[order].std(0) + 1e-3
            if best is None or rows[order[0]][0] > best[0]:
                best = (rows[order[0]][0], pop[order[0]].copy(),
                        rows[order[0]][1])
            z0, z1, f1, o1, sp, up, mg = best[2]
            print(f"  {r}.{it:<3d}{best[0]:9.1f}{z0:7.3f}{z1:7.3f}{f1:6d}"
                  f"{o1:7d}{sp:7.3f}{up:+9.2f}{mg:+9.3f}", flush=True)
    z0, z1, f1, o1, sp, up, mg = best[2]
    # THE HEIGHT CLAUSE WAS MISSING and it mattered: a candidate at pelvis
    # 0.283 against a 0.42 target was printed as "A NEW RUNG" even though the
    # penalty had REJECTED it before the hold ever ran. A pose that holds
    # below the band is not news — the authored route already sits there on
    # its forearms — so the verdict has to require the height it was asked
    # for, and an upright trunk, not just survival.
    ok = (o1 == 0 and f1 >= 2 and z1 > z0 - 0.05 and sp <= a.span_max
          and abs(z0 - a.target_z) <= 0.03 and up1 > 0.2)
    print(f"\n  BEST: pelvis {z0:.3f} -> {z1:.3f} m after {a.hold_s:g} s, "
          f"{f1} feet / {o1} other down, span {sp:.3f} m, upright {up:+.2f}, "
          f"support margin {mg:+.3f} m")
    print(f"  {'A NEW RUNG' if ok else 'NOT a rung — it did not survive'}")
    print("  pike is pelvis 0.300 m, fold 0.511 m")
    if a.save and ok:
        np.savez(a.save, params=best[1], joints=np.asarray(
            [d.qpos[qs[n]] for n in JN], float))
        print(f"  pose -> {a.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
