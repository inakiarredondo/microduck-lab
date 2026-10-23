"""At what pelvis HEIGHT does a feet-under-the-hips pose first exist?

The crawl searches moved the pelvis-to-foot gap from 486 mm to about 470 mm and
stopped, and the obvious reading is "the stepping is bad". But a search can only
reach a configuration that EXISTS, so the prior question is kinematic.

Posed correctly it is a sweep over pelvis HEIGHT, not over trunk pitch: asked
with the height free, "minimise the gap" is trivially solved by hanging the
legs straight down at standing height, which is a stand and not a tuck. What
the get-up needs is the gap small while the pelvis is still LOW — a crouch to
rise out of. If no such pose exists, the feet cannot come in before the trunk
comes up, and every crawl was being asked for a pose outside the configuration
space. That is a different claim from "the policy cannot find it".

Pure kinematics: joint limits, the root settled onto the floor by MuJoCo's own
contact distances (`settle_onto_floor` — a geom_size radius read off a BOX left
the constraint 23 mm unsatisfied at every pitch and settled nothing), and only
the FEET allowed to touch. No dynamics, nothing asked to hold, so this is the
configuration space itself.
"""
from __future__ import annotations

import argparse
import numpy as np

LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--allow", default="",
                    help="non-foot parts that may also touch, e.g. 'forearm'. "
                         "The feet-only sweep says a tuck needs pelvis 0.40 m, "
                         "but the crawl HAS its hands down — arms carrying the "
                         "trunk is a different configuration space, and it is "
                         "the one the route is actually in.")
    ap.add_argument("--arms", action="store_true",
                    help="free the arm joints too (implied by --allow)")
    ap.add_argument("--hold-s", type=float, default=1.5,
                    help="after finding the narrowest tuck at each height, "
                         "COMMAND it and see whether it survives. A pose that "
                         "exists kinematically and collapses is not a rung — "
                         "this repo has twice built a curriculum on poses that "
                         "passed a 0.05 s settle and died in 0.5 s.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=a.seed,
        getup_label="prone", actuator_force="dc", fell_cut=False)
    m, d = env.model, env.data
    JN = list(W.JOINT_NAMES)
    qs = {n: int(m.joint(n).qposadr[0]) for n in JN}
    lim = {n: m.jnt_range[m.joint(n).id].copy() for n in JN}
    legs = [f"{s}_{k}_joint" for s in ("left", "right") for k in LEG
            if f"{s}_{k}_joint" in qs]
    allow = tuple(x.strip() for x in a.allow.split(",") if x.strip())
    if allow or a.arms:
        legs += [n for n in JN if any(k in n for k in
                 ("shoulder", "elbow", "wrist"))]
    pelvis = m.body("pelvis_link").id
    FB = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]

    def put(x):
        d.qpos[:] = 0.0
        d.qpos[2] = 0.70
        pitch = float(x[-1])
        c, s = np.cos(pitch / 2), np.sin(pitch / 2)
        d.qpos[3:7] = [c, 0.0, s, 0.0]
        for n, v in zip(legs, x[:-1]):
            lo, hi = lim[n]
            d.qpos[qs[n]] = float(np.clip(v, lo, hi))
        d.qvel[:] = 0.0
        # A NEGATIVE clearance on purpose: `settle_onto_floor` leaves the robot
        # 1 mm clear, MuJoCo then reports ncon == 0, and `floor_parts` sees an
        # empty contact list — which read as "nothing is touching" and silently
        # disabled the only-the-feet constraint in the first version of this
        # probe. Pressed 1 mm in, the contacts are real.
        R.settle_onto_floor(m, d, clearance=-0.001)
        mujoco.mj_forward(m, d)

    def measure(x):
        put(x)
        parts = [str(v) for v in R.floor_parts(env)]
        feet = sum(1 for v in parts if "foot" in v)
        other = sum(1 for v in parts if "foot" not in v
                    and not any(k in v.lower() for k in allow))
        gap = float(d.xpos[pelvis][0]) - float(
            np.mean([d.xpos[b][0] for b in FB]))
        return gap, feet, other, float(d.xpos[pelvis][2]), len(parts)

    rng = np.random.default_rng(a.seed)
    lo = np.array([lim[n][0] for n in legs] + [-np.pi / 2])
    hi = np.array([lim[n][1] for n in legs] + [0.3])
    print(f"  touching allowed: feet{''.join(' + ' + x for x in allow)}"
          f"   gap = pelvis_x - mean(foot_x)")
    print(f"  {'target z':>9s}{'pelvis z':>10s}{'min |gap|':>11s}"
          f"{'feet':>6s}{'other':>7s}{'nparts':>8s}{'verdict':>22s}"
          f"   | commanded hold")
    for target in (0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64):
        best = None
        for r in range(a.restarts):
            mu = (lo + hi) / 2 + rng.normal(0, 0.15, len(lo))
            sig = (hi - lo) / 4
            for _ in range(a.iters):
                pop = np.clip(rng.normal(mu, sig, (a.pop, len(lo))), lo, hi)
                rows = [measure(pop[i]) for i in range(a.pop)]
                # Feasibility dominates: anything but the two feet, or a pelvis
                # off the requested height, is ruled out before the gap counts.
                key = [abs(v[0]) + 10.0 * v[2] + 10.0 * max(0, 2 - v[1])
                       + 20.0 * abs(v[3] - target) for v in rows]
                order = np.argsort(key)[:max(4, a.pop // 12)]
                mu = pop[order].mean(0)
                sig = pop[order].std(0) + 1e-3
                if best is None or key[order[0]] < best[0]:
                    best = (key[order[0]], rows[order[0]],
                            pop[order[0]].copy())
        # THE HOLD. Re-place the winning pose and command it for --hold-s.
        held = ()
        if a.hold_s > 0:
            x = best[2]
            put(x)
            q_t = np.asarray([d.qpos[qs[n]] for n in JN], float)
            env._obs_stack, env._obs_live = [], False
            act = np.clip((q_t - np.asarray(env.robot.default_pose, float))
                          / np.asarray(env.robot.action_scale, float),
                          -4.0, 4.0).astype(np.float32)
            for _ in range(int(a.hold_s / 0.02)):
                env.step(act)
            hp = [str(v) for v in R.floor_parts(env)]
            held = (float(d.xpos[pelvis][2]),
                    float(d.xpos[pelvis][0]) - float(
                        np.mean([d.xpos[b][0] for b in FB])),
                    sum(1 for v in hp if "foot" in v),
                    sum(1 for v in hp if "foot" not in v))
        gap, feet, other, pz, npart = best[1]
        at_h = abs(pz - target) < 0.03
        ok = ("TUCK EXISTS" if other == 0 and feet >= 2 and abs(gap) < 0.12
              and at_h else
              "not on the feet" if other or feet < 2 else
              "height unreachable" if not at_h else "feet stay OFFSET")
        line = (f"  {target:9.2f}{pz:10.3f}{abs(gap):11.3f}{feet:6d}"
                f"{other:7d}{npart:8d}{ok:>22s}")
        if held:
            hz, hg, hf, ho = held
            kept = "HELD" if ho == 0 and hf >= 2 and hz > pz - 0.05 else "fell"
            line += (f"   | after {a.hold_s:g} s: z {hz:.3f} gap {abs(hg):.3f} "
                     f"{hf}f/{ho}o  {kept}")
        print(line)
    print("\n  fold: pelvis 0.206 m, gap +0.486 m, forearms+feet down")
    print("  stand keyframe: pelvis 0.65 m, gap +0.094 m, feet only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
