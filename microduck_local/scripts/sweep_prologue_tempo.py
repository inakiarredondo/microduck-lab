"""Can a SERVO route reach `fold` dynamically? Sweep its tempo.

`fold` is the only authored waypoint that is a stable configuration (pike,
rock_back and quadruped are 0 of 480 after a 1 s settle), so a narrow get-up
cannot be a ladder of static rungs — the floor-to-fold half has to be a
DYNAMIC route that carries momentum through poses the robot cannot rest in.

The authored route is exactly such a route and it is narrow: played open loop
it holds **0.215 m** of foot span the whole way. It has only ever been played
at one tempo, and this robot's reference is known to HAVE a tempo (it stands
at 0.25 s a leg and falls at 0.5). So sweep the leg duration, and also the
dwell between legs, and ask where the motion ARRIVES — judged on the pose that
survives a hold, not on a peak, which is what hid the pike problem for seven
attempts.

Reports arrival against `fold` (trunk 0.541, upright +0.84, span 0.230) and
then whether the narrow policy can take over from there.
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="teach-asimov_getup_turn-9a3c81")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--hold-s", type=float, default=1.0)
    ap.add_argument("--through", default="fold",
                    help="last authored waypoint the SERVOS drive to")
    a = ap.parse_args(argv)

    import importlib.util
    import sys

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S
    sp = importlib.util.spec_from_file_location(
        "_ga", "scripts/getup_acceptance.py")
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)
    pol = GA.load(a.run)

    names = [w.name for w in R.WAYPOINTS]
    route = names[:names.index(a.through) + 1]
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0,
        getup_label="prone", actuator_force="dc", history=5,
        action_clip=4.0, fell_cut=False, max_episode_s=20.0)
    m, d = env.model, env.data
    default = np.asarray(env.robot.default_pose, float)
    scale = np.asarray(env.robot.action_scale, float)

    def play(leg_s, dwell_s, seed):
        env.reset(seed=seed)
        w0 = R.BY_NAME[route[0]]
        env._place(np.asarray(w0.joints, float), w0.root_pitch)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        env._obs_stack = []
        env._obs_live = False
        cur = np.asarray(w0.joints, float)
        span = 0.0
        for nm in route[1:]:
            tgt = np.asarray(R.BY_NAME[nm].joints, float)
            n_leg = max(1, int(leg_s / 0.02))
            for k in range(n_leg + int(dwell_s / 0.02)):
                f = min(1.0, (k + 1) / n_leg)
                q = cur + (tgt - cur) * f
                env.step(np.clip((q - default) / scale, -4.0, 4.0
                                 ).astype(np.float32))
                span = max(span, float(env._foot_span()))
            cur = tgt
        # HOLD the last target: an arrival that cannot be held is not one.
        for _ in range(int(a.hold_s / 0.02)):
            env.step(np.clip((cur - default) / scale, -4.0, 4.0
                             ).astype(np.float32))
            span = max(span, float(env._foot_span()))
        return (float(d.xpos[env.trunk_body_id][2]),
                float(-env._projected_gravity()[2]), span)

    print(f"servo route {route[0]} -> {a.through}, dc, held {a.hold_s:g} s, "
          f"{a.episodes} seeds\n")
    print(f"  {'leg s':>7s}{'dwell':>7s}{'trunk z':>9s}{'upright':>9s}"
          f"{'peak span':>11s}{'policy stands':>15s}")
    best = None
    for leg_s in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
        for dwell_s in (0.0, 0.12):
            rows = [play(leg_s, dwell_s, s) for s in range(a.episodes)]
            r = np.asarray(rows)
            stands = 0
            for s in range(a.episodes):
                play(leg_s, dwell_s, s)
                env._obs_stack = []
                env._obs_live = False
                o = env._get_obs()
                if hasattr(pol, "reset"):
                    pol.reset()
                up = False
                for _k in range(300):
                    o, _r, t, tr, _i = env.step(pol(o))
                    up = R.is_standing(env)
                    if t or tr:
                        break
                stands += int(up)
            z, u, sp_ = np.median(r[:, 0]), np.median(r[:, 1]), np.median(r[:, 2])
            print(f"  {leg_s:7.2f}{dwell_s:7.2f}{z:9.3f}{u:+9.2f}{sp_:11.3f}"
                  f"{stands:12d}/{a.episodes}")
            if best is None or z > best[0]:
                best = (z, leg_s, dwell_s, stands)
    print(f"\n  fold is trunk 0.541, upright +0.84, span 0.230")
    print(f"  best arrival: trunk {best[0]:.3f} at leg {best[1]:.2f}s "
          f"dwell {best[2]:.2f}s, policy stands {best[3]}/{a.episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
