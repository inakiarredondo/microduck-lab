"""How BIG is the set of states a narrow stand works from?

Every attempt so far asked for a trajectory INTO `fold`, and all of them
failed. This asks the dual question: what is the TARGET? Perturb `fold` by
increasing amounts, run the policy, and record which perturbed states still
reach a stand and how wide they end up.

If the basin is large, the pike->fold problem is over-specified -- we do not
need to hit `fold`, we need to land anywhere in a set -- and the search can be
re-aimed at the nearest point of that set instead of at one pose. If it is
tiny, that is the measurement that says this route is genuinely closed.

Settled by bisection onto first contact, because a placed pose FLOATS (and a
floating pose reports no support polygon at all, which cost three versions of
an earlier probe).
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="asimov-getup-h5pd2")
    ap.add_argument("--rung", default="fold")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args(argv)

    import importlib.util
    import sys

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S
    sp = importlib.util.spec_from_file_location(
        "_ga", "scripts/getup_acceptance.py")
    GA = importlib.util.module_from_spec(sp); sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)
    pol = GA.load(args.run)

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0,
        getup_label="prone", actuator_force="dc", history=5,
        action_clip=13.0, fell_cut=False, max_episode_s=args.seconds)
    m, d = env.model, env.data
    ADR = np.array([int(m.joint(n).qposadr[0]) for n in W.JOINT_NAMES])
    base = np.asarray(R.BY_NAME[args.rung].joints, float)
    pitch0 = R.BY_NAME[args.rung].root_pitch

    def settle(q, pitch):
        d.qpos[:] = 0.0; d.qvel[:] = 0.0
        h = 0.5 * pitch
        d.qpos[3:7] = [np.cos(h), 0.0, np.sin(h), 0.0]
        d.qpos[ADR] = q
        lo, hi = -0.3, 1.2
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            d.qpos[2] = mid
            mujoco.mj_forward(m, d)
            if d.ncon > 0:
                lo = mid
            else:
                hi = mid
        d.qpos[2] = lo
        mujoco.mj_forward(m, d)

    print(f"{args.run} from perturbed {args.rung!r}, dc, "
          f"{args.samples} samples per level\n")
    print(f"  {'joint sd':>9s}{'pitch sd':>10s}{'stands':>9s}"
          f"{'peak span':>11s}{'narrow(<=0.40)':>16s}")
    for jsd, psd in ((0.00, 0.00), (0.10, 0.05), (0.20, 0.10),
                     (0.35, 0.20), (0.50, 0.30), (0.70, 0.45)):
        rng = np.random.default_rng(0)
        stood, spans, narrow = 0, [], 0
        for i in range(args.samples):
            q = base + rng.normal(0, jsd, len(base))
            q = np.clip(q, m.jnt_range[[m.joint(n).id for n in W.JOINT_NAMES], 0],
                        m.jnt_range[[m.joint(n).id for n in W.JOINT_NAMES], 1])
            settle(q, pitch0 + rng.normal(0, psd))
            env._obs_stack = []; env._obs_live = False
            o = env._get_obs()
            if hasattr(pol, "reset"):
                pol.reset()
            mx, up = 0.0, False
            for k in range(int(args.seconds / 0.02)):
                o, _r, t, tr, _i = env.step(pol(o))
                mx = max(mx, float(env._foot_span()))
                up = R.is_standing(env)
                if t or tr:
                    break
            stood += int(up); spans.append(mx)
            narrow += int(up and mx <= 0.40)
        med = float(np.median(spans))
        print(f"  {jsd:9.2f}{psd:10.2f}{stood:6d}/{args.samples}"
              f"{med:11.3f}{narrow:12d}/{args.samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
