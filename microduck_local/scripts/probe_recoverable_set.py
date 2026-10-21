"""How far from `fold` can `9702ba` still recover? The size of the target.

The crossing's spec is known (joints near `fold`, root at height and upright,
lift <= 0.2 m/s, feet only) but not its TOLERANCE. If the catcher recovers from
states 100 mm below `fold`, the crossing only has to reach 0.41 m, not 0.511 —
and that changes what any trainer is being asked for.

The states are produced by letting the robot FALL from `fold` under zero action
for a growing number of steps, so each depth is a configuration the body
demonstrably passes through on the way down — the same states the crossing has
to arrive through on the way up, traversed in reverse. Recorded with velocity,
because a fallen state's momentum is most of what makes it hard.

`AGENTS.md`: reachable is not recoverable. So each depth is scored by the
POLICY and by ZERO ACTION on the same states, and the depth where the policy
stops beating the floor is the edge of the basin.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

DROP = ("task", "clip_file", "track_tol", "ref_speed", "free", "start_phase",
        "handoff_pay", "force_lag", "getup_label", "prologue",
        "getup_assist", "getup_attitude", "fold_pay", "arm_unload")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="teach-asimov_getup_turn-9702ba")
    ap.add_argument("--depths", default="0,5,10,15,20,30,40,60")
    ap.add_argument("--episodes", type=int, default=12)
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    kw = dict(replay_kwargs(Path("runs") / a.run))
    for k in DROP:
        kw.pop(k, None)
    kw.update(getup_label="fold", fell_cut=False, max_episode_s=8.0)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    m, d = env.model, env.data
    pel = m.body("pelvis_link").id
    pol = GA.load(a.run)
    zero = np.zeros(env.action_space.shape, np.float32)

    def fallen(depth, seed):
        """`fold`, then `depth` steps of zero action. Returns the state."""
        env.reset(seed=seed)
        for _ in range(depth):
            env.step(zero)
        return d.qpos.copy(), d.qvel.copy()

    def recover(q0, v0, policy):
        env.reset(seed=0)
        d.qpos[:], d.qvel[:] = q0, v0
        env._obs_stack, env._obs_live = [], False
        mujoco.mj_forward(m, d)
        o = env._get_obs()
        mx, up = 0.0, False
        for _k in range(int(env.max_steps)):
            o, _r, t, tr, _i = env.step(zero if policy is None else pol(o))
            mx = max(mx, float(env._foot_span()))
            up = R.is_standing(env)
            if t or tr:
                break
        return up, mx

    print(f"  {a.run[-28:]} recovering from `fold` + N steps of zero action, "
          f"{a.episodes} eps\n")
    print(f"  {'depth':>6s}{'s':>7s}{'pelvis':>8s}{'upright':>9s}"
          f"{'policy':>9s}{'narrow':>9s}{'zero':>8s}{'verdict':>20s}")
    for dep in [int(x) for x in a.depths.split(",")]:
        pol_ok = nar = z_ok = 0
        zs, us = [], []
        for s in range(a.episodes):
            q0, v0 = fallen(dep, s)
            zs.append(float(d.xpos[pel][2]))
            us.append(float(-env._projected_gravity()[2]))
            o1, m1 = recover(q0, v0, pol)
            o2, _ = recover(q0, v0, None)
            pol_ok += int(o1)
            nar += int(o1 and m1 <= 0.40)
            z_ok += int(o2)
        v = ("IN the basin" if pol_ok >= 0.7 * a.episodes else
             "edge" if pol_ok else "OUTSIDE")
        if pol_ok and pol_ok <= z_ok:
            v = "no better than zero"
        print(f"  {dep:6d}{dep * 0.02:7.2f}{np.median(zs):8.3f}"
              f"{np.median(us):+9.2f}{f'{pol_ok}/{a.episodes}':>9s}"
              f"{f'{nar}/{a.episodes}':>9s}{f'{z_ok}/{a.episodes}':>8s}"
              f"{v:>20s}", flush=True)
    print("\n  the deepest depth still IN the basin is how much of the band a")
    print("  crossing does NOT have to climb — the target's real size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
