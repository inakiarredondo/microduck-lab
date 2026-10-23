"""Can `9702ba` CATCH the state the scaffold's crossing arrives in?

Two full assist ladders taught nothing, and the reason is measured: pelvis 0.30
to 0.511 m is an unholdable band — no feet-under-hips pose exists below 0.30 m
even with the span unconstrained, and the best at 0.316 m collapses to 0.106 m
— so no fadeable scaffold exists inside it and there is no intermediate success
to amplify.

So stop fading. Use the scaffold as a STATE GENERATOR only: it never appears at
training time. Zero action under it reaches a supported, narrow, upright pose at
pelvis 0.48-0.55 m arriving as slow as 0.00 m/s, and `9702ba` stands 43/48 from
`fold` and catches an arrival up to ~0.2 m/s. If it also catches the scaffold's
crossing state, the composition is already a floor-to-stand get-up, and the
thing left to TRAIN is "reach that state without the scaffold" — one objective,
no ladder.

The check AGENTS.md demands before any start distribution is trusted:
**reachable is not recoverable.** A state the robot demonstrably passes through
justifies only that it can be REACHED. So each captured state is replayed in a
SCAFFOLD-FREE env — the deployment one, `9702ba`'s own recorded kwargs — and
zero action is scored on the same states as the floor.

Velocity is carried explicitly rather than through the pose cache, which stores
`qpos` only: the arrival SPEED is the quantity under test, and a cache round
trip would silently zero it.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

DROP = ("task", "clip_file", "track_tol", "ref_speed", "free", "start_phase",
        "handoff_pay", "force_lag", "getup_label", "prologue",
        "getup_assist", "getup_attitude")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--catcher", default="teach-asimov_getup_turn-9702ba")
    ap.add_argument("--lift", type=float, default=0.80)
    ap.add_argument("--gain", type=float, default=120.0)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--window", type=int, default=5,
                    help="frames either side of the closest frame to also try, "
                         "since the best handoff INSTANT is what is being "
                         "looked for and 128 handoffs on this robot said "
                         "EARLIER beats calmer")
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    # 1. THE GENERATOR: zero action under the scaffold, states captured.
    gen = S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0,
        start_phase="prone", prologue="pushup", getup_assist=a.lift,
        getup_attitude=a.gain, history=5, actuator_force="dc",
        action_clip=4.0, fell_cut=False, max_episode_s=8.0)
    gm, gd = gen.model, gen.data
    gp = gm.body("pelvis_link").id
    zero = np.zeros(gen.action_space.shape, np.float32)
    caught = []
    for e in range(a.episodes):
        gen.reset(seed=e)
        traj = []
        for _k in range(int(gen.max_steps)):
            gen.step(zero)
            z = float(gd.xpos[gp][2])
            u = float(-gen._projected_gravity()[2])
            parts = [str(x) for x in R.floor_parts(gen)]
            feet = sum(1 for x in parts if "foot" in x)
            other = len(parts) - feet
            sup = bool(gen._on_floor_contacts())
            d_ = abs(z - 0.511) / 0.05 + abs(u - 0.83) / 0.20
            traj.append((d_ if sup else 9e9, gd.qpos.copy(), gd.qvel.copy(),
                         z, u, float(gen._foot_span()), feet, other))
        if not traj:
            continue
        k = int(np.argmin([r[0] for r in traj]))
        lo = max(0, k - a.window)
        hi = min(len(traj), k + a.window + 1)
        caught.append([traj[i] for i in range(lo, hi) if traj[i][0] < 9e8])

    n = sum(len(c) for c in caught)
    print(f"generator: lift {a.lift:g}, attitude {a.gain:g}, zero action — "
          f"{len(caught)} episodes, {n} supported candidate states\n")
    if not n:
        print("  NO supported candidate states — the generator never reaches "
              "one, so there is nothing to hand off.")
        return 0

    # 2. THE CATCHER, in a SCAFFOLD-FREE env: its own recorded kwargs, minus
    #    every scaffold key, so nothing helps it.
    kw = dict(replay_kwargs(Path("runs") / a.catcher))
    for k in DROP:
        kw.pop(k, None)
    kw.update(getup_label="fold", fell_cut=False, max_episode_s=8.0)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    pol = GA.load(a.catcher)
    zero2 = np.zeros(env.action_space.shape, np.float32)

    def replay(q, v, policy):
        env.reset(seed=0)
        env.data.qpos[:] = q
        env.data.qvel[:] = v
        env._obs_stack, env._obs_live = [], False
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        o = env._get_obs()
        mx, up = 0.0, False
        for _k in range(int(env.max_steps)):
            act = zero2 if policy is None else pol(o)
            o, _r, t, tr, _i = env.step(act)
            mx = max(mx, float(env._foot_span()))
            up = R.is_standing(env)
            if t or tr:
                break
        return up, mx

    # THE CONTROL, and it runs FIRST. A 0/N from a hand-rolled replay is
    # exactly the shape of a broken harness: overwriting qpos/qvel and
    # rebuilding the observation ring is three chances to hand the policy a
    # different robot. So a state that MUST be catchable goes through the same
    # path -- `fold`, placed, at rest -- and if the catcher does not stand from
    # THAT, nothing below means anything.
    ctrl_w = R.BY_NAME["fold"]
    env.reset(seed=0)
    env._place(np.asarray(ctrl_w.joints, float), ctrl_w.root_pitch)
    env.data.qvel[:] = 0.0
    import mujoco as _mj
    _mj.mj_forward(env.model, env.data)
    q_ctrl, v_ctrl = env.data.qpos.copy(), env.data.qvel.copy()
    ok_ctrl = sum(1 for _ in range(8) if replay(q_ctrl, v_ctrl, pol)[0])
    print(f"  CONTROL: `fold` placed, through this same replay path — "
          f"the catcher stands {ok_ctrl}/8")
    if ok_ctrl == 0:
        print("  >> THE HARNESS IS BROKEN. Everything below is void.\n")
    else:
        print("  >> harness sound; the numbers below are about the STATES.\n")

    # WHAT kind of support? `fold` is FEET ONLY, and "any floor contact" also
    # admits a robot propped on its forearms — which is a pose the catcher has
    # never seen, at any height or attitude. So the subset is reported and
    # scored separately rather than averaged into one number.
    flat = [r for c in caught for r in c]
    feet_only = [r for r in flat if r[7] == 0 and r[6] >= 2]
    print(f"  support composition of the {len(flat)} candidates: "
          f"{len(feet_only)} FEET-ONLY, "
          f"{sum(1 for r in flat if r[7] > 0)} with something else down")
    if feet_only:
        zz = np.array([[r[3], r[4], r[5]] for r in feet_only])
        print(f"  the feet-only ones: pelvis {zz[:, 0].min():.3f}-"
              f"{zz[:, 0].max():.3f} m, upright {zz[:, 1].min():+.2f} to "
              f"{zz[:, 1].max():+.2f}, span {np.median(zz[:, 2]):.3f} m")
    print()

    rows = []
    for ci, cand in enumerate(caught):
        for (_d, q, v, z, u, sp_, _f, _o) in cand:
            okp, mxp = replay(q, v, pol)
            okz, mxz = replay(q, v, None)
            rows.append((z, u, sp_, float(abs(v[2])), okp, mxp, okz, mxz,
                         _f, _o))
    arr = rows
    fo = [r for r in arr if r[9] == 0 and r[8] >= 2]
    if fo:
        print(f"  {'(feet-only subset)':30s} stands "
              f"{sum(1 for r in fo if r[4]):3d}/{len(fo)}")
    pol_ok = sum(1 for r in arr if r[4])
    pol_nar = sum(1 for r in arr if r[4] and r[5] <= 0.40)
    z_ok = sum(1 for r in arr if r[6])
    print(f"  {a.catcher[-28:]:30s} stands {pol_ok:3d}/{len(arr)}  "
          f"NARROW {pol_nar:3d}/{len(arr)}")
    print(f"  {'zero action (the floor)':30s} stands {z_ok:3d}/{len(arr)}")
    if pol_ok:
        best = sorted((r for r in arr if r[4]), key=lambda r: r[5])[:4]
        print("\n  states it CAUGHT (pelvis, upright, span at handoff, "
              "arrival lift, peak span):")
        for r in best:
            print(f"    z {r[0]:.3f}  up {r[1]:+.2f}  span {r[2]:.3f}  "
                  f"lift {r[3]:.2f} m/s  ->  peak span {r[5]:.3f} m")
        print("\n  THE COMPOSITION IS A FLOOR-TO-STAND GET-UP. What is left to "
              "train is\n  reaching one of these states WITHOUT the scaffold — "
              "one objective, no ladder.")
    else:
        print("\n  it caught NONE of them: reachable is not recoverable, and "
              "this\n  arrival is not in the catcher's basin. An arrival "
              "objective aimed\n  here would have no anchor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
