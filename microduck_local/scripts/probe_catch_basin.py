"""How fast can a rise ARRIVE at `fold` and still be caught?

There is no static rung between `pike` (pelvis 0.300 m) and `fold` (0.511 m) —
searched directly with the hold inside the objective, the best candidate at
0.316 m collapses to 0.106 m in 1.5 s even at a narrow 0.185 m span. So that
move cannot be laddered; it has to be one DYNAMIC transition that arrives at
`fold` slow enough to be caught by the controller that already owns `fold`
(`9702ba`: 43/48 holds, 37/48 narrow).

That makes the arrival SPEED the spec, and nobody has measured it. The
precedent says it matters and that intuition gets it backwards: the trunk
arrives at the top of a rise at 0.89 m/s while the stance being trained
absorbed 0.22 m/s, and the walk-policy handoff worked better handed off
EARLIER (0.60 m/s) because an active controller has a wider basin than the
passive one the 0.22 was measured on.

So: place `fold`, inject an arrival velocity, and run the real policy. The
velocity is COUPLED the way a pike->fold rise actually arrives — the pelvis
rising and the trunk still rotating up — because injecting vertical speed with
no pitch rate is not an arrival, it is a hop. Over that move the pelvis gains
0.211 m while the trunk rotates roughly a radian, so the pitch rate is scaled
~4.7 rad per m/s of lift; --coupling overrides it, and 0 isolates pure lift.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

#: Kwargs the PLAIN `getup` env cannot take. Only applied when the run is
#: actually a plain-`getup` run — see `_build_env`.
DROP = ("task", "clip_file", "track_tol", "ref_speed", "free", "start_phase",
        "handoff_pay", "force_lag", "getup_label")


def _build_env(S, run_kw, pose):
    """Place `pose` in the env the policy ACTUALLY TRAINED IN.

    This used to force `env_class("getup")` and name the pose with
    `getup_label` for every run, whatever its recorded task. For a
    `getup_ref` run that is a different robot: measured on
    `teach-asimov_getup_ref-c2bae8-s1` (2026-10-19), the same name `fold`
    gave trunk 0.5197 m at a **0.215 m** foot span in the ref env and
    0.5465 m at **0.461 m** in the plain one -- 0.73 rad of max qpos
    difference, 210 of 390 observation slots apart, and already outside the
    0.40 m narrow gate before the policy acts. The run scores 48/48 from
    `fold` under its own flags and 6/24 here, so every basin number this
    probe has ever printed for a `getup_ref` run describes a sprawled pose
    the policy never trained from -- including the "0.2 m/s" that AGENTS.md
    and `docs/roadmap.md` quote as the arrival spec.
    """
    task = str(run_kw.get("task") or "getup_ref")
    kw = dict(run_kw)
    if task == "getup_ref":
        kw.pop("task", None)
        kw["start_phase"] = pose          # the ref env's own, settled waypoint
    else:
        for k in DROP:
            kw.pop(k, None)
        kw["getup_label"] = pose
    kw.update(fell_cut=False, max_episode_s=8.0)
    return S.get("asimov1").env_class(task)(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw), task


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--pose", default="fold")
    ap.add_argument("--gate", type=float, default=0.40)
    ap.add_argument("--coupling", type=float, default=4.7,
                    help="pitch rate (rad/s) per m/s of pelvis lift")
    ap.add_argument("--speeds", default="0,0.2,0.4,0.6,0.8,1.0")
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    env, _task = _build_env(S, dict(replay_kwargs(Path("runs") / a.run)), a.pose)
    pol = GA.load(a.run)
    speeds = [float(x) for x in a.speeds.split(",")]

    print(f"{a.run}: caught from {a.pose!r} with an arrival velocity, "
          f"{a.episodes} eps, coupling {a.coupling:g} rad/s per m/s")
    env.reset(seed=0)
    print(f"  env {_task}: placed trunk z "
          f"{float(env.data.xpos[env.trunk_body_id][2]):.3f} m at a "
          f"{float(env._foot_span()):.3f} m foot span")
    print(f"  {'lift m/s':>9s}{'pitch r/s':>10s}{'holds':>8s}{'narrow':>8s}"
          f"{'med span':>10s}{'verdict':>22s}")
    base = None
    for v in speeds:
        w = v * a.coupling
        holds, narrow, spans = [], [], []
        for s in range(a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            # THE ARRIVAL. Injected after reset so the pose is the cache's own
            # `fold`; only the root twist changes, which is what differs
            # between "placed at fold" and "arriving at fold".
            env.data.qvel[2] += v
            env.data.qvel[4] += w
            o = env._get_obs()
            mx, up = 0.0, False
            for _k in range(int(env.max_steps)):
                o, _r, t, tr, _i = env.step(pol(o))
                mx = max(mx, float(env._foot_span()))
                up = R.is_standing(env)
                if t or tr:
                    break
            holds.append(up)
            narrow.append(bool(up and mx <= a.gate))
            spans.append(mx)
        h, n = sum(holds), sum(narrow)
        if base is None:
            base = h
        frac = h / max(1, base)
        verdict = ("inside the basin" if frac >= 0.8 else
                   "degraded" if frac >= 0.4 else "OUTSIDE the basin")
        print(f"  {v:9.2f}{w:10.2f}{h:5d}/{a.episodes}{n:5d}/{a.episodes}"
              f"{np.median(spans):10.3f}{verdict:>22s}", flush=True)
    print(f"\n  the spec a pike->fold launch must hit is the largest lift "
          f"still 'inside the basin'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
