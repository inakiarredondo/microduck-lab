"""Is the attitude clamp BINDING at the feet-only instants, and does raising it help?

`3ec8cc` got the hands off the floor — feet-only frames 0.0 % -> 35.3 %, arm
load 0.694 -> 0.000 — and the single unmet clause of `is_standing` is
uprightness: the best feet-only pose reaches **+0.56** where 0.85 is needed.

`ATTITUDE_MAX` is 60 N.m, and that number was chosen while the ARMS were still
carrying the trunk: constant torques at 80 and above launched the robot when
combined with a 0.55 lift. With the arms unloaded the hold has to do work it
never had to do before, so the clamp is the suspect — but "the clamp is
binding" is a measurement, not a story. This reports the fraction of feet-only
frames at saturation BEFORE sweeping the clamp, because if it is not saturating
then raising it cannot be the lever and the next idea is somewhere else.

No retraining: the clamp is a class attribute, so the same policy is rolled out
under each value. A policy that stands only because the scaffold got stronger is
not a get-up — but it says the lever is real, which is what a chain would cost
8.4 M steps to find out.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?",
                    default="teach-asimov_getup_gap-3ec8cc-s2")
    ap.add_argument("--clamps", default="60,100,150,220")
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    pol = GA.load(a.run)
    print(f"  {a.run}, {a.episodes} eps — the scaffold's knobs are the ones "
          f"it TRAINED with\n")
    print(f"  {'clamp':>7s}{'|tau| med':>11s}{'at sat':>8s}{'feet-only':>11s}"
          f"{'best up':>9s}{'z there':>9s}{'span':>7s}{'STANDS':>8s}")
    for clamp in [float(x) for x in a.clamps.split(",")]:
        cls = S.get("asimov1").env_class("getup_ref")
        cls.ATTITUDE_MAX = clamp
        kw = dict(replay_kwargs(Path("runs") / a.run))
        kw.pop("task", None)
        kw.update(fell_cut=False, max_episode_s=8.0, getup_attitude=120.0,
                  fold_pay=20.0, arm_unload=2.0, ground_cost=6.0)
        env = cls(domain_rand=False, obs_noise=True, action_delay=True,
                  seed=0, **kw)
        pel = env.model.body("pelvis_link").id
        taus, sat, fo, tot, stands = [], 0, 0, 0, 0
        best = (-9.0, 0.0, 0.0)
        for s in range(a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            up_any = False
            for _k in range(int(env.max_steps)):
                o, _r, t, tr, _i = env.step(pol(o))
                parts = [str(x) for x in R.floor_parts(env)]
                feet = sum(1 for x in parts if "foot" in x)
                tot += 1
                if feet >= 2 and len(parts) == feet:
                    fo += 1
                    tau = abs(float(
                        env.data.xfrc_applied[env.trunk_body_id, 4]))
                    taus.append(tau)
                    if tau >= clamp - 1e-6:
                        sat += 1
                    u = float(-env._projected_gravity()[2])
                    if u > best[0]:
                        best = (u, float(env.data.xpos[pel][2]),
                                float(env._foot_span()))
                if R.is_standing(env):
                    up_any = True
                if t or tr:
                    break
            stands += int(up_any)
        med = float(np.median(taus)) if taus else float("nan")
        pct = f"{100.0 * sat / len(taus):.0f} %" if taus else "-"
        print(f"  {clamp:7.0f}{med:11.1f}{pct:>8s}"
              f"{f'{fo}/{tot}':>11s}{best[0]:+9.2f}{best[1]:9.3f}"
              f"{best[2]:7.3f}{f'{stands}/{a.episodes}':>8s}", flush=True)
    print("\n  'at sat' is the share of FEET-ONLY frames at the clamp. If it is"
          "\n  low, the clamp is not the lever and raising it cannot help.")
    print("  is_standing needs uprightness >= 0.85, trunk up, feet only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
