"""The SAME get-up policy, scored under both actuator models, paired by seed.

    uv run --with onnxruntime python scripts/probe_actuator_ab.py \
        teach-asimov_getup_turn-967072 --episodes 48 --pose supine,prone

Why this exists, and what it retracts. Every Asimov get-up in this repo was
trained with `actuator='xml'`: the MJCF's position servos clamped at Menlo's
URDF continuous effort (12-45 N*m) with NO speed dependence -- a robot that is
weak standing still and unlimited once it is moving. The real motors have a
speed-torque curve, and `robots/asimov_motor` now implements the published one
(mjlab's `dc_motor_clip`, from openhorizon-labs' spec, whose own get-up rises
24/24 inside it).

An earlier measurement reported this policy going **24/24 -> 0/24 under
limits**. That number came from `AsimovGetupEnv._apply_vel_limit`, an opposing
force proportional to overspeed, and it was wrong in BOTH directions: it took
away torque above the rated speed without ever granting the `peak_torque` --
up to 3x the continuous effort, 120 N*m at the hip -- that the real envelope
allows below it. So it over-penalised the policy, and this script is the
re-measurement. The brake is deleted.

Scored with the library's own `is_standing` (trunk up, upright, nothing but
feet down), paired by seed with an exact McNemar, because both arms see the
same episodes and "a median that moved is a null at p=0.424 on the same
episodes" is this repo's most repeated statistical lesson.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _acceptance():
    """The gate's own `load` and `mcnemar` — one definition, not a copy."""
    spec = importlib.util.spec_from_file_location(
        "_getup_acceptance", HERE / "getup_acceptance.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--pose", default="supine,prone")
    ap.add_argument("--source", default="urdf",
                    choices=("urdf", "openhorizon"),
                    help="whose continuous effort the 'dc' arm enforces")
    a = ap.parse_args(argv)

    A = _acceptance()
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_motor as M
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs

    rec = dict(replay_kwargs(Path("runs") / a.run))
    trained_on = rec.get("actuator_force", "xml")
    act = A.load(a.run)
    lim = M.limits(a.source)
    print(f"{a.run}\n  trained under actuator={trained_on!r}; "
          f"'dc' enforces {lim.note()}\n  "
          f"{a.episodes} episodes per pose, paired by seed\n")

    def build(pose, actuator):
        kw = dict(rec)
        kw.pop("task", None)
        kw.pop("handoff_pay", None)   # a TRAINING cut, not a deployment one
        kw["getup_label"] = pose
        kw["actuator_force"] = actuator
        kw.setdefault("limit_cost", 1.0)
        return S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)

    def episode(env, seed):
        o, _ = env.reset(seed=seed)
        rose = up_end = False
        peak_hr = 0.0
        for _ in range(int(env.max_steps)):
            o, _r, t, tr, _i = env.step(act(o))
            st = R.is_standing(env)
            rose = rose or st
            up_end = st
            if env._motor is not None:
                peak_hr = max(peak_hr, float(env._motor.headroom().max()))
            if t or tr:
                break
        return rose, up_end, peak_hr

    for pose in a.pose.split(","):
        pose = pose.strip()
        if not pose:
            continue
        out = {}
        for actuator in ("xml", "dc"):
            env = build(pose, actuator)
            rows = [episode(env, s) for s in range(a.episodes)]
            out[actuator] = rows
            rises = sum(r[0] for r in rows)
            stays = sum(r[1] for r in rows)
            hr = max(r[2] for r in rows)
            print(f"  {pose:7s} actuator={actuator:4s}  "
                  f"rises {rises:3d}/{a.episodes}  stays {stays:3d}/"
                  f"{a.episodes}"
                  + (f"  peak headroom {hr:.3f}" if actuator == "dc" else ""))
        print()
        for label, i in (("rises", 0), ("stays", 1)):
            b = [r[i] for r in out["xml"]]
            c = [r[i] for r in out["dc"]]
            bb, cc, pv = A.mcnemar(b, c)
            verdict = ("real at p<=0.05" if pv <= 0.05
                       else "NULL — inside the noise")
            print(f"    {label:6s} xml {sum(b):3d}/{a.episodes} vs "
                  f"dc {sum(c):3d}/{a.episodes} | discordant {bb}-{cc} "
                  f"| p = {pv:.4f} | {verdict}")
            if pv > 0.05:
                need = next(k for k in range(1, 60)
                            if A._one_sided_p(k) <= 0.05)
                if bb + cc < need:
                    print(f"           (only {bb + cc} episodes disagreed; an "
                          f"exact McNemar needs {need} all one-sided — NO "
                          f"INSTRUMENT, not no effect)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
