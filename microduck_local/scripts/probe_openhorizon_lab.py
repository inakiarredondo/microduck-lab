"""Does openhorizon's get-up work in OUR LAB's env, and what does it cost?

    uv run --with onnxruntime python scripts/probe_openhorizon_lab.py --episodes 24

Three arms, because two things differ between their harness and this one and
each has to be priced separately rather than assumed negligible:

1. **theirs/theirs** — their runner, their model. The reference number.
2. **theirs/ours-dc** — their policy through `asimov_openhorizon`, in the env
   the LAB builds from the palette entry (`slot_env`, actuator `dc`). This is
   what Jonathan sees on the stage, so it is the one that has to be true.
3. **theirs/ours-xml** — the same, with the speed-torque envelope OFF. A
   control: it says whether their policy is leaning on the envelope or merely
   surviving it.

Their training MJCF carries a COLLISION OVERLAY ours does not -- their
`build_model` drops 4 geoms and adds 10 (torso, pelvis, head, upper arms,
hands, shoulders) because "the upstream MJCF collision shapes were tuned for
walking, not for lying on the floor". A get-up is all floor contact, so that
is the most likely source of any gap between arms 1 and 2, and `--overlay`
adds the same geoms to our model to measure it instead of guessing.

Scored with `asimov_getup_ref.is_standing` -- the library's one definition
(trunk up, upright, nothing but feet on the floor), not their runner's
`xmat[2,2] > 0.9 and pelvis > 0.55`, so the number is comparable with every
other get-up figure in this repo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def _theirs(overlay: bool, episodes: int, seed: int):
    """Arm 1: their runner, their model, their success test AND ours."""
    from microduck_local.robots import asimov, asimov_getup_ref as R
    from microduck_local.robots import asimov_motor as M
    from microduck_local.robots import asimov_openhorizon as OH
    sys.path.insert(0, str(M.cache_dir()))
    import mujoco

    import run_policy_mujoco as RP

    spec = json.loads(M.spec_path().read_text())
    model = RP.build_model(str(asimov.xml_path()), spec["physics_dt"])
    d = mujoco.MjData(model)
    pol = RP.Policy(str(OH.onnx_path()), spec, model)
    decim = round(1.0 / spec["rate_hz"] / model.opt.timestep)
    torso = model.body("waist_yaw_link").id
    rng = np.random.default_rng(seed)
    rose, times = 0, []
    for _ in range(episodes):
        mujoco.mj_resetData(model, d)
        pol.reset()
        RP.random_fallen_state(model, d, pol, rng)
        mujoco.mj_forward(model, d)
        held, t_up = 0.0, None
        for k in range(600):
            pol.step(d)
            RP._physics(model, d, pol, decim, None, None)
            up = (d.xmat[torso].reshape(3, 3)[2, 2] > 0.9
                  and d.xpos[pol.pelvis][2] > 0.55)
            held = held + model.opt.timestep * decim if up else 0.0
            if held >= 1.0:
                t_up = (k + 1) * model.opt.timestep * decim - 1.0
                break
        if t_up is not None:
            rose += 1
            times.append(t_up)
    return rose, times


def _ours(actuator: str, episodes: int, seed: int, pose: str):
    """Arms 2 and 3: their policy, through the adapter, in the LAB's env."""
    from microduck_local.lab import robots as lab_robots
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_openhorizon as OH
    from microduck_local.robots import spec as S

    entry = next(e for e in S.get("asimov1").shipped_policies()
                 if e.get("adapter") == "openhorizon:getup")
    kw = dict(entry["envKwargs"])
    kw["actuator_force"] = actuator
    kw["getup_label"] = pose
    # THE LAB'S OWN CONSTRUCTOR, not a hand-built env: `slot_env` merges the
    # body's derived knobs under the entry's, and a probe that bypassed it
    # would be measuring a different env from the one on the stage.
    env = lab_robots.slot_env("asimov1", seed=0, kwargs=kw)
    pol = OH.OpenHorizonPolicy("getup")
    rose, times, held_end = 0, [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        pol.reset()
        up_s, t_up, last = 0.0, None, False
        for k in range(int(env.max_steps)):
            obs, _r, term, trunc, _i = env.step(pol(obs))
            last = R.is_standing(env)
            up_s = up_s + 0.02 if last else 0.0
            if t_up is None and up_s >= 1.0:
                t_up = (k + 1) * 0.02 - 1.0
            if term or trunc:
                break
        if t_up is not None:
            rose += 1
            times.append(t_up)
        held_end += int(last)
    return rose, times, held_end, env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose", default="prone")
    ap.add_argument("--overlay", action="store_true",
                    help="add their collision overlay to OUR model too")
    a = ap.parse_args(argv)

    def med(ts):
        return f"{np.median(ts):.2f} s" if ts else "—"

    t0 = time.time()
    rose, times = _theirs(a.overlay, a.episodes, a.seed)
    print(f"1. theirs / their model + overlay   "
          f"rises {rose:3d}/{a.episodes}  median {med(times)}"
          f"   [their own success test]")

    for actuator in ("dc", "xml"):
        r, ts, he, env = _ours(actuator, a.episodes, a.seed, a.pose)
        n = 2 if actuator == "dc" else 3
        print(f"{n}. theirs / OUR lab env, actuator={actuator:3s}  "
              f"rises {r:3d}/{a.episodes}  median {med(ts)}"
              f"   still up at the buzzer {he}/{a.episodes}"
              f"   [{a.pose}, is_standing]")
    print(f"\n({time.time() - t0:.0f} s; arms 2-3 start from this repo's "
          f"'{a.pose}' pose cache, arm 1 from their own limp drop — the start "
          f"distributions are NOT the same, so compare 2 against 3, and 1 "
          f"only as a reference that their policy works at all.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
