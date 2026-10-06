"""Harvest get-up trajectories from openhorizon-labs' policy, in OUR model.

    uv run --with onnxruntime python scripts/harvest_getup_clips.py --episodes 40

Why this exists. Six optimisers, three ladder designs and ~50 M training steps
failed to cross the floor->stand gap here, and every structural argument that it
could not be done described OUR ROUTE, not the robot:
`openhorizon-labs/asimov1-getup-safefall` (Apache-2.0, pinned sha
f657abede4c0e8c74b96d7a2546568b63fed619f) rises **20/20, median 1.22 s** on our
own fetched `asimov_1.xml` in our own MuJoCo. So a working demonstration exists
in our physics, and the problem becomes imitation rather than search.

What it records, per control step at 50 Hz: the full `qpos` (7 root + 23 joints)
and the action. Joint angles are the right target because their runner drives
`d.qfrc_applied` with its own DC-motor model
(`hi = min(effort_limit, peak_torque * (1 - qd/velocity_limit))`, effort 40,
peak 120) and bypasses MuJoCo's actuators entirely, while our envs use position
servos at a flat +/-45. Ours is the stronger of the two at low speed and never
weaker, so a trajectory theirs produces is one ours can be asked to track; the
TORQUES, by contrast, are not transferable and are deliberately not saved.

Their start states come from their own `random_fallen_state` (limp drop from a
random orientation, 2.5 s settle), which is the same construction as this repo's
pose cache. Only SUCCESSFUL episodes are kept, by their own criterion: torso
z-axis up > 0.9 and pelvis > 0.55 m, held 2.0 s.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
THEIRS = HERE / ".cache/openhorizon_getup"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / ".cache/asimov_1/getup_clips.npz"))
    a = ap.parse_args()

    if not (THEIRS / "run_policy_mujoco.py").is_file():
        raise SystemExit(f"their runner is not fetched: {THEIRS}")
    sys.path.insert(0, str(THEIRS))
    import mujoco
    import run_policy_mujoco as R          # noqa: E402  (their code, their layout)

    spec = json.loads((THEIRS / "policies/getup_policy_spec.json").read_text())
    model = R.build_model(str(HERE / ".cache/asimov_1/xmls/asimov_1.xml"),
                          spec["physics_dt"])
    d = mujoco.MjData(model)
    pol = R.Policy(str(THEIRS / "policies/getup.onnx"), spec, model)
    decim = round(1.0 / spec["rate_hz"] / model.opt.timestep)
    torso = model.body("waist_yaw_link").id
    rng = np.random.default_rng(a.seed)

    clips, starts, times, kept = [], [], [], 0
    for ep in range(a.episodes):
        mujoco.mj_resetData(model, d)
        pol.reset()
        R.random_fallen_state(model, d, pol, rng)
        mujoco.mj_forward(model, d)
        start = d.qpos.copy()
        traj, stood, t_stand = [], 0.0, None
        for k in range(500):
            pol.step(d)
            R._physics(model, d, pol, decim, None, None)
            traj.append(d.qpos.copy())
            up = (d.xmat[torso].reshape(3, 3)[2, 2] > 0.9
                  and d.xpos[pol.pelvis][2] > 0.55)
            stood = stood + model.opt.timestep * decim if up else 0.0
            if stood >= 2.0:
                t_stand = (k + 1) * model.opt.timestep * decim - 2.0
                break
        if t_stand is not None:
            clips.append(np.asarray(traj)); starts.append(start); times.append(t_stand)
            kept += 1
        print(f"  ep {ep:3d}  {'STOOD in %.2fs' % t_stand if t_stand else 'failed':24} "
              f"({len(traj)} control steps)", flush=True)

    if not clips:
        raise SystemExit("no successful episode — nothing to imitate")
    lens = np.array([len(c) for c in clips])
    flat = np.concatenate(clips, axis=0)
    np.savez(a.out, qpos=flat, lengths=lens, starts=np.asarray(starts),
             time_to_stand=np.asarray(times), joint_names=np.array(pol.names),
             source="openhorizon-labs/asimov1-getup-safefall@"
                    "f657abede4c0e8c74b96d7a2546568b63fed619f")
    print(f"\nkept {kept}/{a.episodes} successful clips -> {a.out}")
    print(f"  median time to stand {np.median(times):.2f} s; "
          f"clip length {lens.min()}-{lens.max()} control steps "
          f"({lens.min()*0.02:.2f}-{lens.max()*0.02:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
