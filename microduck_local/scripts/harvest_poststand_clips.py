"""Clips that START where the deployed policy fails, labelled by the teacher.

`teach-asimov_getup_clip-0e31e5` rises 34/36 and is still up at the buzzer only
26/36: **29 of 36 episodes stand and THEN fall**. The measured cause is
covariate shift at the hold — action error 0.000702 rad^2 on the rise against
0.000064 on the hold, i.e. the clone is faithful on the TEACHER's states and
fails on its OWN, arriving upright in configurations the demonstration never
visited.

`harvest_getup_clips.py` starts from the teacher's `random_fallen_state`, so
every clip begins on the floor and the hold is only ever the tail of a long
trajectory. This starts each clip at the moment the DEPLOYED policy first
stands, and records what the TEACHER does from there.

It prices the idea before building on it: if the teacher cannot hold from those
states either, they are unrecoverable and no amount of supervision there helps.
That number is printed first.

    uv run --with onnxruntime python scripts/harvest_poststand_clips.py \
        --student teach-asimov_getup_clip-0e31e5 --episodes 60
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
THEIRS = HERE / ".cache/openhorizon_getup"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--hold-steps", type=int, default=150,
                    help="control steps of teacher to record from each start")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / ".cache/asimov_1/poststand_clips.npz"))
    a = ap.parse_args()

    sys.path.insert(0, str(THEIRS))
    import mujoco
    import run_policy_mujoco as R
    from stable_baselines3 import PPO

    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    run = HERE / "runs" / a.student
    meta = json.loads((run / "run.json").read_text())
    kw = {k: v for k, v in (meta.get("env_kwargs") or {}).items()
          if k in ("track_tol", "history", "force_lag", "action_clip")}
    env = S.get("asimov1").env_class("getup_clip")(
        domain_rand=False, obs_noise=False, seed=a.seed, **kw)
    m, d = env.model, env.data
    print(f"student env flags: {kw}", flush=True)

    net = PPO.load(str(run / "model"), device="cpu")
    with open(run / "vecnormalize.pkl", "rb") as fh:
        vn = pickle.load(fh)
    mu, va, cl, eps = vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs, vn.epsilon

    def student(o):
        n = np.clip((o - mu) / np.sqrt(va + eps), -cl, cl).astype(np.float32)
        return net.predict(n, deterministic=True)[0]

    spec = json.loads((THEIRS / "policies/getup_policy_spec.json").read_text())
    pol = R.Policy(str(THEIRS / "policies/getup.onnx"), spec, m)
    assert list(pol.names) == list(W.JOINT_NAMES), "their joint order is not ours"
    decim = round(1.0 / spec["rate_hz"] / m.opt.timestep)
    torso = m.body("waist_yaw_link").id
    rng = np.random.default_rng(a.seed)

    def upright():
        return (float(d.xmat[torso].reshape(3, 3)[2, 2]) > 0.9
                and float(d.xpos[pol.pelvis][2]) > 0.55)

    clips, starts, kept, reached, teacher_held = [], [], 0, 0, 0
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        stand_q = None
        for _ in range(260):
            obs, _r, t, tr, _i = env.step(np.asarray(student(obs), np.float32))
            if stand_q is None and upright():
                stand_q = d.qpos.copy()          # the instant it first stands
                break
            if (t or tr) and bool(getattr(env, "_stood", False)):
                break
        if stand_q is None:
            print(f"  ep {ep:3d}  student never stood", flush=True)
            continue
        reached += 1

        # now the TEACHER, from exactly that state
        d.qpos[:] = stand_q
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        pol.reset()
        traj, up_steps = [], 0
        for _ in range(a.hold_steps):
            pol.step(d)
            for _ in range(decim):
                d.qfrc_applied[:] = 0.0
                d.ctrl[np.array([m.actuator(n).id for n in W.JOINT_NAMES])] = pol.target
                mujoco.mj_step(m, d)
            traj.append(d.qpos.copy())
            up_steps += int(upright())
        held = up_steps >= int(0.9 * a.hold_steps)
        teacher_held += int(held)
        if held:
            clips.append(np.asarray(traj))
            starts.append(stand_q)
            kept += 1
        print(f"  ep {ep:3d}  student stood; teacher then held "
              f"{up_steps}/{a.hold_steps} steps {'KEPT' if held else 'dropped'}",
              flush=True)

    print(f"\nstudent reached a stand in {reached}/{a.episodes} episodes")
    if reached:
        print(f"THE PRICE OF THE IDEA: the teacher holds from the student's own "
              f"post-stand state in **{teacher_held}/{reached}** of them "
              f"({100*teacher_held/reached:.0f} %).")
        if teacher_held < 0.5 * reached:
            print("  -> under half. These states are largely UNRECOVERABLE even "
                  "for the teacher, so supervision there cannot be the fix, and "
                  "the gap is the approach to the stand, not the hold itself.")
    if not clips:
        raise SystemExit("no recoverable post-stand state — nothing to imitate")
    lens = np.array([len(c) for c in clips])
    np.savez(a.out, qpos=np.concatenate(clips, axis=0), lengths=lens,
             starts=np.asarray(starts),
             time_to_stand=np.zeros(len(clips)),
             joint_names=np.array(pol.names),
             source=f"post-stand states of {a.student}, labelled by "
                    "openhorizon-labs/asimov1-getup-safefall")
    print(f"kept {kept} clips -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
