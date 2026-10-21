"""Can the robot STEP its feet under its mass, instead of leaping there?

The geometric bound says the CoM sits ~a leg length ahead of the feet, so the
support cannot be put under the mass in ONE step and widening is all that is
left. That argument is about a single step. It says nothing about SEVERAL.

Menlo ship a controller that takes small steps — their walker. It has never
been handed a low narrow pose. So: play the authored servo route (narrow,
0.215 m, free to reach), then hand the robot to the walker and see whether it
walks its feet forward under itself and comes up, at what span.

Scored on the pose that survives, and on the SPAN, because stepping forward
keeps the feet together while sprawling does not -- that is the whole point.
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--through", default="fold")
    ap.add_argument("--cmd", type=float, default=0.0,
                    help="forward velocity command given to the walker")
    a = ap.parse_args(argv)

    import mujoco
    import onnxruntime as ort
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0,
        getup_label="prone", actuator_force="dc", action_clip=4.0,
        fell_cut=False, max_episode_s=30.0)
    m, d = env.model, env.data
    default = np.asarray(env.robot.default_pose, float)
    scale = np.asarray(env.robot.action_scale, float)
    names = [w.name for w in R.WAYPOINTS]
    route = names[:names.index(a.through) + 1]
    fb = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]

    sess = ort.InferenceSession(str(W.policy_onnx()),
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    stood, narrow, spans, travel, zs = 0, 0, [], [], []
    for s in range(a.episodes):
        env.reset(seed=s)
        w0 = R.BY_NAME[route[0]]
        env._place(np.asarray(w0.joints, float), w0.root_pitch)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        cur = np.asarray(w0.joints, float)
        for nm in route[1:]:
            tgt = np.asarray(R.BY_NAME[nm].joints, float)
            for k in range(13):
                q = cur + (tgt - cur) * min(1.0, (k + 1) / 12)
                env.step(np.clip((q - default) / scale, -4.0, 4.0
                                 ).astype(np.float32))
            cur = tgt
        # HAND TO THE WALKER, which is a STEPPING controller.
        p0 = np.array([d.xpos[b][:2].copy() for b in fb])
        env.twist_cmd[:] = (a.cmd, 0.0, 0.0)
        o = env._get_obs()
        span, up = float(env._foot_span()), False
        for _k in range(int(a.seconds / 0.02)):
            act = sess.run(None, {inp: o[-78:][None].astype(np.float32)})[0][0]
            o, _r, t, tr, _i = env.step(act.astype(np.float32))
            env.twist_cmd[:] = (a.cmd, 0.0, 0.0)
            span = max(span, float(env._foot_span()))
            up = R.is_standing(env)
            if t or tr:
                break
        p1 = np.array([d.xpos[b][:2].copy() for b in fb])
        stood += int(up)
        narrow += int(up and span <= 0.40)
        spans.append(span)
        travel.append(float(np.linalg.norm(p1 - p0, axis=1).max()))
        zs.append(float(d.xpos[env.trunk_body_id][2]))

    print(f"servo route -> {a.through}, then Menlo's WALKER for {a.seconds:g} s, "
          f"cmd {a.cmd:g} m/s, {a.episodes} episodes\n")
    print(f"  stands {stood}/{a.episodes}   NARROW (<=0.40 m) {narrow}/{a.episodes}")
    print(f"  median peak span {np.median(spans):.3f} m   "
          f"median foot travel {np.median(travel):.3f} m   "
          f"final trunk {np.median(zs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
