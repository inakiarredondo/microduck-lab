"""FILM a get-up: video + a labelled contact sheet + what it actually did.

Reward curves and eval sums have lied repeatedly on this body, and so have
summary counts — "9/24 stands" was the shallow band of a curriculum, and a
"narrow crossing at 0.615 m" was the robot face down on its elbows with its
hips in the air. This renders the thing and labels every frame with the three
numbers that decide whether a get-up is a get-up:

    trunk height    upright (+1 standing, -1 face down)    foot span

plus what is TOUCHING the floor, because knees and forearms are not feet.

    uv run python scripts/film_getup.py --run <run> --from pike
    uv run python scripts/film_getup.py --servos            # no policy at all
    uv run python scripts/film_getup.py --servos --route    # the whole push-up

`--from` is any authored waypoint: prone, plant_forearms, draw_knees_in,
quadruped, pike, fold, crouch, stand.
"""
import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import spec as S

ROUTE = ["prone", "plant_forearms", "draw_knees_in", "quadruped", "pike"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="a trained run under runs/")
    ap.add_argument("--servos", action="store_true",
                    help="no policy: ramp the position servos")
    ap.add_argument("--route", action="store_true",
                    help="with --servos, play the whole authored push-up")
    ap.add_argument("--from", dest="start", default="pike")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--out", default="/tmp/film")
    a = ap.parse_args()

    clip = a.clip
    if clip is None and a.run:
        ek = json.load(open(f"runs/{a.run}/run.json")).get("env_kwargs") or {}
        clip = float(ek.get("action_clip") or S.get("asimov1").action_clip)
    clip = float(clip or S.get("asimov1").action_clip)

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, action_clip=clip)
    m, d = env.model, env.data
    TB = env.trunk_body_id
    sc = float(np.atleast_1d(env.robot.action_scale)[0])
    DEF = np.asarray(env.robot.default_pose, float)
    SZ = R.BY_NAME["stand"].settled_z
    net = vn = None
    if a.run:
        from stable_baselines3 import PPO
        net = PPO.load(f"runs/{a.run}/model", device="cpu")
        vn = pickle.load(open(f"runs/{a.run}/vecnormalize.pkl", "rb"))

    def act(q):
        return np.clip((np.asarray(q, float) - DEF) / sc,
                       -clip, clip).astype(np.float32)

    def touching():
        out = set()
        for c in range(int(d.ncon)):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if m.geom_bodyid[g1] == 0 or m.geom_bodyid[g2] == 0:
                b = int(m.geom_bodyid[g2] if m.geom_bodyid[g1] == 0
                        else m.geom_bodyid[g1])
                out.add(m.body(b).name.replace("_link", "").replace("left_", "L ")
                        .replace("right_", "R "))
        return out

    ren = mujoco.Renderer(m, height=420, width=560)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def frame(tag, driver):
        cam.azimuth, cam.elevation, cam.distance = 44, -9, 2.7
        cam.lookat[:] = [float(d.xpos[TB][0]), float(d.xpos[TB][1]), 0.33]
        ren.update_scene(d, cam)
        img = Image.fromarray(ren.render())
        dr = ImageDraw.Draw(img)
        rep = R.standing_report(env)        # the ONE definition of a stand
        z, u, good = rep["trunk_z"], rep["upright"], rep["standing"]
        sp = env._foot_span()
        dr.rectangle([0, 0, 559, 56],
                     fill=(20, 62, 24) if good else (26, 28, 64))
        dr.text((10, 6), f"{driver}   {tag}", fill=(255, 255, 255))
        dr.text((10, 22), f"trunk {z:.3f}   upright {u:+.2f}   span {sp:.3f}",
                fill=(170, 255, 170) if good else (180, 198, 255))
        # PARTS, not the vendor's link names: `knee_link` is the SHIN, and
        # reading those labels by eye produced a false finding once already.
        dr.text((10, 38), "on floor: " + (", ".join(R.floor_parts(env)) or "-"),
                fill=(205, 205, 215))
        return np.asarray(img), z, u, sp

    shots, clips_ = [], []
    phases = ROUTE if (a.servos and a.route) else [a.start]
    eps = 1 if a.servos else a.episodes
    for ep in range(eps):
        env.reset(seed=ep)
        w0 = R.BY_NAME[phases[0]]
        env._place(np.asarray(w0.joints, float), w0.root_pitch)
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        cur = np.asarray(w0.joints, float)
        for _ in range(40):
            d.xfrc_applied[:] = 0.0
            env.step(act(cur))
        f, *_ = frame(phases[0], "SERVOS" if a.servos else "START")
        shots.append(f)
        if a.servos:
            for nxt in phases[1:]:
                tgt = np.asarray(R.BY_NAME[nxt].joints, float)
                n = max(2, int(0.25 * 50))
                for k in range(n):
                    t = (k + 1) / n
                    t = t * t * (3.0 - 2.0 * t)
                    d.xfrc_applied[:] = 0.0
                    env.step(act(cur + (tgt - cur) * t))
                    f, *_ = frame(nxt.replace("_", " "), "SERVOS")
                    shots.append(f)
                cur = tgt
                for _ in range(25):
                    d.xfrc_applied[:] = 0.0
                    env.step(act(cur))
                    f, *_ = frame(nxt.replace("_", " "), "SERVOS")
                    shots.append(f)
            clips_.append((phases[-1], shots[-1]))
            continue
        obs = env._get_obs()
        for k in range(int(a.seconds * 50)):
            if net is None:
                u_ = np.zeros(env.nj, np.float32)
            else:
                n = np.clip((obs - vn.obs_rms.mean)
                            / np.sqrt(vn.obs_rms.var + vn.epsilon),
                            -vn.clip_obs, vn.clip_obs).astype(np.float32)
                u_ = net.predict(n, deterministic=True)[0]
            obs, _r, t_, tr, _i = env.step(np.asarray(u_, np.float32))
            f, z, uu, sp = frame(f"ep{ep} +{(k + 1) / 50:.2f}s",
                                 "ZERO ACTION" if net is None else "POLICY")
            shots.append(f)
            if t_ or tr:
                break
        rep = R.standing_report(env)
        print(f"  ep{ep}: ends trunk {rep['trunk_z']:.3f}  "
              f"upright {rep['upright']:+.2f}  span {env._foot_span():.3f}  "
              f"UP={rep['standing']}" + (f"   ({rep['why']})" if rep['why'] else ""))

    import imageio
    imageio.mimsave(out / "film.mp4", shots, fps=25, quality=8)
    step = max(1, len(shots) // 12)
    tiles = [Image.fromarray(s).resize((280, 210)) for s in shots[::step][:12]]
    sheet = Image.new("RGB", (6 * 280, 2 * 210), (12, 12, 14))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 6) * 280, (i // 6) * 210))
    sheet.save(out / "sheet.png")
    print(f"{out}/film.mp4   {out}/sheet.png   ({len(shots)} frames)")


if __name__ == "__main__":
    main()
