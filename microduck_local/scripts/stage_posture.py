"""WHAT POSTURE IS EACH BODY ON THE LAB STAGE IN? Read off the live frames.

There is no GET for this — the roster and poses exist only in the streamed
frames — and reading it has been needed repeatedly: to check a staged get-up
actually starts face-down, and to tell a stand from a fall without taking a
screenshot.

The trap this script exists to avoid: `bodies` is emitted in SCENE order and
looked up by NAME (`Duck.pose_payload`), so index 0 is NOT the root and not
the torso. Decoding slot 0 as the trunk reported "other" for 42 consecutive
episode starts on 2026-10-13 — including for a policy that can only ever
start supine, which is what gave it away. The slot is resolved here through
the same mapping the server emits with.

`f_z` is the world-z component of the torso's FORWARD axis: -1 face down,
+1 face up, ~0 on a side. It is what `asimov_getup.label` takes the argmax
of, and what the get-up's `roll_pay` term pays. Uprightness cannot be used —
it reads ~0 for prone, supine AND side alike.

    uv run --with websockets python scripts/stage_posture.py [host:port] [secs]
"""
from __future__ import annotations

import asyncio
import json
import sys

import numpy as np
import websockets


def torso_slot(robot: str) -> int | None:
    """Which index of `bodies` carries the torso, for this body's scene."""
    from microduck_local import viz_server as V
    from microduck_local.robots import asimov_getup as G
    from microduck_local.robots import spec as S
    try:
        env = S.get(robot).env_class("getup")(
            domain_rand=False, obs_noise=False, seed=0)
    except Exception:                               # noqa: BLE001
        return None
    m = env.model
    import mujoco
    want = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, G.TORSO_BODY)
    mapping = (V._scene_to_env(m) if robot == "microduck"
               else V._robot_scene_to_env(m, robot))
    for i, (env_b, parent_b) in enumerate(mapping):
        if (env_b if env_b >= 0 else parent_b) == want:
            return i
    return None


def f_z(quat) -> float:
    """World-z of the body's local +x axis. MuJoCo quats are (w, x, y, z)."""
    w, x, y, z = (float(v) for v in quat)
    return 2.0 * (x * z - w * y)


def name_of(f: float) -> str:
    if f < -0.5:
        return "face DOWN"
    if f > 0.5:
        return "face UP"
    return "on a side / upright"


async def watch(addr: str, secs: float):
    slots: dict[str, int | None] = {}
    seen: dict[str, dict] = {}
    async with websockets.connect(f"ws://{addr}/ws", open_timeout=8) as ws:
        import time
        t0 = time.time()
        while time.time() - t0 < secs:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=8))
            for d in (msg.get("ducks") or []):
                who, robot = d.get("id"), d.get("robot")
                b, step = d.get("bodies") or [], d.get("step")
                if not b or robot is None:
                    continue
                if who not in slots:
                    slots[who] = torso_slot(str(robot))
                i = slots[who]
                if i is None or i >= len(b):
                    continue
                r = seen.setdefault(who, {"pol": d.get("policy"), "now": [],
                                          "starts": [], "last": None})
                f = f_z(b[i][3:7])
                r["now"].append(f)
                if r["last"] is not None and step is not None and step < r["last"]:
                    r["starts"].append(f)
                if step is not None:
                    r["last"] = step
    return seen


def main(argv):
    addr = argv[0] if argv else "127.0.0.1:8788"
    secs = float(argv[1]) if len(argv) > 1 else 60.0
    seen = asyncio.run(watch(addr, secs))
    if not seen:
        print(f"no bodies with poses on the {addr} stage")
        return 1
    for who, r in sorted(seen.items()):
        st = np.array(r["starts"]) if r["starts"] else np.array([])
        pol = (r["pol"] or "?").split("-")[-1]
        print(f"{who:8} {pol:9} {len(r['now'])} frames | f_z now "
              f"{r['now'][-1]:+.2f} ({name_of(r['now'][-1])})")
        if len(st):
            print(f"{'':8} {len(st)} episode starts: face DOWN "
                  f"{int((st < -0.5).sum())}, face UP {int((st > 0.5).sum())}, "
                  f"other {int(((st >= -0.5) & (st <= 0.5)).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
