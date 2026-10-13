"""IF THE ROBOT COULD ROLL ONTO ITS BACK, WOULD THE EXISTING POLICY STAND?

Measure the value of the roll BEFORE authoring the roll. The authored
reference asserts a supine rise is impossible on this robot and one must
"ROLL to prone first" — and `teach-asimov_getup_turn-540bc4` falsifies that
at 96/96 from supine while managing only ~25 % from prone. So the untried fix
for a face-down fall is the INVERSE of the authored premise: roll onto the
back and use the route that already works.

This does not author a roll. It takes each harvested PRONE pose, rotates the
root 180 deg about the body's long axis, settles it onto the floor, checks the
repo's own body-axis classifier now calls it supine, and scores the existing
policy from there. That is the CEILING the roll would buy: if it is low,
authoring a roll is wasted effort.

CONTROL: the harvested supine poses, through the same code path — so a low
number cannot be the harness.

    uv run python scripts/probe_roll_to_supine.py <run> [n_poses]
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import mujoco
import numpy as np

from microduck_local.robots import asimov_getup as G
from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import spec as S
from microduck_local.run_record import replay_kwargs


def onnx(p):
    import onnxruntime as ort
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return lambda o: s.run(None, {n: o.reshape(1, -1).astype(np.float32)})[0][0]


def build(run: str):
    kw = dict(replay_kwargs(Path("runs") / run))
    kw.pop("task", None)
    kw.pop("handoff_pay", None)
    kw["getup_label"] = "prone"
    return S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)


def rolled(env, q):
    """`q` with the root turned 180 deg about the body's LONG axis, settled."""
    m, d = env.model, env.data
    d.qpos[:] = q
    d.qvel[:] = 0.0
    # body-frame roll: q_new = q_old * rot(pi, x). MuJoCo quats are (w,x,y,z).
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(d.qpos[3:7], float),
                       np.array([0.0, 1.0, 0.0, 0.0]))
    d.qpos[3:7] = out
    mujoco.mj_forward(m, d)
    R.settle_onto_floor(m, d)           # MuJoCo's own contact distances
    mujoco.mj_forward(m, d)
    return d.qpos.copy(), G.label(m, d)[0]


def score(env, states, pol, walk):
    up_n = stay_n = 0
    for q in states:
        env.reset(seed=0)
        env.data.qpos[:] = q
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        o = env._obs() if hasattr(env, "_obs") else env._get_obs()
        handed = rose = last = False
        for _ in range(int(env.max_steps)):
            if not handed and env.handoff_ready():
                handed = True
            o, _r, t, tr, _ = env.step((walk if handed else pol)(o))
            last = R.is_standing(env)
            rose = rose or last
            if t or tr:
                break
        up_n += bool(rose)
        stay_n += bool(last)
    return up_n, stay_n


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    run = argv[0]
    n = int(argv[1]) if len(argv) > 1 else 40
    env = build(run)
    pol = onnx(Path("runs") / run / "policy.onnx")
    walk = onnx(".cache/asimov_1/policy/policy.onnx")
    zero = np.zeros(env.action_space.shape, np.float32)

    prone = list(G.load_cache("prone")[:n])
    flipped, labels = [], []
    for q in prone:
        fq, lab = rolled(env, q)
        flipped.append(fq)
        labels.append(lab)
    print(f"rotated {len(prone)} prone poses 180 deg about the long axis; the "
          f"body-axis classifier now calls them: {dict(collections.Counter(labels))}")
    ok = [q for q, lab in zip(flipped, labels) if lab == "supine"]
    print(f"  {len(ok)}/{len(flipped)} land genuinely SUPINE\n")

    rows = [("prone, as harvested", prone),
            ("THE SAME POSES, ROLLED", flipped),
            ("harvested supine (CONTROL)", list(G.load_cache("supine")[:n]))]
    for name, st in rows:
        u, s = score(env, st, pol, walk)
        z, _ = score(env, st, lambda _o: zero, walk)
        print(f"  {name:28} policy rises {u:3}/{len(st)}  stays {s:3}/{len(st)}"
              f"   | zero action {z:3}/{len(st)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
