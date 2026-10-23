"""DOES THE EXISTING POLICY STAND FROM WHAT THE SCRIPTED PUSH-UP DELIVERS?

Measured 2026-10-12 (`probe_prone_openloop.py`): driven through the servos
with NO policy, the authored reference lifts 39 of 40 harvested PRONE poses
off the floor and leaves them at ~0.253 m resting on both feet and both
forearms — about 40 mm below `pike`'s recorded 0.300 m, which is a
cold-placement figure.

So prone may be a COMPOSITION problem rather than a training one: scripted
push-up -> the policy that already stands 96/96 from supine -> Menlo's
walker. This asks that question directly, and prices the anchor the way this
repo requires before any start distribution is believed: the ZERO-ACTION
policy is scored from the same states.

The delivered state is captured as (qpos, qvel) and INJECTED into a fresh
episode, rather than continuing the prologue's episode, so the policy gets
its full horizon — and so the states are in the shape a start-state cache
needs if this works.

    uv run python scripts/probe_pushup_handover.py <run> [n_poses]
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

from microduck_local.robots import asimov_getup as G
from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import asimov_walk as W
from microduck_local.robots import spec as S
from microduck_local.run_record import replay_kwargs

HOLD_S = 1.5          # settle onto the delivered pose before handing over


def onnx(p):
    import onnxruntime as ort
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return lambda o: s.run(None, {n: o.reshape(1, -1).astype(np.float32)})[0][0]


def action(q, clip):
    a = (np.asarray(q) - np.asarray(W.DEFAULT_POSE)) / W.ACTION_SCALE
    return a.astype(np.float32).clip(-clip, clip)


def build(run: str, pose: str):
    kw = dict(replay_kwargs(Path("runs") / run))
    kw.pop("task", None)
    kw.pop("handoff_pay", None)
    kw["getup_label"] = pose
    return S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)


def deliver(env, start_q, stop_t):
    """Run the scripted push-up; return the (qpos, qvel) it leaves behind."""
    m, d = env.model, env.data
    clip = env.robot.action_clip
    d.qpos[:] = start_q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    t, dt = 0.0, 0.02
    while t < stop_t:
        d.xfrc_applied[:] = 0.0
        env.step(action(R.pose_at(t)[0], clip))
        t += dt
    q_end, _, _ = R.pose_at(stop_t)
    for _ in range(int(HOLD_S * 50)):
        d.xfrc_applied[:] = 0.0
        env.step(action(q_end, clip))
    lf = m.body(m.geom_bodyid[env.foot_geoms["left"]]).id
    rf = m.body(m.geom_bodyid[env.foot_geoms["right"]]).id
    span = float(np.hypot(d.xpos[lf][0] - d.xpos[rf][0],
                          d.xpos[lf][1] - d.xpos[rf][1]))
    return (d.qpos.copy(), d.qvel.copy(),
            float(d.xpos[env.trunk_body_id][2]), span)


def score(env, states, pol, walk):
    up_n = stay_n = 0
    for q, v, _z, _s in states:
        env.reset(seed=0)
        env.data.qpos[:] = q
        env.data.qvel[:] = v
        mujoco.mj_forward(env.model, env.data)
        o = env._obs() if hasattr(env, "_obs") else env._get_obs()
        handed = rose = False
        last = False
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
    rest = [a for a in argv[1:] if not a.startswith("--policy=")]
    n = int(rest[0]) if rest else 40
    # Extra policies to score FROM THE SAME delivered states in the SAME env.
    # One env for a cross-policy question is the right call — the question is
    # "does this policy stand from this state in the deployment env" — and
    # the flags are printed so a mismatch cannot hide (train-loop rule 1).
    extra = [a.split("=", 1)[1] for a in argv[1:] if a.startswith("--policy=")]

    stop = 0.0
    t, dt = 0.0, 0.01
    while t < R.duration() + 0.5:                 # end of pike's dwell
        if R.pose_at(t)[2] == R.VERIFIED_THROUGH:
            stop = t
        t += dt

    env = build(run, "prone")
    poses = G.load_cache("prone")[:n]
    states, low, wide = [], 0, 0
    for q in poses:
        st = deliver(env, q, stop)
        if st[2] < 0.15:                 # never left the floor
            low += 1
            continue
        if st[3] > R.EPISODE_SPAN_CUT_M:  # the episode cut would end it at step 1
            wide += 1
            continue
        states.append(st)
    print(f"the scripted push-up delivered {len(states)}/{len(poses)} usable "
          f"states  (dropped {low} still on the floor, {wide} wider than the "
          f"{R.EPISODE_SPAN_CUT_M} m episode span cut)")
    if not states:
        return 1
    print(f"  delivered trunk  median {np.median([s[2] for s in states]):.3f} m"
          f" | foot span median {np.median([s[3] for s in states]):.3f} m")

    pol = onnx(Path("runs") / run / "policy.onnx")
    walk = onnx(".cache/asimov_1/policy/policy.onnx")
    zero = np.zeros(env.action_space.shape, np.float32)

    cands = [("zero action", lambda _o: zero), (run, pol)]
    import json
    for r in extra:
        try:
            cands.append((r, onnx(Path("runs") / r / "policy.onnx")))
        except Exception as e:                       # noqa: BLE001
            print(f"  (skipping {r}: {type(e).__name__})")
    print(f"\nfrom the DELIVERED pose, {len(states)} episodes, "
          f"scored in {run[-6:]}'s env:")
    for name, p in cands:
        u, s = score(env, states, p, walk)
        flags = ""
        rj = Path("runs") / name / "run.json"
        if rj.is_file():
            k = json.loads(rj.read_text()).get("env_kwargs", {})
            flags = (f"  [trained from {k.get('start_phase') or k.get('getup_label')!r}"
                     f", clip {k.get('action_clip', 4.0)}]")
        print(f"  {name[-28:]:30} rises {u:3}/{len(states)}  "
              f"stays {s:3}/{len(states)}{flags}")
    print("\n  (the same policy from RAW prone scores ~12/48 = 25% — "
          "`getup_acceptance.py --pose=prone`)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
