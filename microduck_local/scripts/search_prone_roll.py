"""AUTHOR A PRONE -> SUPINE ROLL, by scanning a small parameterised motion.

Measured 2026-10-12: from a genuinely supine pose the existing policy stands
**40/40**; from prone it manages 8/40. A crude 180 deg root flip lands only
4/40 genuinely supine and already lifts the policy to 19/40. So the whole
prone problem reduces to "get onto the back", and this searches for a servo
motion that does it.

The body constrains the shape. `shoulder_roll` ADDUCTS to zero only (left
-90..0, right 0..+90), so there is no reach-across-the-chest; the levers are
`waist_yaw` (+-90 deg, the biggest single joint on this robot), an arm planted
and pushing, and a hip/leg scissor (`hip_roll`, `hip_yaw`, +-45 deg).

Judged by the repo's own body-axis classifier after settling, never by
uprightness — uprightness ~0 means lying in ANY orientation, which is how an
earlier probe reported "rolled onto its side" for poses the classifier calls
prone.

    uv run python scripts/search_prone_roll.py [n_poses]
"""
from __future__ import annotations

import itertools
import sys

import mujoco
import numpy as np

from microduck_local.robots import asimov_getup as G
from microduck_local.robots import asimov_walk as W
from microduck_local.robots import spec as S

ON_BACK = ("supine",)
NEARLY = ("side_left", "side_right")


def build():
    return S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="prone")


def reach(env):
    """The joint box a scripted motion can actually COMMAND.

    `ACTION_SCALE` is 0.25 and the clip is 4.0, so every target is capped at
    **+-1.0 rad from `DEFAULT_POSE`**, intersected with the joint range. The
    first version of this search asked for waist 1.45 and shoulder 2.6 — both
    outside the box — so 32 candidates collapsed to the same commanded motion
    and all returned 0/6. Check a knob's reachable set before scanning it.
    """
    m = env.model
    span = float(env.robot.action_clip) * float(W.ACTION_SCALE)
    out = {}
    for n, dflt in zip(W.JOINT_NAMES, np.asarray(W.DEFAULT_POSE, float)):
        lo, hi = m.jnt_range[m.joint(n).id]
        out[n] = (max(lo, dflt - span), min(hi, dflt + span))
    return out


def act(q, clip):
    a = (np.asarray(q) - np.asarray(W.DEFAULT_POSE)) / W.ACTION_SCALE
    return a.astype(np.float32).clip(-clip, clip)


def pose(box, side: int, waist: float, push: float, scissor: float,
         tuck: float):
    """One roll candidate as a joint-target vector, CLAMPED to `box`.

    `waist`, `push`, `scissor`, `tuck` are fractions in [0,1] of the
    reachable travel in the chosen direction, so the scan cannot ask for
    anything the action space would silently clip.
    """
    q = dict(zip(W.JOINT_NAMES, np.array(W.DEFAULT_POSE, float)))
    s = "right" if side > 0 else "left"
    o = "left" if side > 0 else "right"

    def toward(name, frac, sign):
        lo, hi = box[name]
        d = q[name]
        return d + frac * ((hi - d) if sign > 0 else (lo - d))

    q["waist_yaw_joint"] = toward("waist_yaw_joint", waist, side)
    # the PUSHING arm: out to the side and elbow bent, to plant and drive
    q[f"{s}_shoulder_pitch_joint"] = toward(f"{s}_shoulder_pitch_joint", push, side)
    q[f"{s}_shoulder_roll_joint"] = toward(f"{s}_shoulder_roll_joint", 1.0, side)
    q[f"{s}_elbow_joint"] = toward(f"{s}_elbow_joint", 0.8, -side)
    # the FREE arm thrown the other way, out of the roll's path
    q[f"{o}_shoulder_pitch_joint"] = toward(f"{o}_shoulder_pitch_joint", 0.8, -side)
    # the legs scissor to carry the hips over
    q[f"{s}_hip_roll_joint"] = toward(f"{s}_hip_roll_joint", scissor, side)
    q[f"{o}_hip_roll_joint"] = toward(f"{o}_hip_roll_joint", scissor, side)
    q[f"{s}_hip_yaw_joint"] = toward(f"{s}_hip_yaw_joint", scissor, -side)
    q[f"{s}_knee_joint"] = toward(f"{s}_knee_joint", tuck, +1)
    out = np.array([q[n] for n in W.JOINT_NAMES], float)
    for i, n in enumerate(W.JOINT_NAMES):      # belt and braces
        out[i] = min(max(out[i], box[n][0]), box[n][1])
    return out


def play(env, start_q, target, secs=1.6, settle_s=2.0, windup=None,
         windup_s=0.3):
    """Optional WIND-UP, then a ramp to `target`, then settle and classify.

    A log roll is dynamic: it needs angular momentum from a fast asymmetric
    throw, not a slow approach to a held pose. The first scan ramped over
    1.6 s — quasi-static — and got 0/6 onto the back from every one of 32
    candidates. This repo already measured that a reference HAS a tempo (this
    robot stands at 0.25 s a leg and falls at 0.5), so `secs` is now scanned.
    """
    m, d = env.model, env.data
    clip = env.robot.action_clip
    d.qpos[:] = start_q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    cur = np.array([d.qpos[m.joint(n).qposadr[0]] for n in W.JOINT_NAMES])
    if windup is not None:
        nw = max(2, int(windup_s * 50))
        for k in range(nw):
            f = (k + 1) / nw
            d.xfrc_applied[:] = 0.0
            env.step(act(cur + (windup - cur) * f, clip))
        cur = windup
    n = max(2, int(secs * 50))
    for k in range(n):
        f = (k + 1) / n
        f = f * f * (3.0 - 2.0 * f)              # smoothstep
        d.xfrc_applied[:] = 0.0
        env.step(act(cur + (target - cur) * f, clip))
    for _ in range(int(settle_s * 50)):
        d.xfrc_applied[:] = 0.0
        env.step(act(target, clip))
    return G.label(m, d)[0], float(-env._projected_gravity()[2])


def main(argv):
    n_pose = int(argv[0]) if argv else 6
    env = build()
    poses = list(G.load_cache("prone")[:n_pose])

    box = reach(env)
    print(f"the commandable box is DEFAULT_POSE +-"
          f"{env.robot.action_clip * W.ACTION_SCALE:.2f} rad; e.g. waist_yaw "
          f"{box['waist_yaw_joint'][0]:+.2f}..{box['waist_yaw_joint'][1]:+.2f}")
    grid = list(itertools.product(
        (+1, -1),                 # roll right / left
        (1.0,),                   # waist twist, FRACTION of reachable travel
        (1.0,),                   # pushing shoulder pitch, fraction
        (1.0,),                   # hip scissor, fraction
        (0.0, 0.6),               # knee tuck, fraction
        (0.16, 0.30, 0.60, 1.20),  # TEMPO: seconds for the throw
        (False, True),            # wind up to the opposite side first?
    ))
    print(f"scanning {len(grid)} roll candidates x {len(poses)} prone poses")
    best = []
    for side, waist, push, scissor, tuck, secs, wind in grid:
        tgt = pose(box, side, waist, push, scissor, tuck)
        wu = pose(box, -side, 0.7, 0.3, 0.5, tuck) if wind else None
        labs = [play(env, q, tgt, secs=secs, windup=wu)[0] for q in poses]
        back = sum(x in ON_BACK for x in labs)
        near = sum(x in NEARLY for x in labs)
        best.append((back, near, side, secs, wind, tuck, labs))
    best.sort(key=lambda r: (-r[0], -r[1]))
    print(f"\n{'on back':>8} {'on side':>8}  side  tempo  windup  knee")
    for back, near, side, secs, wind, tuck, labs in best[:8]:
        print(f"{back:>5}/{len(poses)} {near:>5}/{len(poses)}  "
              f"{side:+d}    {secs:.2f}   {str(wind):5}   {tuck:.2f}")
    import collections
    b = best[0]
    print(f"\nbest candidate lands: {dict(collections.Counter(b[6]))}")
    print(f"  params: side {b[2]:+d} tempo {b[3]} windup {b[4]} knee {b[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
