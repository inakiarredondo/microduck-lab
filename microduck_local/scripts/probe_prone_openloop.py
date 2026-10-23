"""DOES *ANY* CONTROLLER LEAVE PRONE ON THIS BODY? The open-loop answer.

Four trained attempts at the prone get-up have returned nulls (12/48, 13/48,
9/48, all paired p > 0.3), and the failures go flat and still within a second
into states nothing recovers from — zero action 0/12, the policy itself 0/12.
This repo's rule for that situation is to play the authored motion OPEN LOOP
before training it again, because a route the servos cannot drive is not a
route a reward can find.

The authored reference already starts at `prone` — its own waypoint note says
a supine get-up is impossible on this robot and "needs a ROLL to prone
first" — so the reference IS a prone get-up. The question is whether it works
from the 220 prone poses in the harvested cache, or only from the single
hand-authored prone pose it was written for.

CONTROL FIRST: the same reference from its OWN start. Without it a failure
from cache poses cannot be told from "the reference does not work here".

    uv run python scripts/probe_prone_openloop.py [n_poses]
"""
from __future__ import annotations

import sys

import mujoco
import numpy as np

from microduck_local.robots import asimov_getup as G
from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import asimov_walk as W
from microduck_local.robots import spec as S

#: "Off its face" in the terms the repo's own classifier uses — which body
#: axis points up — not uprightness. Uprightness ~0 means LYING in ANY
#: orientation, and reading it as a roll indicator is how this probe's
#: predecessor reported "all 48 roll onto their side" for 36 poses the
#: classifier calls prone.
HOLD_S = 6.0

OFF_FACE = ("supine", "side_left", "side_right", "sitting", "kneeling")


def env_of():
    return S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_blend=0.0,
        getup_assist=0.0, getup_label="prone")


def action(q, clip):
    a = (np.asarray(q) - np.asarray(W.DEFAULT_POSE)) / W.ACTION_SCALE
    return a.astype(np.float32).clip(-clip, clip)


def pike_window():
    """When `pike`'s dwell ENDS, in reference time.

    Following the reference PAST its verified waypoint drives the robot into
    `rock_back`/`fold`/`stand`, none of which are verified — which is why the
    first version of this probe reported "final posture: prone" for all 40
    poses while peaking at pike height. Stop where the verification stops.
    """
    goal = R.VERIFIED_THROUGH
    t, dt, last = 0.0, 0.01, None
    while t < R.duration() + 0.5:
        if R.pose_at(t)[2] == goal:
            last = t
        t += dt
    if last is None:
        raise RuntimeError(f"{goal!r} never appears in the reference timeline")
    return last


def follow(env, start_q=None, pad_s=1.0, stop_t=None, hold_s=0.0):
    """Drive the authored reference open loop; return where it ended up.

    `stop_t` cuts the drive at a reference time and `hold_s` then COMMANDS
    THAT POSE for that long — because "reached" is a peak and the repo has
    been fooled by a peak before (a 0.615 m "stand" that was a robot
    face-down on its elbows). 6 s is this repo's hold window: a crouch that
    sits still for two seconds collapses at three.
    """
    m, d = env.model, env.data
    clip = env.robot.action_clip
    if start_q is None:                      # the reference's OWN start
        q0, rp0, _ = R.pose_at(0.0)
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[0:3] = [0.0, 0.0, 1.0]
        half = float(rp0) / 2.0
        d.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]
        for n, v in zip(W.JOINT_NAMES, q0):
            d.qpos[m.joint(n).qposadr[0]] = v
        mujoco.mj_forward(m, d)
        R.settle_onto_floor(m, d)
    else:
        d.qpos[:] = start_q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
    d.xfrc_applied[:] = 0.0

    t, dt = 0.0, 0.02
    total = R.duration() if stop_t is None else float(stop_t)
    peak_z, peak_u, best_label = -9.9, -9.9, "prone"
    while t < total + (pad_s if stop_t is None else 0.0):
        q, _, _ = R.pose_at(t)
        d.xfrc_applied[:] = 0.0
        env.step(action(q, clip))
        z = float(d.xpos[env.trunk_body_id][2])
        u = float(-env._projected_gravity()[2])
        peak_z, peak_u = max(peak_z, z), max(peak_u, u)
        lab = G.label(m, d)[0]
        if lab in OFF_FACE:
            best_label = lab
        t += dt
    arrive_z = float(d.xpos[env.trunk_body_id][2])

    hold = []
    if hold_s > 0.0:
        q, _, _ = R.pose_at(total)
        for _ in range(int(hold_s * 50)):
            d.xfrc_applied[:] = 0.0
            env.step(action(q, clip))
            hold.append(float(d.xpos[env.trunk_body_id][2]))

    return dict(z=float(d.xpos[env.trunk_body_id][2]),
                u=float(-env._projected_gravity()[2]),
                peak_z=peak_z, peak_u=peak_u, arrive_z=arrive_z,
                hold_z=(float(np.median(hold[-50:])) if hold else arrive_z),
                hold_min=(float(min(hold)) if hold else arrive_z),
                label=G.label(m, d)[0], off_face=best_label,
                contacts=sorted({R.part_of(m, b) for b in R.floor_parts(env)}),
                standing=bool(R.is_standing(env)))


def main(argv):
    n = int(argv[0]) if argv else 40
    env = env_of()

    # THE RIGHT CRITERION. The reference is verified through `pike`, NOT
    # through `stand` — the repo's staged design is "servos push up, policy
    # stands from pike" (15/16 vs 18/36 end to end). Judging this probe on
    # STANDING called its own control a failure at peak trunk 0.301 m, which
    # is pike's recorded 0.300 m to a millimetre.
    goal = R.BY_NAME[R.VERIFIED_THROUGH]
    want = float(goal.settled_z)
    tol = 0.04                       # the same window `chain()` uses
    print(f"criterion: reach {R.VERIFIED_THROUGH!r} at trunk {want:.3f} m "
          f"+-{tol} — the reference is NOT verified to stand")

    stop = pike_window()
    print(f"driving to t={stop:.2f}s (end of {R.VERIFIED_THROUGH}'s dwell), "
          f"then COMMANDING that pose for {HOLD_S:.0f}s")

    print("\nCONTROL — the reference from its OWN authored start:")
    c = follow(env, None, stop_t=stop, hold_s=HOLD_S)
    ok = abs(c["arrive_z"] - want) < tol
    held = abs(c["hold_z"] - want) < tol
    print(f"  arrives {c['arrive_z']:.3f} m -> "
          f"{'REACHED ' + R.VERIFIED_THROUGH if ok else 'off reference'}"
          f" | after {HOLD_S:.0f}s hold {c['hold_z']:.3f} m "
          f"(min {c['hold_min']:.3f}) -> "
          f"{'HELD at the recorded height' if held else ('DRIFTED to a lower STABLE pose' if c['hold_z'] > 0.6 * want else 'COLLAPSED to the floor')}")
    print(f"  on the floor: {', '.join(c['contacts']) or 'nothing'}")
    if not ok:
        print("  !! the reference misses its own verified waypoint here — a "
              "failure below would say nothing about the cache poses")

    poses = G.load_cache("prone")[:n]
    print(f"\nTHE QUESTION — the same reference from {len(poses)} harvested "
          f"prone poses:")
    rows = [follow(env, q, stop_t=stop, hold_s=HOLD_S) for q in poses]
    arrive = [abs(r["arrive_z"] - want) < tol for r in rows]
    holds = [abs(r["hold_z"] - want) < tol for r in rows]
    print(f"  ARRIVES at {R.VERIFIED_THROUGH:<6} {sum(arrive)}/{len(rows)}")
    settled = [r["hold_z"] > 0.6 * want for r in rows]
    print(f"  STILL AT {want:.3f} m after {HOLD_S:.0f}s  {sum(holds)}/{len(rows)}"
          f"   <- 'reached' was a PEAK; this is the number that matters")
    print(f"  still OFF THE FLOOR after {HOLD_S:.0f}s  {sum(settled)}/{len(rows)}"
          f"   <- a drift to a lower STABLE pose is not a collapse, and the "
          f"recorded {want:.3f} m is a COLD-PLACEMENT number")
    print(f"  arrive trunk median {np.median([r['arrive_z'] for r in rows]):.3f}"
          f" m | after hold {np.median([r['hold_z'] for r in rows]):.3f} m")

    # WHAT ARE THE MISSES? A count is not a diagnosis.
    miss = [r for r, a in zip(rows, arrive) if not a]
    hit = [r for r, a in zip(rows, arrive) if a]
    print(f"\n  THE {len(miss)} THAT MISS:")
    import collections
    print(f"    arrive trunk  {sorted(round(r['arrive_z'], 3) for r in miss)}")
    print(f"    too LOW (never got up) {sum(r['arrive_z'] < want - tol for r in miss)}"
          f" | too HIGH (overshot) {sum(r['arrive_z'] > want + tol for r in miss)}")
    print(f"    on the floor: {dict(collections.Counter(
        ','.join(r['contacts'])[:40] for r in miss))}")
    if hit:
        print(f"  for contrast, the {len(hit)} that arrive rest on: "
              f"{dict(collections.Counter(
                  ','.join(r['contacts'])[:40] for r in hit))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
