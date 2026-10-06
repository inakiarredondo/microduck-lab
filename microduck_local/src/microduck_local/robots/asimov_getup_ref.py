"""The get-up as an AUTHORED REFERENCE — `docs/roadmap.md` Track 14, phase 3c.

Four get-up runs paid a reward built on `height`, `u_z` and `feet` and found
four different ways to stop at the first phase. The diagnosis, measured
2026-10-04, is that those three axes are **non-monotone along a correct get-up**
and so cannot shape it however they are weighted — this body's own ladder, every
row of it verified to hold for 6 s under its own position servos, is:

| phase | root pitch | pelvis z | u_z | what carries it | peak torque |
|---|---|---|---|---|---|
| `supine` | -1.57 | 0.112 | +0.00 | the trunk | 14 % |
| `quadruped` | +1.45 | 0.211 | **-0.06** | ankles + knees + forearms | 58 % |
| `pike` | +1.20 | 0.300 | **-0.37** | ankles + forearms | 66 % |
| `fold` | +0.60 | 0.517 | +0.83 | **ankles alone** | 36 % |
| `crouch` | +0.30 | 0.561 | +0.96 | ankles | 14 % |
| `stand` | 0.00 | 0.608 | +1.00 | ankles | 14 % |

`u_z` goes +0.00 -> -0.37 -> +0.83: the robot must turn FACE DOWN to get its
forearms under it, so any reward paying uprightness per step is paying it to
skip the only phase that can lift it. That is the whole of what the first four
runs discovered, four times.

**Four things this body does that a get-up authored from anatomy gets wrong.**

1. **There are no hands.** `left_wrist_yaw_link` carries a visual mesh and no
   collision geom at all, so a hand cannot touch the floor or bear any load.
   The weight-bearing surfaces are the pelvis, waist, neck, hip links, the knee
   (a 0.13 m half-length capsule down the whole shin), the four 5 mm foot
   spheres per ankle, the shoulder rolls and the ELBOW. So Asimov plants
   FOREARMS. `asimov_walk._effectors` has the measurement.
2. **It cannot squat.** `ankle_pitch` is limited to +-0.35 rad, so the shin
   cannot tilt forward and the knees stay over the ankles. The deepest pose it
   holds on its feet alone is `fold` at 0.517 m — 81 % of standing — and that
   one only because the arms reach FORWARD to carry the centre of mass over the
   feet. Everything below 0.517 m needs a non-foot contact.
3. **It cannot sit up and rise.** The route that never turns over — supine, sit
   with the knees drawn up, tuck the feet under, push — was measured across 12
   variants and every one of them sags 0.35 to 0.58 rad away from the commanded
   pose: the hip servos cannot hold a knees-up sit against gravity. The floor
   has to take the load through the knees and forearms, which means turning
   over, which is why `u_z` dips.
4. **`pike` -> `fold` IS NOT A POSE.** Between 0.300 m and 0.517 m there is no
   configuration that holds: crossing it means rocking the pelvis back 0.217 m
   and the trunk through 1.20 of uprightness while the support drops from four
   contacts to two. Nothing static lives there. A phase state machine over
   held poses therefore cannot represent this get-up, and that is why the
   reference below is a TIMED trajectory with `ROCK_BACK` marked as the span
   where tracking, not holding, is the only thing being asked for.

**The verification discipline that produced the table.** Each row is placed,
commanded by the position servos, allowed to settle, and then watched. The
window matters: a 2 s window called a 0.511 m crouch the deepest hold, and it
collapses at 3 s — a slow collapse and a hold are indistinguishable early. 6 s
is the window here, and `tests/test_asimov_getup_ref.py` pins it.
"""

from __future__ import annotations

from pathlib import Path as _P

import numpy as np

from . import asimov_walk as W

#: Joint-pair stems whose two sides take OPPOSITE values for one symmetric
#: pose, because the right-side axis is -Y (`model.joint(n).axis`, measured).
MIRRORED = ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "elbow")

_IDX = {n: i for i, n in enumerate(W.JOINT_NAMES)}


def joints(**kw) -> np.ndarray:
    """A 23-vector from readable kwargs; `DEFAULT_POSE` for anything unset.

    A mirrored kwarg IS THE LEFT SIDE'S VALUE and the right side takes its
    negation, so passing a joint its own default is a no-op — the property that
    makes a pose draft readable, and a real trap when it is missing. The sign
    `DEFAULT_POSE` uses is NOT uniform across the pairs: `left_knee` is +0.45
    against right -0.45, but `left_hip_pitch` is -0.15 against right +0.15, and
    `left_shoulder_pitch` is -0.25 while `left_elbow` is +0.40. A helper that
    applied one sign to all of them swung both arms the opposite way from the
    default while the legs obeyed, and a crouch that holds read as collapsing.
    """
    q = np.array(W.DEFAULT_POSE, dtype=np.float64)
    for k, v in kw.items():
        if k in _IDX:
            q[_IDX[k]] = v
            continue
        if f"{k}_joint" in _IDX:
            q[_IDX[f"{k}_joint"]] = v
            continue
        hits = 0
        for side, mul in (("left", 1.0), ("right", -1.0 if k in MIRRORED else 1.0)):
            n = f"{side}_{k}_joint"
            if n in _IDX:
                q[_IDX[n]] = mul * v
                hits += 1
        if not hits:
            raise KeyError(f"no joint of {W.spec().id} matches {k!r}")
    return q


_AABB_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1)
                          for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def settle_onto_floor(model, data, clearance: float = 0.001,
                      iters: int = 12) -> float:
    """Lower (or raise) the free root until the robot just touches the floor.

    Returns the net drop applied. Uses MuJoCo's OWN collision detection rather
    than a bound on the geometry, because every bound is wrong somewhere:

    * `geom_rbound` is a BOUNDING-SPHERE radius, so a 0.13 m half-length capsule
      reports its bottom up to 6 cm too low;
    * the geom AABB, rotated, puts corners below a rounded surface.

    Placing with `rbound` left resets wrong in BOTH directions and nobody looked
    until the robot was seen hanging in the air on the `/sim` stage:
    `plant_forearms` floated **50 mm** above the floor and `draw_knees_in`
    **52 mm**, while `prone` started **25 mm**, `quadruped` 12 mm and `pike`
    13 mm BURIED in it. A floating reset falls before the policy acts; a buried
    one is shoved out by the solver on step 1, which is where the 100 % torque
    transients at reset were coming from.

    Bisection on the root height against the smallest contact distance: a few
    iterations are enough, and it is exact for whatever geometry the model has.
    """
    import mujoco

    def gap() -> float:
        mujoco.mj_forward(model, data)
        if data.ncon == 0:
            return float("inf")
        return min(float(data.contact[i].dist) for i in range(data.ncon))

    z0 = float(data.qpos[2])
    # START FROM A CHEAP GEOMETRIC ESTIMATE, then bisect in a tight bracket
    # around it. Scanning for the bracket from the staging height cost up to
    # 240 `mj_forward` calls — **625 ms per reset, 24 steps' worth** — and with
    # episodes of 6 to 100 steps at the hard rungs that is most of the training
    # budget: the lab fell from 788 to 99 steps/s at 37 % CPU, which is a
    # trainer waiting, not a machine throttling.
    #
    # The AABB is a BOUND and not the answer (that is why the bisection exists —
    # `geom_rbound` was out by 63 mm), but it is a good bracket: one forward for
    # the estimate, then ~14 more to land inside a millimetre.
    corners = _AABB_CORNERS
    gl = [g for g in range(model.ngeom)
          if int(model.geom_contype[g]) != 0 and model.geom_bodyid[g] != 0]
    mujoco.mj_forward(model, data)
    est = min(
        float((data.geom_xpos[g]
               + (model.geom_aabb[g][:3] + model.geom_aabb[g][3:] * corners)
               @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
        for g in gl)
    data.qpos[2] = z0 - est + 0.04          # clear by ~4 cm, bracket from there
    hi = float(data.qpos[2])
    lo = hi - 0.12
    data.qpos[2] = lo
    if gap() > 0.0:                          # bracket missed low: widen once
        lo -= 0.25
        data.qpos[2] = lo
    for _ in range(16):                      # 0.12 m / 2^16 is far below 1 um
        mid = 0.5 * (lo + hi)
        data.qpos[2] = mid
        if gap() > clearance:
            hi = mid
        else:
            lo = mid
    data.qpos[2] = hi
    mujoco.mj_forward(model, data)
    return float(data.qpos[2]) - z0


class Waypoint:
    """One pose on the reference, and what was MEASURED about it.

    `equilibrium` is the claim a test can falsify: True means the robot,
    commanded to this pose, was still in it 6 s later. False means it is a
    TRANSIT pose — the reference passes through it and nothing holds there, so
    a hold test on it would fail by design and asserting one would be a lie.
    """

    __slots__ = ("name", "root_pitch", "kw", "dwell_s", "equilibrium",
                 "verified", "settled_z", "settled_u", "contacts",
                 "torque_frac", "note")

    def __init__(self, name, root_pitch, kw, dwell_s, equilibrium,
                 settled_z, settled_u, contacts, torque_frac, note,
                 verified=False):
        self.name = name
        self.root_pitch = float(root_pitch)
        self.kw = dict(kw)
        self.dwell_s = float(dwell_s)
        self.equilibrium = bool(equilibrium)
        #: Was the leg INTO this waypoint executed open loop, servos only?
        self.verified = bool(verified)
        self.settled_z = float(settled_z)
        self.settled_u = float(settled_u)
        self.contacts = tuple(contacts)
        self.torque_frac = float(torque_frac)
        self.note = note

    @property
    def joints(self) -> np.ndarray:
        return joints(**self.kw)

    def __repr__(self) -> str:
        return (f"Waypoint({self.name!r}, z={self.settled_z:.3f}, "
                f"u={self.settled_u:+.2f}, eq={self.equilibrium})")


#: The reference, in order, for the PRONE get-up. Every `settled_*` number is
#: measured by `scripts/probe_asimov_getup_ref.py` with randomisation off, over
#: a 6 s window, and pinned by `tests/test_asimov_getup_ref.py`.
#:
#: `verified` is the stronger claim: the leg INTO this waypoint was executed
#: open loop, by ramping the position servos from the previous one with no
#: policy and no feedback. Six of the nine legs are verified. The one that is
#: not, and cannot be, is `rock_back` — and that is the whole result of phase
#: 3c: the learning problem is ONE dynamic transition, not a six-phase motion.
WAYPOINTS = (
    Waypoint(
        "prone", 1.57,
        dict(hip_pitch=0.0, knee=0.10, ankle_pitch=0.0,
             shoulder_pitch=-0.25, elbow=0.10),
        dwell_s=0.3, equilibrium=True, verified=True,
        settled_z=0.007, settled_u=0.01,
        contacts=("ankle_roll", "neck_pitch", "waist_yaw"), torque_frac=0.26,
        note="Face down, flat. THE START, and the reason is measured: from "
             "SUPINE this robot cannot rise at all. Three scripted routes off "
             "its back — forearms overhead then push, elbows under and tuck, "
             "knees over the head first — every one ends at pelvis 0.097 m "
             "with the hips on the floor, and slowing the reference 8x changes "
             "nothing. The hip servos cannot lift the trunk from horizontal: "
             "the sit-up pose alone demands 92 % of rated torque just to be "
             "approached. A supine get-up therefore needs a ROLL to prone "
             "first, which is a root rotation about the long axis and a "
             "separate problem (phase 3d).",
    ),
    Waypoint(
        "plant_forearms", 1.57,
        dict(hip_pitch=0.0, knee=0.10, ankle_pitch=0.0,
             shoulder_pitch=-1.20, elbow=1.90),
        dwell_s=0.0, equilibrium=False, verified=True,
        settled_z=0.020, settled_u=-0.02,
        contacts=("ankle_roll", "elbow", "neck_pitch"), torque_frac=0.69,
        note="TRANSIT. The forearms come in under the shoulders. The pelvis "
             "DROPS here, 0.135 -> 0.014 m, which is the move working: the "
             "robot is loading its elbows, not lifting. Any reward paying "
             "height per step charges this phase for happening.",
    ),
    Waypoint(
        "draw_knees_in", 1.50,
        dict(hip_pitch=-0.80, knee=1.20, ankle_pitch=-0.35,
             shoulder_pitch=-1.40, elbow=1.60),
        dwell_s=0.0, equilibrium=False, verified=True,
        settled_z=0.120, settled_u=-0.20,
        contacts=("elbow", "knee"), torque_frac=0.61,
        note="TRANSIT. Knees come under the hips while the forearms hold the "
             "front. The four-contact base forms here.",
    ),
    Waypoint(
        "quadruped", 1.45,
        dict(hip_pitch=-1.45, knee=1.50, ankle_pitch=-0.35,
             shoulder_pitch=-1.40, elbow=1.40),
        dwell_s=0.3, equilibrium=True, verified=True,
        settled_z=0.211, settled_u=-0.06,
        contacts=("ankle_roll", "elbow", "knee"), torque_frac=0.58,
        note="Face down on four contacts — the knee capsules and the forearms. "
             "u_z -0.06: the trunk is HORIZONTAL, and this is the phase every "
             "uprightness reward was paying the robot to skip. 58 % of rated "
             "elbow torque.",
    ),
    Waypoint(
        "pike", 1.20,
        dict(hip_pitch=-0.90, knee=0.05, ankle_pitch=-0.35,
             shoulder_pitch=-1.30, elbow=1.60),
        dwell_s=0.2, equilibrium=True, verified=True,
        settled_z=0.300, settled_u=-0.37,
        contacts=("ankle_roll", "elbow"), torque_frac=0.66,
        note="Knees off the floor, hips high, weight split between feet and "
             "forearms. The LAST pose that holds before the rock-back, the "
             "furthest open-loop playback reaches, and the most expensive at "
             "66 % of rated elbow torque.",
    ),
    Waypoint(
        "rock_back", 1.523,
        # ankle clamped from the recorded -0.351: the physics solver permits a
        # hair past a stop, a reference must not ask for it.
        dict(hip_pitch=-1.144, knee=1.012, ankle_pitch=-0.350,
             shoulder_pitch=-0.796, elbow=1.389),
        dwell_s=0.0, equilibrium=False, verified=False,
        settled_z=0.377, settled_u=0.048,
        contacts=("ankle_roll",), torque_frac=0.36,
        note="**THE ONE MOVE THAT HAS TO BE LEARNED**, and this waypoint is "
             "RECORDED rather than authored. Nothing between pelvis 0.300 and "
             "0.517 m is an equilibrium, so there is no pose to place and "
             "settle — but the REVERSE is free: shove `fold` forward and it "
             "falls onto its forearms, crossing the whole gap in 0.12 s **on "
             "its feet alone**. This is the state at the middle of that "
             "crossing: a jack-knife, trunk folded horizontal (root pitch "
             "1.52, u +0.05) over planted feet at pelvis 0.377 m. Every number "
             "here is off that trajectory, so it is a configuration the body "
             "demonstrably passes through.\n\n"
             "The authored version of this waypoint recorded pelvis 0.170 m — "
             "BELOW the `pike` it follows — because its numbers were taken "
             "from where a training chain happened to collapse. Harmless while "
             "tracking was joint-only; as a body target it paid the robot to "
             "fall.\n\n"
             "What makes this leg hard is now measured and it is not torque: "
             "the fallen state and the standing `fold` have **the same joint "
             "angles** (mean difference 0.0516 rad) and completely different "
             "bodies (0.197 m / u -0.35 against 0.516 m / u +0.84). The joints "
             "do not determine the pose, the CONTACT MODE does — which is why "
             "5x rated torque produced a bit-identical result, and why the "
             "rise needs the arms to PUSH OFF rather than the legs to lift.",
    ),
    Waypoint(
        "fold", 0.60,
        dict(hip_pitch=-1.40, knee=1.00, ankle_pitch=-0.35,
             shoulder_pitch=-0.80, elbow=1.40),
        dwell_s=0.2, equilibrium=True, verified=False,
        settled_z=0.517, settled_u=0.83,
        contacts=("ankle_roll",), torque_frac=0.36,
        note="The deepest pose this robot holds ON ITS FEET ALONE, 81 % of "
             "standing — and only because the arms reach FORWARD to carry the "
             "centre of mass over a footprint the +-0.35 rad ankle cannot "
             "reach under. Arms at the default it manages 0.545 m at best. "
             "This is the pose that closes the rock-back.",
    ),
    Waypoint(
        "crouch", 0.30,
        dict(hip_pitch=-0.75, knee=0.80, ankle_pitch=-0.35),
        dwell_s=0.2, equilibrium=True, verified=False,
        settled_z=0.562, settled_u=0.95,
        contacts=("ankle_roll",), torque_frac=0.16,
        note="Arms back at the default, trunk nearly upright. A knee of 1.00 "
             "at this pitch sits at 0.511 m for two full seconds and collapses "
             "at three — which is why the window is 6 s and this row is knee "
             "0.80. The 2 s measurement that called 0.514 m the deepest hold "
             "was reporting a slow collapse.",
    ),
    Waypoint(
        "stand", 0.00, {},
        dwell_s=1.0, equilibrium=True, verified=False,
        settled_z=0.608, settled_u=1.00,
        contacts=("ankle_roll",), torque_frac=0.14,
        note="DEFAULT_POSE. Holds open loop with no policy at all (6/6, "
             "2026-10-04) — which is why 'it cannot balance' was the wrong "
             "diagnosis of run 66ee20, and a donor the wrong next lever.",
    ),
)

BY_NAME = {w.name: w for w in WAYPOINTS}

#: The last waypoint open-loop playback reaches unaided. Everything up to here
#: is a VERIFIED reference; everything after it is authored and waiting on a
#: policy. Phase 3c's deliverable is this being `pike` and not `prone`.
#:
#: Read it precisely: the chain TRACKS the authored reference this far, and
#: the break is the leg `pike` -> `rock_back`, where the servos reach 0.209 m
#: against a RECORDED 0.377 m. **That one leg is the learning problem**, and
#: everything before it is a motion this robot has already been shown to make.
#: It was `rock_back` until the gap waypoint stopped being an invention: the
#: authored one sat at pelvis 0.170 m, below the pike, so "reaching" it was
#: falling.
VERIFIED_THROUGH = "pike"

#: Seconds to travel between consecutive waypoints — MEASURED, not assumed.
#:
#: The first four are the ramp lengths the open-loop chain uses, and the servos
#: execute them. The last four are a measured window, and getting them wrong is
#: why even the TAIL — whose every pose holds for 6 s — never stood:
#:
#: | `fold` -> `crouch` | `crouch` -> `stand` | result |
#: |---|---|---|
#: | 0.20-0.30 s | any of 0.20-0.80 s | **STANDS**, 0.608 m, u +1.00, held 4 s |
#: | 0.40 s | 0.40 s or slower | stands |
#: | 0.50 s or slower | anything | falls |
#:
#: The reference shipped 0.60 s and 1.00 s — outside the window entirely. The
#: move is momentum-dependent: too slow and the robot settles at each step
#: instead of carrying through, and sits down. Nothing about the POSES was
#: wrong, and six training chains ran against a tail that could not finish.
#:
#: `pike` -> `rock_back` is 0.30 s and not 0.16: the knee travels 0.96 rad on
#: that leg, and 0.16 s asks for 0.18 rad per control step, which is a command
#: no servo tracks. The timing there is a guess either way — no speed from
#: 0.08 s to 0.60 s crosses it open loop — so it is set to what the reference
#: can actually ask for.
LEGS = (0.8, 0.8, 0.8, 0.8, 0.30, 0.12, 0.25, 0.30)
assert len(LEGS) == len(WAYPOINTS) - 1


def duration() -> float:
    """Total reference length, dwells included."""
    return float(sum(LEGS) + sum(w.dwell_s for w in WAYPOINTS))


def _timeline() -> list[tuple[float, float, int, int]]:
    """(t0, t1, from_index, to_index) spans; a dwell has from == to."""
    out, t = [], 0.0
    for i, w in enumerate(WAYPOINTS):
        if w.dwell_s > 0.0:
            out.append((t, t + w.dwell_s, i, i))
            t += w.dwell_s
        if i < len(LEGS):
            out.append((t, t + LEGS[i], i, i + 1))
            t += LEGS[i]
    return out


def pose_at(t: float) -> tuple[np.ndarray, float, str]:
    """The reference at time `t`: `(joints, root_pitch, phase_name)`.

    Interpolated linearly in joint space and in root pitch, with a smoothstep
    on the leg fraction so the reference has no velocity discontinuity at a
    waypoint — an imitation reward differentiates this, and a corner in it
    reads as an impossible acceleration the policy is charged for missing.
    """
    spans = _timeline()
    total = spans[-1][1]
    t = float(np.clip(t, 0.0, total))
    for t0, t1, i, j in spans:
        if t <= t1 or (t0, t1, i, j) == spans[-1]:
            a, b = WAYPOINTS[i], WAYPOINTS[j]
            if i == j or t1 <= t0:
                return a.joints, a.root_pitch, a.name
            f = (t - t0) / (t1 - t0)
            f = f * f * (3.0 - 2.0 * f)              # smoothstep
            qa, qb = a.joints, b.joints
            return (qa + (qb - qa) * f,
                    a.root_pitch + (b.root_pitch - a.root_pitch) * f,
                    b.name if f > 0.5 else a.name)
    raise AssertionError("unreachable: the timeline covers [0, total]")


def phase_index(t: float) -> int:
    """Which waypoint the reference is heading for at `t` — the phase clock a
    phase-indexed reward advances on achievement rather than on time."""
    name = pose_at(t)[2]
    return next(i for i, w in enumerate(WAYPOINTS) if w.name == name)


# --------------------------------------------------------------- the RUNGS
#
# THE MEASUREMENT THIS EXISTS FOR (2026-10-06). Three reverse curricula for
# this get-up have failed, and all three built their rungs by INTERPOLATING
# between a fallen pose and the standing pose. An average of two poses is not a
# pose this robot can hold, so none of those rungs was a rung. Searched
# properly — place a pose, command it, and require the trunk to still be there
# 6 s later — the stable set is in two disconnected pieces:
#
#   | trunk height   | what holds               | n   |
#   |----------------|--------------------------|-----|
#   | 0.10 - 0.20 m  | nothing                  |  0  |
#   | 0.20 - 0.40 m  | ARM-SUPPORTED (push-up)  | 95  |
#   | 0.40 - 0.50 m  | nothing                  |  0  |
#   | 0.50 - 0.65 m  | feet-only crouch         |  5  |
#
# (The feet-only sweep here is coarser than the 945-point one that first found
# that plateau, which returned 7 rather than 5 — same two-piece structure, same
# bounds to 3 mm, fewer samples inside. The grid is in `_search_rungs`; widen
# it there rather than quoting a count this code does not produce.)
#
# (floor 0.11 m, standing 0.639 m.)
#
# The arm-supported plateau had never been searched: the only previous sweep
# required FEET-ONLY contact and held the arms fixed, so by construction it
# could only find standing crouches and reported that nothing holds below
# 0.516 m. The push-up position does hold, all the way down to 0.212 m.
#
# What that explains:
#
#   * The 15 cm gap from 0.365 to 0.516 is BALLISTIC. No reverse curriculum can
#     put a rung in it, which is why all three ladder designs died there.
#   * A from-scratch PPO run reaches **0.395 m** and stalls — the top of the
#     arm plateau, one step into the dead gap. It climbs until there is nothing
#     left to stand on.
#   * The one get-up known to work crosses the gap exactly as a launch: between
#     0.35 and 0.50 of standing its arms carry **0.594** of body weight and the
#     trunk rises at **1.0 m/s**, and above 0.5 the arm load falls to 0.037 and
#     the legs finish it.
#
# So the task is two rungs with real pose sets, and ONE unlearned transition
# between them.

HOLDABLE_HOLD_S = 6.0
#: Settle-then-hold tolerance: the trunk must not move more than this between
#: 1.2 s and `HOLDABLE_HOLD_S`. A pose that sits still for two seconds and
#: collapses at three is not a rung (`add-a-robot` SKILL.md §4c).
HOLDABLE_DRIFT_M = 0.03


def rungs_path(dest=None):
    from . import asimov as _asimov

    return _P(dest or _asimov.CACHE_DIR) / "holdable_rungs.npz"


def holdable_rungs(dest=None, rebuild: bool = False):
    """Poses this robot can actually HOLD, as `(joints, root_pitch, trunk_z,
    support)` where `joints` is the full SETTLED `qpos` — `support` is "arm" or "feet".

    Written atomically, like the fallen-pose cache, because a training run
    builds many workers and any of them may be the one that finds the file
    missing.
    """
    import os

    import numpy as _np

    path = rungs_path(dest)
    if rebuild or not path.is_file():
        joints, pitch, zs, sup = _search_rungs()
        path.parent.mkdir(parents=True, exist_ok=True)
        # NOT ".npz.tmp": `np.savez` appends ".npz" to any name that does not
        # already end in it, so that writes `...npz.tmp.npz` and the rename
        # fails on a file that was never there.
        tmp = path.with_name(path.stem + ".tmp.npz")
        _np.savez(tmp, joints=joints, root_pitch=pitch, trunk_z=zs,
                  support=_np.array(sup))
        os.replace(tmp, path)
    return _np.load(path, allow_pickle=True)


def _search_rungs():
    """The two sweeps. Kept here so the table above is reproducible, not a
    number somebody once printed into a chat window."""
    import itertools

    import mujoco

    from . import spec as S

    env = S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, seed=0, action_clip=13.0)
    m, d = env.model, env.data
    trunk = env.trunk_body_id
    scale = float(np.atleast_1d(env.robot.action_scale)[0])
    ARMB = ("shoulder", "elbow", "wrist")

    def hold(q, pitch, steps):
        env._place(np.asarray(q), float(pitch))
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        act = np.clip((np.asarray(q) - np.asarray(env.robot.default_pose))
                      / scale, -13.0, 13.0).astype(np.float32)
        zs = []
        for _ in range(steps):
            d.xfrc_applied[:] = 0.0
            env._write_ctrl(env.robot.scale_action(act))
            for _ in range(4):
                mujoco.mj_step(m, d)
            zs.append(float(d.xpos[trunk][2]))
        settled = d.qpos.copy()
        arm = feet = body = False
        for c in range(int(d.ncon)):
            for g in (d.contact[c].geom1, d.contact[c].geom2):
                nm = m.body(m.geom_bodyid[g]).name
                if any(x in nm for x in ARMB):
                    arm = True
                elif "ankle_roll" in nm:
                    feet = True
                elif nm != "world":
                    # Pelvis, waist, thigh, knee, head — the robot RESTING on
                    # itself. Not a rung: a pose held up by its own torso is
                    # lying down. This was missing, and the classifier called
                    # anything with feet down and no arms "feet-only", so three
                    # poses sitting on their PELVIS at trunk 0.094 m were
                    # catalogued as feet-only equilibria below the crouch
                    # plateau.
                    body = True
        return (zs, float(-env._projected_gravity()[2]), arm, feet, body,
                settled)

    settle = int(1.2 * 50)
    full = int(HOLDABLE_HOLD_S * 50)
    out = []

    # ARM-SUPPORTED. The grid is wide and there is NO height threshold, and
    # both are CORRECTIONS. The first version required `trunk > 0.15` at the
    # 1.2 s mark, which discarded every low pose by construction and produced a
    # table claiming nothing holds between the floor and 0.20 m. Searched
    # without it, 97 poses hold from **0.120 m** — essentially on the ground.
    #
    # Third time on this task that a "nothing exists" came from a grid that
    # could not express the answer: root pitch pinned at 0, then feet-only
    # contact required, then this. When a sweep reports an empty region the
    # first suspect is the sweep.
    for sh, el, hip, knee, pitch in itertools.product(
            (-0.4, -1.2, -2.0, -2.8), (0.2, 0.7, 1.2, 1.7),
            (-0.4, -1.0, -1.8, -2.4), (0.3, 0.9, 1.5, 2.0),
            (0.0, 0.2, 0.5, 0.9, 1.2)):
        q = joints(hip_pitch=hip, knee=knee, ankle_pitch=-0.35,
                   shoulder_pitch=sh, elbow=el)
        zs, _u, arm, _f, body, _q = hold(q, pitch, settle)
        if not (arm and not body and zs[-1] > 0.115):
            continue
        zs, _u, arm, feet, body, qs = hold(q, pitch, full)
        if (arm and not body
                and abs(zs[-1] - zs[settle - 1]) < HOLDABLE_DRIFT_M
                # ...and it did not COLLAPSE before the drift window opened. A
                # pose that fell in the first 1.2 s and then lay still passed
                # the drift test: one claimed rung at 0.456 m dropped to 0.097.
                and zs[-1] > 0.6 * max(zs[:5])):
            out.append((qs, pitch, zs[-1], "arm"))

    # FEET-ONLY: the crouch plateau. Root pitch IS in the grid — fixing it at 0
    # is what made an earlier sweep report that only standing holds.
    for hip, knee, ank, pitch in itertools.product(
            (-0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4, -1.6),
            (0.25, 0.5, 0.75, 1.0, 1.25), (-0.35, -0.2, 0.0),
            (0.02, 0.15, 0.25, 0.35, 0.47, 0.60, 0.70)):
        q = joints(hip_pitch=hip, knee=knee, ankle_pitch=ank,
                   shoulder_pitch=-0.4, elbow=0.6)
        zs, u, _a, feet, body, _q = hold(q, pitch, settle)
        if not (feet and not body and u > 0.5):
            continue
        zs, u, _a, feet, body, qs = hold(q, pitch, full)
        if (feet and not body and u > 0.5
                and abs(zs[-1] - zs[settle - 1]) < HOLDABLE_DRIFT_M
                and zs[-1] > 0.6 * max(zs[:5])):
            out.append((qs, pitch, zs[-1], "feet"))

    # THE FINAL FILTER, and the only one that tests what the env actually
    # does. Everything above places a joint CONFIGURATION and asks whether it
    # settles somewhere stable. `AsimovGetupEnv.reset` instead writes the
    # SETTLED state in and the policy commands from there — so a rung has to
    # hold when replayed that way, which is a different question.
    #
    # It matters: of 55 rungs that passed the search, **11 failed this replay**
    # by more than 3 cm, including poses that collapsed 36 cm. A curriculum
    # built on rungs that do not hold is precisely what made the three
    # interpolated ladders useless, so the cache carries only rungs that
    # survive their own replay.
    validated = []
    for qs, pitch, zz, kind in out:
        d.qpos[:] = np.asarray(qs)
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        a = np.clip((d.qpos[env._q_adr] - np.asarray(env.robot.default_pose))
                    / scale, -13.0, 13.0).astype(np.float32)
        z0 = float(d.xpos[trunk][2])
        for _ in range(int(HOLDABLE_HOLD_S * 50)):
            d.xfrc_applied[:] = 0.0
            env._write_ctrl(env.robot.scale_action(a))
            for _ in range(4):
                mujoco.mj_step(m, d)
        if abs(float(d.xpos[trunk][2]) - z0) <= HOLDABLE_DRIFT_M:
            validated.append((qs, pitch, float(d.xpos[trunk][2]), kind))
    out = validated

    out.sort(key=lambda r: r[2])
    # `qpos` is the SETTLED state after the full hold, not the commanded joint
    # configuration. Those are 25 cm apart: placing the lowest arm pose puts the
    # trunk at 0.462 m and it sinks to 0.212 m over the first second. An episode
    # that starts from the configuration starts with a fall already in progress,
    # which is precisely the defect that made three interpolated ladders
    # useless. A rung has to BE the equilibrium, not aim at it.
    return (np.array([r[0] for r in out]), np.array([r[1] for r in out]),
            np.array([r[2] for r in out]), [r[3] for r in out])
