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

#: EVERY left/right joint pair on this robot takes OPPOSITE values for one
#: symmetric pose, so `joints()` negates the right side of all of them.
#:
#: That is measured, not assumed, and it is the only form of the rule that has
#: survived: mirror each left link's world orientation across the sagittal
#: plane and compare it to its right counterpart, and `l = +v, r = -v` gives a
#: residual of **exactly 0.000000** on all eleven stems while `l = r = +v`
#: gives 1.2-5.6. `test_MIRRORED_is_complete` re-derives this from the model
#: and fails if a stem is missing, because the tuple was wrong twice:
#:
#:   * `hip_roll` and `hip_yaw` were missing until 2026-10-06, and the cost was
#:     a whole plateau. `joints(hip_roll=0.7)` LEANED the robot — stance width
#:     unchanged at 0.215 m, midline shifted +0.214 — where it reads as a
#:     splay. Because `_search_rungs` builds every pose through this helper and
#:     never passed these stems, both sweeps searched a NARROW stance only,
#:     where the support set is a LINE, and the feet-only plateau was reported
#:     to bottom out at 0.471 m. With the splay it reaches 0.399 m.
#:   * `shoulder_roll`, `shoulder_yaw`, `ankle_roll` and `wrist_yaw` were found
#:     missing in the same audit. `shoulder_roll` is the one that was provably
#:     wrong rather than merely unused: `DEFAULT_POSE` holds it at -0.050 left
#:     against +0.050 right, so `joints(shoulder_roll=-0.05)` moved the RIGHT
#:     shoulder — breaking the "passing a joint its own default is a no-op"
#:     property that makes a pose draft readable.
#:
#: Why the two groups look different in the model: five stems have
#: anti-parallel axes (left +Y, right -Y) and six share an axis. Both end up
#: negating, which is exactly why reasoning from the axis was the wrong route
#: and the mirror residual is the right one.
#:
#: Signs, measured, so nobody re-derives them: positive `hip_roll` splays the
#: legs APART; positive `hip_yaw` turns the toes IN.
MIRRORED = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch",
            "ankle_roll", "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
            "elbow", "wrist_yaw")

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

#: Foot-to-foot span, metres, past which a pose is not a rung and an episode
#: is over (`AsimovGetupEnv.max_span`). ONE constant, because a cut tighter
#: than the cache would terminate every episode on its own start pose — the
#: same shape of bug as a cache built at a clip nothing trains at.
#:
#: 0.70 is the measured knee, not a preference. Restricting the ladder by span
#: costs NO height coverage down to 0.70 and starts costing immediately below:
#:
#:   span <= 0.60  arm 0.136-0.206, feet 0.425-0.606  -> gap 220 mm
#:   span <= 0.70  arm 0.136-0.280, feet 0.416-0.606  -> gap 136 mm
#:   span <= 1.10  arm 0.134-0.280, feet 0.416-0.606  -> gap 136 mm
#:
#: So the widest 49 rungs add nothing but width. What they DO enable is the
#: splits: teach-asimov_getup_clip-79ef4d rises 288/288 and gets there by
#: sprawling its feet a median 0.989 m apart on a 1.2 m robot whose standing
#: stance is 0.215 m — the break-dancing route ruled out at the start of this
#: track as dangerous on hardware.
MAX_FOOT_SPAN_M = 0.70

#: The EPISODE cut, which is a different number with a different job.
#:
#: `MAX_FOOT_SPAN_M` filters the rung CACHE and is the measured knee for
#: height coverage. The episode cut has to admit everything the staged route
#: actually passes through, and the scripted ramp overshoots: its rungs are
#: all within 0.70 but it ARRIVES at 0.705-0.724 m, because the end state is
#: a dynamic result and not one of the rungs. Set equal to the rung filter,
#: every one of the 24 handover states was over the cut and the episode ended
#: on its own start pose — a whole training run void, and the same shape of
#: fault as a rung cache built at a clip nothing trains at.
#:
#: 0.75 admits the ladder (<=0.70) and the ramp (<=0.724) with margin, and
#: still forbids what it exists to forbid: the teacher's own demonstrations
#: sprawl a median 0.929 m and the policy cloned from them 0.989 m, on a
#: 1.2 m robot whose standing stance is 0.215 m.
EPISODE_SPAN_CUT_M = 0.75

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


#: What each of the vendor's links ACTUALLY IS, keyed by the joint it hangs
#: from — because an MJCF link is named after the joint ABOVE it, so
#: `knee_link` is the segment BELOW the knee: the shin.
#:
#: This table exists because reading those names by eye produced a false
#: finding on 2026-10-09. A grep of body names for "shin"/"calf" returned
#: nothing, which was reported as "this robot has no shin collision geom" and
#: used to argue a half-kneel was impossible. The shin is there — a 90 mm x
#: 40 mm capsule on `left_knee_link` — and so are the thighs. Never infer a
#: part from a vendor's link name; ask `part_of`, which derives it from the
#: kinematic chain and is pinned by
#: `test_the_vendor_link_names_mean_what_the_CHAIN_says`.
_PART_BY_JOINT = {
    "floating_base": "pelvis",
    "hip_pitch": "upper thigh", "hip_roll": "thigh", "hip_yaw": "thigh",
    "knee": "SHIN",
    "ankle_pitch": "ankle", "ankle_roll": "foot",
    "waist_yaw": "torso",
    "shoulder_pitch": "shoulder", "shoulder_roll": "upper arm",
    "shoulder_yaw": "upper arm",
    "elbow": "forearm", "wrist_yaw": "hand",
    "neck_pitch": "head", "neck_yaw": "head",
}


def part_of(model, body) -> str:
    """'left SHIN' for `left_knee_link` — the part, not the vendor's label.

    Derived from the chain: a body is named for the joint that attaches it to
    its parent, so the part it represents is the segment between that joint
    and its own child joint.
    """
    name = body if isinstance(body, str) else model.body(int(body)).name
    stem = name[:-5] if name.endswith("_link") else name
    side = ""
    for pre, lbl in (("left_", "left "), ("right_", "right ")):
        if stem.startswith(pre):
            side, stem = lbl, stem[len(pre):]
            break
    return side + _PART_BY_JOINT.get(stem.replace("_joint", ""), stem)


def floor_parts(env) -> list:
    """What is touching the floor, in PARTS — 'left SHIN', not 'knee_link'."""
    m, d = env.model, env.data
    out = set()
    for c in range(int(d.ncon)):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if m.geom_bodyid[g1] == 0 or m.geom_bodyid[g2] == 0:
            b = int(m.geom_bodyid[g2] if m.geom_bodyid[g1] == 0
                    else m.geom_bodyid[g1])
            out.add(part_of(m, b))
    return sorted(out)


#: Fraction of standing trunk height that counts as up.
STAND_FRAC = 0.86
#: Uprightness (the projected-gravity cosine) that counts as up.
STAND_UPRIGHT = 0.85


def support_margin(env) -> float | None:
    """Signed horizontal distance from the CoM to the edge of the SUPPORT
    POLYGON, in metres. Positive = inside (balanced), negative = outside
    (falling over, however vertical it looks). `None` = fewer than three
    floor contacts, so there is no polygon to be inside.

    THE CLAUSE A STAND WAS MISSING. `standing_report` asks for the trunk up,
    the robot upright and nothing but feet down, and a robot toppling about
    its ankles satisfies all three for a few hundred milliseconds on the way
    over. Measured on `teach-asimov_getup_turn-380ec2`, the first policy to
    rise without hopping: while it is "standing" by that test, the CoM is
    OUTSIDE this polygon **74 % of the time**, median margin **-22 mm**, p10
    **-131 mm**. Every lost stand has the same trace — uprightness holds at
    +0.98 while the gap from the feet grows 200 -> 320 mm and the trunk
    slides down. It was never balanced; it passed through vertical.

    Built from MuJoCo's own contact points rather than from geom positions,
    for the reason `settle_onto_floor` exists: a pad hovering 2 mm clear is
    not support, and no bound on the geometry can tell you that.
    """
    import mujoco as _mj
    import numpy as _np

    m, d = env.model, env.data
    pts = _np.array([d.contact[c].pos[:2] for c in range(int(d.ncon))
                     if m.geom_bodyid[d.contact[c].geom1] == 0
                     or m.geom_bodyid[d.contact[c].geom2] == 0])
    if len(pts) < 3:
        return None
    _mj.mj_comPos(m, d)
    com = _np.asarray(d.subtree_com[0], float)[:2]

    def _cross(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    order = pts[_np.lexsort((pts[:, 1], pts[:, 0]))]

    def _half(q):
        h: list = []
        for x in q:
            while len(h) > 1 and _cross(h[-1] - h[-2], x - h[-2]) <= 1e-12:
                h.pop()
            h.append(x)
        return h

    hull = _np.array(_half(order)[:-1] + _half(order[::-1])[:-1])
    if len(hull) < 3:
        return None
    inside, best = True, 1e9
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        e = b - a
        n = _np.array([-e[1], e[0]]) / (float(_np.linalg.norm(e)) + 1e-12)
        dist = float(_np.dot(com - a, n))
        if dist < 0.0:
            inside = False
        best = min(best, abs(dist))
    return best if inside else -best


def standing_report(env) -> dict:
    """Is the robot STANDING — and if not, which clause failed?

    THE ONE PLACE THIS IS DECIDED. Ten scripts had their own copy of the
    test on 2026-10-08 and they did not agree: some asked whether the PEAK
    trunk height over the episode cleared the bar, which is not a stand and
    twice reported one that was not there.

      * At `fold` the episode STARTS at 0.523 m and the bar is 0.86 x 0.608 =
        0.523 m, so every episode passed before it did anything. A whole
        curriculum was designed on a "free rise" of 23/24 that was the
        starting pose being counted.
      * A trajectory search maximising peak height returned a "narrow
        crossing" at 0.615 m that was the robot face down on its elbows with
        its hips thrown in the air — uprightness **-0.61**, four contacts.

    So height alone is never enough. A stand is the trunk up, the robot
    upright, AND nothing but feet on the floor — knees and forearms are how
    this body holds a quadruped at 0.211 m and an arm plateau at 0.59 m.
    Evaluate it on the pose that SURVIVES, never on a maximum.
    """

    m, d = env.model, env.data
    z = float(d.xpos[env.trunk_body_id][2])
    upright = float(-env._projected_gravity()[2])
    floor = set()
    for c in range(int(d.ncon)):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if m.geom_bodyid[g1] == 0 or m.geom_bodyid[g2] == 0:
            b = int(m.geom_bodyid[g2] if m.geom_bodyid[g1] == 0
                    else m.geom_bodyid[g1])
            floor.add(m.body(b).name)
    feet = {m.body(int(m.geom_bodyid[env.foot_geoms[s]])).name
            for s in ("left", "right")}
    on_feet = bool(floor & feet)
    only_feet = bool(floor) and floor <= feet
    bar = STAND_FRAC * BY_NAME["stand"].settled_z
    ok = (z > bar) and (upright > STAND_UPRIGHT) and on_feet and only_feet
    margin = support_margin(env)
    return {"standing": bool(ok), "trunk_z": z, "bar": float(bar),
            # REPORTED, not required. Making it a clause of `standing` would
            # silently rewrite every number this repo has published, so both
            # are carried and a scorer says which one it means.
            "com_margin": margin,
            "balanced": bool(ok and margin is not None and margin > 0.0),
            "floor_parts": sorted(part_of(m, b) for b in floor),
            "upright": upright, "on_feet": on_feet, "only_feet": only_feet,
            "floor_contacts": sorted(floor),
            "why": ("" if ok else
                    "; ".join(x for x in (
                        f"trunk {z:.3f} <= {bar:.3f}" if z <= bar else "",
                        f"upright {upright:+.2f} <= {STAND_UPRIGHT}"
                        if upright <= STAND_UPRIGHT else "",
                        "no foot on the floor" if not on_feet else "",
                        f"not only feet: {sorted(floor - feet)}"
                        if floor and not only_feet else "") if x))}


def is_standing(env) -> bool:
    """`standing_report(env)["standing"]` — see it for why height is not it."""
    return bool(standing_report(env)["standing"])


#: THE SEAT — the pose the authored route never had, and the teacher's own.
#:
#: Measured 2026-10-09. The authored push-up ends face DOWN (uprightness
#: -0.37 at `pike`), and from there hip flexion swings a leg into the air —
#: 751-1095 mm up — because the pelvis is inverted, so the feet can never come
#: under the mass. The teacher does the opposite: it reaches uprightness
#: **+0.49 at 0.06-0.12 m**, chest up while still low, and rises already
#: upright. This is that pose for this robot: a long sit, legs straight out,
#: resting on the backs of the thighs.
#:
#:    trunk 0.106 m   uprightness +0.64   foot span 0.215 m   on both thighs
#:
#: It does NOT hold open loop — 0.02-0.04 s before it rolls back, because the
#: CoM sits behind the thigh contact. That is not a verdict: "open-loop holds
#: topple" is on record for the duck and the G1 in poses a trained policy
#: holds. It is a start state to TRAIN from, not a rung to stand on.
SEAT_JOINTS = dict(hip_pitch=-1.553, knee=0.087, ankle_pitch=0.349,
                   shoulder_pitch=-2.094, elbow=0.349)
SEAT_ROOT_PITCH = 0.600


def seat_pose():
    """(qpos joints, root_pitch) for the long sit. See `SEAT_JOINTS`."""
    return joints(**SEAT_JOINTS), SEAT_ROOT_PITCH


def seat_states(n: int = 16, seed: int = 0):
    """A spread of seats, jittered, for a curriculum to reset onto.

    Each is PLACED and settled a few steps so the thigh contact is real
    rather than a pose hanging in the air — the same reason the kneel search
    had to settle before counting contacts.
    """
    import mujoco as _mj

    from . import spec as _S

    # start_rung="" EXPLICITLY. The ctor falls back to the environment, and
    # the trainer sets MICRODUCK_ASIMOV_GETUP_START_RUNG=seat — so without
    # this the env built here asks for seat states, which builds an env, which
    # asks for seat states: RecursionError, and the run dies with only a
    # train.log. Every `*_states` builder has to pin the rung it is building.
    env = _S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, start_rung="",
        action_clip=float(_S.get("asimov1").action_clip))
    m, d = env.model, env.data
    sc = float(np.atleast_1d(env.robot.action_scale)[0])
    default = np.asarray(env.robot.default_pose, float)
    clip = float(_S.get("asimov1").action_clip)
    rng = np.random.default_rng(seed)
    qs, vs = [], []
    for i in range(n):
        kw = dict(SEAT_JOINTS)
        if i:
            kw = {k: v + float(rng.normal(0.0, 0.03)) for k, v in kw.items()}
        q = joints(**kw)
        # CLAMP to the model's own stops: the jitter pushed the ankle 18 mrad
        # past a +-0.350 limit and `_place` refused it, which is the guard
        # doing its job — a jittered pose is still a pose.
        adr = getattr(env, "_q_adr", env.joint_qpos_adr)
        for k, a_ in enumerate(np.atleast_1d(adr)):
            j = int(np.argmax(m.jnt_qposadr == a_))
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            if hi > lo:
                q[k] = min(max(q[k], lo), hi)
        env._place(q, SEAT_ROOT_PITCH + (0.0 if not i else rng.normal(0, 0.04)))
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        _mj.mj_forward(m, d)
        a = np.clip((q - default) / sc, -clip, clip).astype(np.float32)
        # SETTLE UNTIL IT IS ACTUALLY AT REST, and verify it. Eight steps left
        # the root moving at 0.28 m/s, so the "start state" was the robot
        # mid-fall; the policy trained on it reached trunk 0.633 at +0.87
        # uprightness with NO FOOT ON THE FLOOR — airborne, not standing.
        # A start state is settled, in contact, and still.
        for _ in range(200):
            d.xfrc_applied[:] = 0.0
            env.step(a)
            if float(np.linalg.norm(d.qvel[:6])) < 0.05 and int(d.ncon) > 0:
                break
        if float(np.linalg.norm(d.qvel[:6])) >= 0.08 or int(d.ncon) == 0:
            continue                      # never came to rest: not a state
        # and it has to be a SEAT: on the thighs or pelvis with the chest up.
        # Jittered variants collapse onto the head or the torso, which is a
        # different pose entirely and would be trained on as if it were this
        # one. Uprightness +0.4 is the teacher's own signature band (+0.49 at
        # 0.06-0.12 m); below it the robot is simply lying down.
        parts = floor_parts(env)
        seated = any(("thigh" in q_ or "pelvis" in q_) for q_ in parts)
        spoiled = any(("head" in q_ or "torso" in q_) for q_ in parts)
        if not seated or spoiled:
            continue
        if float(-env._projected_gravity()[2]) < 0.40:
            continue
        qs.append(d.qpos.copy())
        vs.append(d.qvel.copy())
    if not qs:
        raise RuntimeError("no seat came to rest in contact")
    return np.array(qs), np.array(vs)


def handover_path(dest=None):
    from . import asimov as _asimov
    return _P(dest or _asimov.CACHE_DIR) / "handover_states.npz"


def handover_states(dest=None, rebuild: bool = False):
    """Where the SCRIPTED arm-plateau rise hands over to a policy.

    Phase 3i measured that the arm plateau traverses OPEN LOOP -- 0.136 ->
    0.280 m, 7 legs, position servos only, no policy, at every tempo. So the
    bottom of the get-up does not have to be learned, it has to be PLAYED, and
    the learning problem shrinks from a whole get-up to the one transition
    above it.

    Each entry is `(qpos, qvel)`, and the qvel is the point. A curriculum that
    resets onto a settled rung hands the policy a robot AT REST, and the gap
    above the arm plateau is ballistic -- the one get-up known to work crosses
    it with the trunk already rising at 1.0 m/s. `asimov_getup_floor` stage 2
    starts from settled `arm_top` rungs and rises **0/24**. Played instead, the
    ramp arrives at 0.253 m already moving at +0.139 m/s, on forearms and feet.

    A SET, not one state, because a rung is a set of starts: training each rung
    on its single newest start is what once collapsed this robot's hold from
    7.76 s to 0.24 s while every per-rung score still read 12/12.
    """
    import os

    import numpy as _np

    path = handover_path(dest)
    if rebuild or not path.is_file():
        qpos, qvel, zs = _play_ramp()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        _np.savez(tmp, qpos=qpos, qvel=qvel, trunk_z=zs)
        os.replace(tmp, path)
    return _np.load(path, allow_pickle=True)


def _play_ramp(n_variants: int = 24, seed: int = 0):
    """Ramp the servos up the arm plateau and record where it ends up.

    Variety comes from the RUNG the ramp starts on and from small joint jitter
    on each waypoint -- not from randomising the end state, which would cache
    poses the ramp cannot actually produce.
    """
    import mujoco

    from . import asimov_walk as _W
    from . import spec as S

    clip = float(S.get("asimov1").action_clip)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, getup_blend=0.0,
        getup_assist=0.0, action_clip=clip)
    m, d = env.model, env.data
    trunk = env.trunk_body_id
    qadr = np.array([m.joint(n).qposadr[0] for n in _W.JOINT_NAMES])

    z = holdable_rungs()
    tz = z["trunk_z"]
    sup = np.array([str(x) for x in z["support"]])
    J = z["joints"]
    Q = J[:, qadr]
    arm = np.where(sup == "arm")[0]
    order = arm[np.argsort(tz[arm])]

    def act(q):
        return ((np.asarray(q) - np.asarray(_W.DEFAULT_POSE)) / _W.ACTION_SCALE
                ).astype(np.float32).clip(-clip, clip)

    # the ladder: greedy in joint space, climbing, each leg verified to arrive
    path = [int(order[0])]
    while True:
        cur = path[-1]
        cand = [i for i in arm if i not in path and tz[i] > tz[cur] + 0.004]
        if not cand:
            break
        cand.sort(key=lambda i: float(np.abs(Q[cur] - Q[i]).sum()))
        nxt = None
        for i in cand[:6]:
            d.qpos[:] = J[cur]
            d.qvel[:] = 0.0
            d.xfrc_applied[:] = 0.0
            mujoco.mj_forward(m, d)
            _ramp_leg(env, Q[cur], Q[i], 0.25, act, qadr)
            for _ in range(50):
                d.xfrc_applied[:] = 0.0
                env.step(act(Q[i]))
            if abs(float(d.xpos[trunk][2]) - tz[i]) < 0.04:
                nxt = i
                break
        if nxt is None:
            break
        path.append(nxt)
    if len(path) < 3:
        raise RuntimeError(f"the scripted ramp found only {len(path)} rungs")

    rng = np.random.default_rng(seed)
    out_q, out_v, out_z = [], [], []
    for v in range(n_variants):
        first = path[0] if v == 0 else path[int(rng.integers(0, max(1, len(path) // 3)))]
        d.qpos[:] = J[first]
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        cur = Q[first]
        for i in path[path.index(first) + 1:]:
            tgt = Q[i] + (rng.normal(0.0, 0.01, Q[i].shape) if v else 0.0)
            _ramp_leg(env, cur, tgt, 0.25, act, qadr)
            cur = tgt
        lf = m.body(m.geom_bodyid[env.foot_geoms["left"]]).id
        rf = m.body(m.geom_bodyid[env.foot_geoms["right"]]).id
        span = float(np.hypot(d.xpos[lf][0] - d.xpos[rf][0],
                              d.xpos[lf][1] - d.xpos[rf][1]))
        if span > EPISODE_SPAN_CUT_M:
            # never cache a start the episode cut would end on step 1
            continue
        out_q.append(d.qpos.copy())
        out_v.append(d.qvel.copy())
        out_z.append(float(d.xpos[trunk][2]))
    if not out_q:
        raise RuntimeError(
            "every ramp variant ended wider than the episode span cut "
            f"({EPISODE_SPAN_CUT_M} m) — the ramp and the cut disagree")
    return np.array(out_q), np.array(out_v), np.array(out_z)


def _ramp_leg(env, cur, tgt, secs, act, qadr):
    """One smoothstep leg of the scripted ramp, through the position servos."""
    n = max(2, int(secs * 50))
    for k in range(n):
        f = (k + 1) / n
        f = f * f * (3.0 - 2.0 * f)
        env.data.xfrc_applied[:] = 0.0
        env.step(act(np.asarray(cur) + (np.asarray(tgt) - np.asarray(cur)) * f))


def fall_path(dest=None):
    from . import asimov as _asimov
    return _P(dest or _asimov.CACHE_DIR) / "fall_states.npz"


def route_states(lo: float = 0.0, hi: float = 9.9, dest=None):
    """States a FEASIBLE get-up passes through, filtered to a trunk band.

    The authored waypoints do not sample the route that works. Per-foot CoM
    distance is 0.577 at `pike` and 0.137 at `fold` with nothing between, and
    uprightness goes to -0.37 at `pike` -- face down. Twelve reward economies
    and a 4x budget all failed to cross that.

    The vendor's shipped policy, however, rises **12/12 from prone in this
    env under the real `dc` envelope**, and its route is different in kind:
    uprightness stays POSITIVE throughout (+0.19 -> +1.00), both feet are
    down by 1.0 s, and the per-foot distance closes GRADUALLY and
    monotonically (0.651 -> 0.115) through a wide stance that narrows only at
    the end. So the continuum exists and the reference misses it. Harvested
    by `scripts/harvest_route_states.py`, the bands are monotone in both
    quantities: trunk 0.50-0.60 upright +0.93 per-foot 0.346; 0.40-0.50 +0.73
    / 0.417; 0.30-0.40 +0.48 / 0.482; 0.20-0.30 +0.47 / 0.505; 0.00-0.20
    +0.38 / 0.53-0.57.

    ONLY `qpos`/`qvel` are taken -- geometry the body demonstrably passes
    through, from episodes that actually ended standing. No actions, no
    observations, no distillation: a policy started here learns its own
    actions from scratch, and the demonstration only supplies the question
    "can you recover from this state?".
    """
    import numpy as _np

    from . import asimov as _a

    path = dest or (_a.CACHE_DIR / "route_states.npz")
    if not _P(str(path)).is_file():
        raise RuntimeError(
            f"{path} missing — run scripts/harvest_route_states.py first")
    d = _np.load(str(path))
    z = _np.asarray(d["trunk_z"])
    sel = (z >= lo) & (z < hi)
    if not sel.any():
        raise RuntimeError(
            f"no route states in trunk band [{lo}, {hi}) — the cache spans "
            f"{z.min():.3f}..{z.max():.3f} m")
    return {"qpos": _np.asarray(d["qpos"])[sel],
            "qvel": _np.asarray(d["qvel"])[sel],
            "trunk_z": z[sel]}


def fall_states(dest=None, rebuild: bool = False):
    """States the robot passes through FALLING from `fold` toward `pike`.

    The leg `rock_back` -> `fold` cannot be learned from a static start: the
    policy stands 16/16 from `fold` and **0/16** from `rock_back`, even when
    trained directly on it, and the move fails open loop at 25 tempos, with an
    authored counter-movement, and at 5x arm torque. What decides it is the
    contact mode, and the contact mode is decided by momentum.

    But the fall is FREE. Shove a settled `fold` forward and it drops through
    the whole gap in about a second — 0.517 -> 0.491 -> 0.423 -> 0.284 — so
    every state between the two is reachable BY CONSTRUCTION, carrying the
    velocity it really has. That is exactly what three interpolated pose
    ladders lacked: their rungs were averages of two poses and fell on their
    own.

    Each entry is `(qpos, qvel, trunk_z)`. Training reads them shallow-first:
    near `fold` the recovery is nearly the finish the policy already owns, and
    the curriculum grows the shove until the start is effectively `pike`.
    """
    import os

    import numpy as _np

    path = fall_path(dest)
    if rebuild or not path.is_file():
        q, v, z = _search_fall()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        _np.savez(tmp, qpos=q, qvel=v, trunk_z=z)
        os.replace(tmp, path)
    return _np.load(path, allow_pickle=True)


def _search_fall(pushes=(60.0, 75.0, 90.0, 110.0), settle=100):
    """Settle at `fold`, shove forward, and record the whole descent."""
    import mujoco

    from . import spec as S

    clip = float(S.get("asimov1").action_clip)
    env = S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, seed=0, action_clip=clip)
    m, d = env.model, env.data
    trunk = env.trunk_body_id
    sc = float(np.atleast_1d(env.robot.action_scale)[0])
    fold, pike = BY_NAME["fold"], BY_NAME["pike"]
    a = np.clip((np.asarray(fold.joints) - np.asarray(env.robot.default_pose))
                / sc, -clip, clip).astype(np.float32)
    qs, vs, zs = [], [], []
    for push in pushes:
        env._place(fold.joints, fold.root_pitch)
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(settle):
            d.xfrc_applied[:] = 0.0
            env.step(a)
        for k in range(200):
            d.xfrc_applied[:] = 0.0
            if k < 10:
                d.xfrc_applied[trunk, 0] = push
            env.step(a)
            z = float(d.xpos[trunk][2])
            # keep only what is IN the gap and still upright-ish; below pike
            # the robot is on its forearms and that is a different task
            if pike.settled_z - 0.02 <= z <= fold.settled_z + 0.01:
                qs.append(d.qpos.copy())
                vs.append(d.qvel.copy())
                zs.append(z)
            if z < pike.settled_z - 0.02:
                break
    if len(qs) < 20:
        raise RuntimeError(f"only {len(qs)} fall states — the shove is wrong")
    return np.array(qs), np.array(vs), np.array(zs)


def pike_path(dest=None):
    from . import asimov as _asimov
    return _P(dest or _asimov.CACHE_DIR) / "pike_states.npz"


def pike_states(dest=None, rebuild: bool = False):
    """The AUTHORED pike, settled — narrow, on forearms and feet.

    Not the same thing as the arm-plateau rungs, and the difference is the
    whole point of the hybrid: the rung cache's arm poses are SPLAYED (0.70 m
    foot span) because the search was free to open the hips, while every
    authored waypoint sits at **0.215 m**, exactly the standing stance. A
    hybrid that keeps the narrow stance everywhere except the launch has to
    start from the narrow pose.
    """
    import os

    import numpy as _np

    path = pike_path(dest)
    if rebuild or not path.is_file():
        q, v, z = _settle_pike()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        _np.savez(tmp, qpos=q, qvel=v, trunk_z=z)
        os.replace(tmp, path)
    return _np.load(path, allow_pickle=True)


def _settle_pike(n_variants: int = 16, seed: int = 0):
    """Place the authored pike, hold it, and record where it settles."""
    import mujoco

    from . import spec as S

    clip = float(S.get("asimov1").action_clip)
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, getup_blend=0.0,
        getup_assist=0.0, action_clip=clip)
    m, d = env.model, env.data
    trunk = env.trunk_body_id
    sc = float(np.atleast_1d(env.robot.action_scale)[0])
    w = BY_NAME["pike"]
    rng = np.random.default_rng(seed)
    qs, vs, zs = [], [], []
    for i in range(n_variants):
        q = np.asarray(w.joints, float)
        if i:
            q = q + rng.normal(0.0, 0.015, q.shape)
        env._place(q, w.root_pitch + (0.0 if not i else rng.normal(0, 0.02)))
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        a = np.clip((q - np.asarray(env.robot.default_pose)) / sc,
                    -clip, clip).astype(np.float32)
        for _ in range(100):
            d.xfrc_applied[:] = 0.0
            env.step(a)
        z = float(d.xpos[trunk][2])
        if z < 0.24:                     # it fell: not a pike
            continue
        qs.append(d.qpos.copy())
        vs.append(d.qvel.copy())
        zs.append(z)
    if not qs:
        raise RuntimeError("the authored pike did not settle in any variant")
    return np.array(qs), np.array(vs), np.array(zs)


def _search_rungs():
    """The two sweeps. Kept here so the table above is reproducible, not a
    number somebody once printed into a chat window."""
    import itertools

    import mujoco

    from . import spec as S

    # THE CLIP IS THE ROBOT'S, NOT A NUMBER CHOSEN HERE. It used to be a
    # hardcoded 13.0 while every trainer and the deployment run at
    # `RobotSpec.action_clip` (4.0), so this function certified rungs under
    # conditions nothing else uses: of the 221 rungs it passed at 13.0, only
    # **82 still hold at 4.0**. A curriculum then started its episodes on 139
    # poses the policy cannot hold — each one a fall already in progress,
    # which is exactly the defect that made three interpolated ladders
    # useless. Measured on `teach-asimov_getup_floor-cdd31c`: all 64 of stage
    # 2's rungs demanded more than the clip it trained with, and the chain
    # scored 0/36 from the floor at every one of its five stages — identical
    # to commanding nothing.
    #
    # Raising the TRAINER to 13.0 is the wrong direction, and was tried and
    # rejected on 2026-10-04 (see the note by `asimov_walk.ACTION_SCALE`): it
    # broke `fold`, which stopped holding at all, because every verified
    # waypoint was measured as a CLIPPED pose. So the clip is the robot's
    # everywhere, and a rung has to be an equilibrium THERE.
    _clip = float(S.get("asimov1").action_clip)
    env = S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, seed=0, action_clip=_clip)
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
                      / scale, -_clip, _clip).astype(np.float32)
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
    for sh, el, hip, knee, pitch, roll in itertools.product(
            (-0.4, -1.2, -2.0, -2.8), (0.2, 0.7, 1.2, 1.7),
            (-0.4, -1.0, -1.8, -2.4), (0.3, 0.9, 1.5, 2.0),
            (0.0, 0.2, 0.5, 0.9, 1.2),
            # the splay, which neither sweep used to vary at all
            (0.0, 0.45, 0.785)):
        q = joints(hip_pitch=hip, knee=knee, ankle_pitch=-0.35,
                   hip_roll=roll, shoulder_pitch=sh, elbow=el)
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
    #
    # STANCE WIDTH is in the grid, added 2026-10-06. With `hip_roll` and
    # `hip_yaw` at 0 the two feet sit 0.215 m apart and the support set is a
    # LINE, so the deepest hold is whatever the +-0.35 rad ankle can reach
    # with the CoM over that line. Splayed toward the hip stop the feet are
    # 0.59-0.74 m apart and the robot holds a wide sumo squat.
    #
    # The number, against its own control INSIDE this sweep, at the robot's
    # own clip (corrected 2026-10-07 — it was first measured at a hardcoded
    # 13.0, which nothing trains or deploys at):
    #
    #   feet-only rungs, hips square   n= 7, lowest **0.529 m**
    #   feet-only rungs, splayed       n=71, lowest **0.416 m**
    #
    # So the splay is worth **113 mm** here — more than the 53 mm it looked
    # worth at the wrong clip — and it is what makes a feet-only pose below
    # the crouch exist at all.
    #
    # It does NOT close the gap, and the first write-up of this said it did.
    # At the real clip the arm plateau tops out at 0.280 m and the feet
    # plateau starts at 0.416, so **136 mm separates them** — the two
    # overlapped by 41 mm only at 13.0. The handover across that gap is still
    # the one move nothing here can do open loop: the arm plateau traverses
    # 0.136 -> 0.267 m in 7 consecutive legs with no policy at all, and the
    # leg into the feet plateau breaks at every tempo (roadmap 3i).
    #
    # Phase 4, for this plateau: it is CHEAP. Over all 78 feet rungs that
    # hold, hip_roll peaks at a median 2 % of rated and a maximum of 19 %, and
    # none exceeds rated — against a walk baseline that already runs
    # ankle_pitch, ankle_roll and shoulder_roll at 100 %. A wide stance is not
    # what will stop this working on hardware.
    #
    # The arm reach is in the grid too: `fold` holds "only because the arms
    # reach FORWARD to carry the centre of mass", and this sweep used to pin
    # them at shoulder -0.4.
    for hip, knee, ank, pitch, roll, yaw, sh in itertools.product(
            (-0.2, -0.6, -1.0, -1.2, -1.4, -1.6),
            (0.25, 0.5, 0.9, 1.2, 1.5), (-0.35, -0.2, 0.0),
            (0.02, 0.15, 0.25, 0.35, 0.47, 0.60, 0.70),
            (0.0, 0.35, 0.55, 0.70, 0.785),
            # negative = toes OUT (measured; see MIRRORED)
            (0.0, -0.45, -0.785),
            (-0.4, -1.4)):
        q = joints(hip_pitch=hip, knee=knee, ankle_pitch=ank, hip_roll=roll,
                   hip_yaw=yaw, shoulder_pitch=sh,
                   elbow=1.4 if sh < -1.0 else 0.6)
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
        # The span is measured HERE, on the state the cache actually STORES,
        # not on wherever the replay happens to end. Measured on the replay's
        # end state instead, two rungs came out 2.6 mm over the cut — stored
        # as rungs the env would terminate on at step 1. Same shape as every
        # other fault on this task: validate the artefact, not a neighbour of
        # it.
        _lf = m.body(m.geom_bodyid[env.foot_geoms["left"]]).id
        _rf = m.body(m.geom_bodyid[env.foot_geoms["right"]]).id
        span = float(np.hypot(d.xpos[_lf][0] - d.xpos[_rf][0],
                              d.xpos[_lf][1] - d.xpos[_rf][1]))
        if span > MAX_FOOT_SPAN_M:
            continue
        a = np.clip((d.qpos[env._q_adr] - np.asarray(env.robot.default_pose))
                    / scale, -_clip, _clip).astype(np.float32)
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
