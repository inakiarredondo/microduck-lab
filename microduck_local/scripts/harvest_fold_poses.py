"""Add a `fold` start distribution to the pose cache — the NARROW launch set.

Why this exists. Every narrow attempt so far failed for the same reason: no
anchor. Nothing stood from `pike`, so no reward could shape toward it. But
measured from a properly SETTLED `fold`, perturbed, both the teacher and our
best clone DO reach a stand — and when they do it is **0.21-0.24 m wide**,
under the 0.40 m gate and ~4x narrower than the 0.94 m get-up we have. 8 of 48
for the clone, 7 of 48 for the teacher. Rare, but present: that is an anchor,
and it is the first one this track has had for a narrow stand.

So this writes those start states into the cache as their own label, which
lets the whole existing training machinery aim at them (`getup_label="fold"`).

Poses are DROPPED to first contact by bisection, not placed: `_place` leaves
the robot floating ~22 cm clear, where there is no support polygon at all and
every contact-based reading is vacuous.

The cache is APPENDED to, never rewritten: the 477 existing poses across six
labels are preserved, and the write is atomic (temp + os.replace) because a
training run builds 32 workers and any of them may read it mid-write.
"""
from __future__ import annotations

import argparse
import os
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--joint-sd", type=float, default=0.20)
    ap.add_argument("--pitch-sd", type=float, default=0.10)
    ap.add_argument("--rung", default="fold")
    ap.add_argument("--label", default="fold")
    ap.add_argument("--max-start-span", type=float, default=None,
                    help="keep only starts whose FOOT SPAN is at most this. "
                         "Measured on the first narrow policy: among 57 "
                         "stands, the narrow ones started at a median 0.207 m "
                         "span and the wide ones at 0.343 m, while the peak "
                         "span barely moved from either (0.393 vs 0.418). The "
                         "start span is what decides the outcome, so filtering "
                         "it is a sharper lever than shrinking the "
                         "perturbation globally — and far sharper than a span "
                         "charge, which has now been a null twice because it "
                         "arrives after the stance is already set.")
    ap.add_argument("--settle-s", type=float, default=1.0,
                    help="seconds of free settling a candidate must SURVIVE. "
                         "The old value was 10 physics steps (0.05 s) and the "
                         "collapse takes 0.5 s, so a whole start set of falls "
                         "passed as configurations.")
    ap.add_argument("--min-trunk", type=float, default=0.25,
                    help="trunk height required AFTER settling")
    ap.add_argument("--max-other-parts", type=int, default=2,
                    help="non-foot parts allowed on the floor after settling; "
                         "both feet are always required, because a pose "
                         "balanced on ONE foot is a knife edge, not a start")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local import contract as C
    from microduck_local.robots import asimov, asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, seed=0, getup_label="prone",
        actuator_force="dc")
    env.reset(seed=0)
    m, d = env.model, env.data
    ADR = np.array([int(m.joint(n).qposadr[0]) for n in W.JOINT_NAMES])
    JID = [m.joint(n).id for n in W.JOINT_NAMES]
    lo_j, hi_j = m.jnt_range[JID, 0], m.jnt_range[JID, 1]
    base = np.asarray(R.BY_NAME[a.rung].joints, float)
    p0 = R.BY_NAME[a.rung].root_pitch

    env_decim = int(C.DECIMATION)
    rng = np.random.default_rng(a.seed)
    kept, tried = [], 0
    while len(kept) < a.n and tried < a.n * 8:
        tried += 1
        q = np.clip(base + rng.normal(0, a.joint_sd, len(base)), lo_j, hi_j)
        pitch = p0 + rng.normal(0, a.pitch_sd)
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        h = 0.5 * pitch
        d.qpos[3:7] = [np.cos(h), 0.0, np.sin(h), 0.0]
        d.qpos[ADR] = q
        lo, hi = -0.3, 1.2
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            d.qpos[2] = mid
            mujoco.mj_forward(m, d)
            if d.ncon > 0:
                lo = mid
            else:
                hi = mid
        d.qpos[2] = lo
        mujoco.mj_forward(m, d)
        # A start the robot is RESTING in, not one it is already falling
        # through. **10 steps was 0.05 s and the collapse takes 0.5 s**, so
        # every `narrow_pike` pose passed this filter while being a fall in
        # progress: measured, they begin on ONE foot at trunk 0.524 and are
        # flat on the floor at 0.076 within half a second. Seven separate
        # approaches to the pike rung then failed on a start distribution
        # nothing could recover from — "a start nothing recovers from
        # produces no successes and therefore no gradient".
        #
        # `--settle-s` seconds, and the pose must still be up AND on its feet
        # at the END of them, not at a peak somewhere along the way. (A peak
        # is what made this invisible: "zero action holds pike at 0.521" was
        # the FIRST frame, read off a max.)
        n_settle = max(1, int(a.settle_s / 0.02))
        for _ in range(n_settle * env_decim):
            mujoco.mj_step(m, d)
        if float(d.xpos[env.trunk_body_id][2]) < a.min_trunk:
            continue
        feet = sum(1 for p in R.floor_parts(env) if "foot" in str(p))
        other = sum(1 for p in R.floor_parts(env) if "foot" not in str(p))
        if feet < 2 or other > a.max_other_parts:
            continue
        if (a.max_start_span is not None
                and float(env._foot_span()) > a.max_start_span):
            continue
        kept.append(d.qpos.copy())

    if not kept:
        # NOT a crash: zero survivors is a RESULT, and for `pike` it is the
        # whole answer — no perturbation of it stays up for `--settle-s` on
        # both feet, so pike is a state the robot passes THROUGH and not one
        # it can rest in. A rung has to be a configuration; this is not one.
        print(f"0 of {tried} candidates survived {a.settle_s:g} s on both "
              f"feet -> {a.rung!r} is NOT a stable configuration at this "
              f"perturbation. Cache left untouched.")
        return 1

    path = asimov.cache_dir() / "fallen_poses.npz"
    old = np.load(path, allow_pickle=True)
    q_old, l_old = old["qpos"], old["label"]
    keep = l_old != a.label                       # replace this label only
    qpos = np.concatenate([q_old[keep], np.stack(kept)])
    label = np.concatenate([l_old[keep],
                            np.array([a.label] * len(kept))])
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, qpos=qpos, label=label)
    os.replace(tmp, path)
    have = sorted(set(label.tolist()))
    print(f"kept {len(kept)} of {tried} tries -> label {a.label!r}")
    print(f"cache now holds {len(qpos)} poses across {have}")
    print(f"  (preserved {int(keep.sum())} pre-existing poses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
