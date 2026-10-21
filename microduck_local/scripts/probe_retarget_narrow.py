"""Can the demonstration be RETARGETED narrow, rather than searched for?

The narrow track's missing piece is a demonstration of the crossing. A
trajectory search never found one, span penalties in training are measured
dead, and so is filtering the existing demos — but all three are either
"select among what exists" or "charge for width after the fact". RETARGETING
is neither: it takes a demonstration that WORKS and modifies the channel that
makes it wide.

The sprawl is hip ABDUCTION. `getup_clips_dc.npz` holds 40 clips harvested in
THIS physics under the real actuator envelope, each a (qpos, action) sequence
that rises. So the question is whether scaling the hip-roll component of those
actions narrows the stance while the rise survives — and it is answered by
REPLAYING, open loop, in the deployment env, not by arguing about it.

A scale that keeps the rise and halves the span is a narrow demonstration, and
a narrow demonstration is the thing this track has never had. A scale that
loses the rise says the abduction is load-bearing, which is also worth knowing
and is the `the-sprawl-is-the-launch` claim tested on the ACTIONS rather than
on a reward.
"""
from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", default=".cache/asimov_1/getup_clips_dc.npz")
    ap.add_argument("--scales", default="1.0,0.75,0.5,0.25,0.0")
    ap.add_argument("--n", type=int, default=12, help="clips to replay")
    a = ap.parse_args(argv)

    import mujoco
    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    z = np.load(a.clips, allow_pickle=True)
    qpos, acts = z["qpos"], z["actions"]
    lengths, starts = z["lengths"], z["starts"]
    names = [str(x) for x in z["joint_names"]]
    JN = list(W.JOINT_NAMES)
    assert names == JN, "clip joint order is not the env's"
    roll = [i for i, n in enumerate(JN) if "hip_roll" in n]
    print(f"  {len(lengths)} clips, scaling the HIP-ROLL action channel "
          f"{[JN[i] for i in roll]}\n")

    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=False, action_delay=False, seed=0,
        getup_label="prone", actuator_force="dc", action_clip=4.0,
        fell_cut=False, max_episode_s=12.0)
    m, d = env.model, env.data
    pel = m.body("pelvis_link").id

    offs = np.concatenate([[0], np.cumsum(lengths)])
    print(f"  {'scale':>6s}{'rises':>8s}{'held 1s':>9s}{'peak span':>11s}"
          f"{'span@rise':>11s}{'peak z':>9s}")
    for sc in [float(x) for x in a.scales.split(",")]:
        rises = held = 0
        peaks, at_rise, zs = [], [], []
        for c in range(min(a.n, len(lengths))):
            lo, hi = int(offs[c]), int(offs[c + 1])
            A = acts[lo:hi].copy()
            A[:, roll] *= sc
            env.reset(seed=c)
            # START FROM THE CLIP'S OWN START, so the only thing that differs
            # between scales is the abduction channel.
            d.qpos[:] = starts[c]
            d.qvel[:] = 0.0
            env._obs_stack, env._obs_live = [], False
            mujoco.mj_forward(m, d)
            mx, pz, up_run, best_run, span_up = 0.0, 0.0, 0, 0, None
            for k in range(len(A)):
                env.step(A[k].astype(np.float32))
                s = float(env._foot_span())
                mx = max(mx, s)
                pz = max(pz, float(d.xpos[pel][2]))
                if R.is_standing(env):
                    up_run += 1
                    best_run = max(best_run, up_run)
                    if span_up is None:
                        span_up = mx
                else:
                    up_run = 0
            if best_run:
                rises += 1
                at_rise.append(span_up if span_up is not None else mx)
            if best_run * 0.02 >= 1.0:
                held += 1
            peaks.append(mx)
            zs.append(pz)
        n = min(a.n, len(lengths))
        ar = f"{np.median(at_rise):.3f}" if at_rise else "-"
        print(f"  {sc:6.2f}{f'{rises}/{n}':>8s}{f'{held}/{n}':>9s}"
              f"{np.median(peaks):11.3f}{ar:>11s}{np.median(zs):9.3f}",
              flush=True)
    print("\n  'held 1s' requires a full second of is_standing, not a frame —"
          "\n  a one-footed flicker already fooled one measurement today.")
    print("  the acceptance gate asks for a peak span under 0.40 m; the")
    print("  unretargeted teacher sprawls ~0.93-0.94 m.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
