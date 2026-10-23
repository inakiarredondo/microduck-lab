"""Does the assisted crossing ever come NEAR the catch basin?

At assist 0.85 `asimov_getup_gap` reaches peak trunk 0.672 m — above a stand —
and stands 0/24, because it arrives at uprightness -0.86: face down. The plan
that follows is to pay for the ARRIVAL rather than the apex, but a reward can
only amplify a success that OCCURS, so the prior question is whether any
episode ever passes close to `fold`'s pose.

`fold` placed is the target: pelvis 0.511 m, uprightness +0.83, pelvis-to-foot
gap -0.071 m, span 0.215 m, and `9702ba` catches it below 0.2 m/s of arrival
lift. So each episode is searched for the frame CLOSEST to that pose and the
four components are reported separately — because a single scalar distance
hides which clause is the one that is missed, and this repo has twice invented
progress by reporting two different frames' maxima as one pose.

Reports the best frame per episode, then the distribution. The question it
answers: is the gap between the assisted crossing and the basin something a
reward term could close, or is it a different motion entirely?
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

#: `fold`, PLACED and measured (probe_com_decompose.py), not the cache mean.
TARGET = dict(z=0.511, upright=0.83, gap=-0.071, span=0.215, speed=0.20)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--episodes", type=int, default=32)
    a = ap.parse_args(argv)

    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    kw = dict(replay_kwargs(Path("runs") / a.run))
    task = kw.pop("task", None) or "getup_ref"
    kw.update(fell_cut=False, max_episode_s=8.0)
    env = S.get("asimov1").env_class(task)(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    m, d = env.model, env.data
    pelvis = m.body("pelvis_link").id
    FB = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
    pol = GA.load(a.run)

    def frame():
        z = float(d.xpos[pelvis][2])
        up = float(-env._projected_gravity()[2])
        gap = z * 0.0 + float(d.xpos[pelvis][0]) - float(
            np.mean([d.xpos[b][0] for b in FB]))
        return z, up, gap, float(env._foot_span()), float(abs(d.qvel[2]))

    rows = []
    for s in range(a.episodes):
        o, _ = env.reset(seed=s)
        if hasattr(pol, "reset"):
            pol.reset()
        best = None
        for _k in range(int(env.max_steps)):
            o, _r, t, tr, _i = env.step(pol(o))
            z, up, gap, span, v = frame()
            # One scalar to PICK the frame, four to report it. Scaled so each
            # clause contributes comparably at its own tolerance.
            dist = (abs(z - TARGET["z"]) / 0.05
                    + abs(up - TARGET["upright"]) / 0.20
                    + abs(gap - TARGET["gap"]) / 0.08
                    + max(0.0, span - TARGET["span"]) / 0.10
                    + max(0.0, v - TARGET["speed"]) / 0.20)
            if best is None or dist < best[0]:
                best = (dist, z, up, gap, span, v)
            if t or tr:
                break
        rows.append(best)

    arr = np.array(rows)
    print(f"{a.run}: closest approach to FOLD's pose, {a.episodes} episodes")
    print(f"  target: pelvis {TARGET['z']:.3f} m, upright {TARGET['upright']:+.2f}, "
          f"gap {TARGET['gap']:+.3f} m, span <= {TARGET['span']:.3f} m, "
          f"lift <= {TARGET['speed']:.2f} m/s\n")
    names = ("distance", "pelvis z", "upright", "pelvis gap", "span", "lift")
    print(f"  {'':12s}{'best':>9s}{'p10':>9s}{'median':>9s}{'p90':>9s}")
    for i, n in enumerate(names):
        c = arr[:, i]
        k = np.argmin(arr[:, 0])
        print(f"  {n:<12s}{c[k]:9.3f}{np.percentile(c, 10):9.3f}"
              f"{np.median(c):9.3f}{np.percentile(c, 90):9.3f}")
    print("\n  'best' is the single CLOSEST frame of all episodes, read as one "
          "instant;\n  the percentiles are over each episode's own closest "
          "frame.")
    d0, z0, u0, g0, s0, v0 = rows[int(np.argmin(arr[:, 0]))]
    miss = []
    if abs(z0 - TARGET["z"]) > 0.05:
        miss.append(f"height by {abs(z0 - TARGET['z']) * 1000:.0f} mm")
    if abs(u0 - TARGET["upright"]) > 0.20:
        miss.append(f"uprightness by {abs(u0 - TARGET['upright']):.2f}")
    if abs(g0 - TARGET["gap"]) > 0.08:
        miss.append(f"the gap by {abs(g0 - TARGET['gap']) * 1000:.0f} mm")
    if s0 > TARGET["span"] + 0.10:
        miss.append(f"span by {(s0 - TARGET['span']) * 1000:.0f} mm")
    if v0 > TARGET["speed"] + 0.20:
        miss.append(f"lift by {v0 - TARGET['speed']:.2f} m/s")
    print(f"\n  the closest frame misses: "
          f"{', '.join(miss) if miss else 'NOTHING — it reached the basin'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
