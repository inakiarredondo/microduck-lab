"""Is openhorizon's 5-frame history LOAD-BEARING, or is it spare capacity?

    uv run --with onnxruntime python scripts/probe_history_ablation.py --episodes 24

THE question for a contract-preserving get-up here. Menlo's shipped layout is
78 floats of CURRENT state and nothing else; their get-up takes 375, which is
the same quantities 5 frames deep (`robots/asimov_openhorizon` measures that
the terms, scales, joint orders and default pose all match exactly). So the
whole of the gap between their policy and our contract is memory.

If their policy still rises when every history slot is filled with the CURRENT
frame -- a policy with no memory at all, driven by the same weights -- then a
78-input single-frame student can represent this controller and the
distillation route is sound. If it collapses, the history is doing real work
and a one-frame get-up on this body needs either a different policy class or a
change to the published observation, which is a finding worth sending to Menlo
rather than a training problem.

This is an ABLATION of the input, not of the network: the graph, the weights
and the action mapping are untouched, so nothing here depends on retraining.
"""
from __future__ import annotations

import argparse

import numpy as np

from microduck_local.robots import asimov_openhorizon as OH


class NoMemory(OH.OpenHorizonPolicy):
    """Their policy with the ring frozen to the present.

    Every slot holds the current frame, which is exactly what their own spec
    says happens on the FIRST step after a (re)start ("every history slot is
    filled with the first value") -- so this is not an out-of-distribution
    input shape, it is the input they already feed on step one, held forever.
    """

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self._buf = None          # force the "first frame" fill every step
        return super().__call__(obs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--pose", default="prone,supine")
    ap.add_argument("--actuator", default="dc", choices=("xml", "dc"))
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S

    print(f"openhorizon get-up, actuator={a.actuator!r}, "
          f"{a.episodes} episodes per pose, paired by seed\n")
    for pose in [p.strip() for p in a.pose.split(",") if p.strip()]:
        res = {}
        for label, cls in (("5 frames (as published)", OH.OpenHorizonPolicy),
                           ("1 frame  (history ablated)", NoMemory)):
            env = S.get("asimov1").env_class("getup")(
                domain_rand=False, obs_noise=False, action_delay=False, seed=0,
                getup_label=pose, actuator_force=a.actuator, action_clip=13.0,
                fell_cut=False, max_episode_s=8.0, limit_cost=0.0)
            pol = cls("getup")
            rises, stays, times = [], [], []
            for s in range(a.episodes):
                o, _ = env.reset(seed=s)
                pol.reset()
                up_s, t_up, last = 0.0, None, False
                for k in range(int(env.max_steps)):
                    o, _r, t, tr, _i = env.step(pol(o))
                    last = R.is_standing(env)
                    up_s = up_s + 0.02 if last else 0.0
                    if t_up is None and up_s >= 1.0:
                        t_up = (k + 1) * 0.02 - 1.0
                    if t or tr:
                        break
                rises.append(t_up is not None)
                stays.append(last)
                if t_up is not None:
                    times.append(t_up)
            res[label] = rises
            med = f"{np.median(times):.2f} s" if times else "—"
            print(f"  {pose:7s} {label:28s} rises {sum(rises):3d}/{a.episodes}"
                  f"  stays {sum(stays):3d}/{a.episodes}  median {med}")
        b, c = res["5 frames (as published)"], res["1 frame  (history ablated)"]
        bb = sum(1 for x, y in zip(b, c) if x and not y)
        cc = sum(1 for x, y in zip(b, c) if y and not x)
        print(f"          -> history is "
              f"{'LOAD-BEARING' if bb > cc else 'NOT load-bearing'}: "
              f"{bb} episodes rise only WITH it, {cc} only without\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
