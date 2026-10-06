"""An ONNX teacher that holds the nearest REFERENCE pose — 78 obs -> 23 actions.

    uv run python scripts/make_hold_teacher.py

Why this exists. The deployment contract is
`ctrl = DEFAULT_POSE + ACTION_SCALE * action` with `use_default_offset: true`
(Menlo's `env.yaml`), so **action 0 commands STANDING**. On the floor that is a
lunge: at the `pike` the zero action is up to 1.649 rad (95 deg) from the joint
the reset just settled. MEASURED 2026-10-04, 2 s of action 0 from each rung:

| rung | end trunk z | tracking earned | with its OWN holding action |
|---|---|---|---|
| prone | 0.009 | 0.56 | 0.76 |
| plant_forearms | 0.009 | 0.36 | **0.85** |
| draw_knees_in | 0.008 | 0.27 | **0.93** |
| quadruped | 0.008 | 0.20 | **0.77** |
| pike | 0.009 | 0.31 | **0.89** |

Every rung collapses flat. A policy initialised with a zero-mean output has to
discover a 23-dim constant offset of up to 4.8 action units BEFORE any part of
the get-up can be rewarded, and six training chains never did.

It does not have to be discovered. The observation carries
`jpos = qpos - DEFAULT_POSE` at slots 9..31 (`OBS_JOINT_POS`), permuted by
`slot_order()`, so the action commanding ANY pose is just
`(pose - DEFAULT_POSE) / ACTION_SCALE` and the current pose is readable.

TRIED FIRST, and it fails: commanding the pose you are CURRENTLY in. That is a
zero-stiffness follower — it asks for exactly where the joints already are, so
gravity is unopposed and it ratchets down. MEASURED: `plant_forearms` 0.161 ->
0.028, `draw_knees_in` 0.196 -> 0.082, `quadruped` 0.212 -> 0.110, tracking
0.43-0.60. Better than the lunge and still a collapse.

So it snaps to the nearest WAYPOINT instead. That has restoring stiffness — as
the robot sags the command stays on the waypoint and pulls it back — and it is
still a function of the observation alone, which is what lets it be an ONNX and
be cloned by `distill` with machinery that is already tested. The chain then
warm-starts from a policy that holds its rung rather than one that falls off it,
and what PPO has to discover is how to ADVANCE, which is the actual skill.

AND THE SNAP IS SOFT, which is the difference between a warm start PPO keeps and
one it throws away. A hard `argmin` DITHERS: near a boundary the nearest
waypoint flips back and forth and the command swings the whole distance between
two waypoint actions, up to 6 units, every flip. MEASURED on the hard-snap
clone, mean |da| per joint per control step was **1.60** against
**0.055-0.078** for the trained stand — roughly 20 rad/s of sustained chatter —
so `action_rate_penalty` charged it **-431** against positive terms worth +196,
and 1.2 M steps of PPO correctly walked away from it. The penalty was right; the
teacher was jerky.

TRIED SECOND, and it fails for a subtler reason: rate-limiting the command
through `OBS_LAST_ACTION`. The TEACHER is then beautifully smooth (mean |da|
0.005-0.020, penalty -0.2) and the CLONE is not (1.80-2.26, -232 to -1327),
because a rate limiter is RECURSIVE — it reads its own previous output, so the
student's residual error feeds back through the loop and oscillates. A 0.00065
rad^2 fit says nothing about closed-loop stability.

So the weights are a softmax over waypoint distance at tau = 0.01, which is well
inside the closest pair's separation (`rock_back` and `fold` are 0.0246 apart in
mean joint distance; the median nearest-neighbour gap is 0.0957). That is smooth
in the joint positions, has no dither to chatter on, and is a pure feed-forward
function of the observation — which is what makes it safe to clone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microduck_local.robots import asimov  # noqa: E402
from microduck_local.robots import asimov_getup_ref as R  # noqa: E402
from microduck_local.robots import asimov_walk as W  # noqa: E402


class HoldPose(torch.nn.Module):
    """Nearest waypoint by joint distance, approached at a bounded rate."""

    def __init__(self, inv_slot, poses, acts, tau: float):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(inv_slot, dtype=torch.long))
        self.register_buffer("poses", torch.as_tensor(poses, dtype=torch.float32))
        self.register_buffer("acts", torch.as_tensor(acts, dtype=torch.float32))
        self.tau = float(tau)

    def forward(self, obs):
        jpos = obs[:, W.OBS_JOINT_POS[0]:W.OBS_JOINT_POS[1]].index_select(1, self.idx)
        # (n, 1, 23) - (1, k, 23) -> (n, k) mean |error| to every waypoint
        d = (jpos.unsqueeze(1) - self.poses.unsqueeze(0)).abs().mean(dim=2)
        w = torch.softmax(-d / self.tau, dim=1)          # (n, k), no dither
        return w @ self.acts


def main() -> None:
    if not asimov.ready():
        raise SystemExit("Asimov's model is not fetched (uv run fetch-robot asimov1)")
    slot = np.asarray(W.slot_order())
    inv = np.argsort(slot)                      # obs slot -> JOINT_NAMES order
    clip = 4.0                                  # RobotSpec's inherited default
    poses = np.stack([np.asarray(w.joints, np.float64) - W.DEFAULT_POSE
                      for w in R.WAYPOINTS])    # in jpos units, JOINT_NAMES order
    acts = np.clip(poses / W.ACTION_SCALE, -clip, clip)
    # well inside the closest pair's separation (rock_back/fold, 0.0246)
    tau = 0.01
    net = HoldPose(inv, poses, acts, tau).eval()

    out = Path(asimov.CACHE_DIR) / "hold_pose.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, W.OBS_DIM)
    torch.onnx.export(net, (dummy,), str(out), input_names=["obs"],
                      output_names=["action"], opset_version=17,
                      dynamo=False,   # the repo's own exporter (export_onnx.py)
                      dynamic_axes={"obs": {0: "n"}, "action": {0: "n"}})

    # it must reproduce the holding action for a REAL pose, not just zeros
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst_step, worst_settled = 0.0, 0.0
    for i, w in enumerate(R.WAYPOINTS):
        q = np.asarray(w.joints, np.float64) + rng.uniform(-0.05, 0.05, W.NUM_JOINTS)
        obs = np.zeros((1, W.OBS_DIM), np.float32)
        obs[0, W.OBS_JOINT_POS[0]:W.OBS_JOINT_POS[1]] = (q - W.DEFAULT_POSE)[slot]
        # near a waypoint the blend must be dominated by THAT waypoint
        got = sess.run(None, {"obs": obs})[0][0]
        err = float(np.abs(got - acts[i]).max())
        worst_settled = max(worst_settled, err)
        if err > 0.6:
            raise SystemExit(f"{w.name}: the blend is too soft to hold it "
                             f"({err:.2f} action units off its own command)")
        # and it must be CONTINUOUS: a small pose change moves it a little
        obs2 = obs.copy()
        obs2[0, W.OBS_JOINT_POS[0]:W.OBS_JOINT_POS[1]] += 0.01
        jump = float(np.abs(sess.run(None, {"obs": obs2})[0][0] - got).max())
        worst_step = max(worst_step, jump)
    print(f"wrote {out}")
    print(f"  78 -> 23, {len(R.WAYPOINTS)} waypoints, tau {tau}")
    print(f"  worst blend error at a waypoint {worst_settled:.3f} action units; "
          f"a 0.01 rad pose nudge moves the command at most {worst_step:.3f}")
    print(f"  the widest action it ever asks for: {np.abs(acts).max():.2f} "
          f"of a {clip} clip")


if __name__ == "__main__":
    main()
