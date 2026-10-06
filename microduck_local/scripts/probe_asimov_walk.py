"""Does Asimov 1's SHIPPED policy walk in this repo's MuJoCo?

    uv run python scripts/probe_asimov_walk.py

The check that decides whether `robots/asimov_walk.py`'s 78-float layout is
their layout. A wrong permutation still produces actions and still moves the
robot; it just falls over, and every measurement taken afterwards describes a
different robot. Their own policy walking here is the only evidence that the
contract was copied and not invented.

Deterministic on purpose: no observation noise, no action delay, one spawn
pose (`AGENTS.md`, "Verify policies deterministically").
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from microduck_local.robots import asimov_walk as W


def run(cmd, seconds=10.0, friction=True, delay=0, noise=False, seed=0):
    model = W.walk_model(True, friction)
    data = mujoco.MjData(model)
    walker = W.AsimovWalker(model, policy=str(W.policy_onnx()),
                            obs_noise=noise, action_delay=delay, seed=seed)
    walker.spawn(data)
    walker.cmd[:] = cmd
    mujoco.mj_forward(model, data)
    n = int(seconds / W.CTRL_DT)
    zs, fell_at = [], None
    # UNWRAPPED, because the wrapped one lies: a 0.6 rad/s turn held for 10 s
    # is 4.79 rad, which prints as -1.49 and reads like a stall.
    prev_yaw = walker.pose(data)[2]
    yaw_total = 0.0
    torque_peak = np.zeros(W.NUM_JOINTS)
    for i in range(n):
        walker.step(data)
        zs.append(float(data.xpos[walker.pelvis][2]))
        torque_peak = np.maximum(torque_peak,
                                 np.abs(data.actuator_force[walker.act_id]))
        yaw_now = walker.pose(data)[2]
        yaw_total += (yaw_now - prev_yaw + np.pi) % (2 * np.pi) - np.pi
        prev_yaw = yaw_now
        if fell_at is None and walker.fallen(data):
            fell_at = i * W.CTRL_DT
    x, y, yaw = walker.pose(data)
    return {
        "cmd": tuple(cmd), "x": x, "y": y, "yaw": yaw,
        "yaw_rate": yaw_total / seconds,
        "dist": float(np.hypot(x, y)),
        "speed": float(np.hypot(x, y) / seconds),
        "z_mean": float(np.mean(zs)), "z_min": float(np.min(zs)),
        "fell_at": fell_at, "torque_peak": torque_peak,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--no-friction", action="store_true",
                    help="drop the joint frictionloss (the one interpreted term)")
    args = ap.parse_args(argv)

    if not W.policy_ready():
        print("policy not fetched — run "
              "`uv run python -m microduck_local.robots.asimov_walk --fetch`",
              file=sys.stderr)
        return 1

    cmds = [("stand", (0.0, 0.0, 0.0)),
            ("forward 0.5", (0.5, 0.0, 0.0)),
            ("forward 0.8", (0.8, 0.0, 0.0)),
            ("back 0.4", (-0.4, 0.0, 0.0)),
            ("left 0.3", (0.0, 0.3, 0.0)),
            ("turn 0.6", (0.0, 0.0, 0.6))]
    print(f"Asimov 1 shipped policy — {W.POLICY_REPO} @ {W.POLICY_SHA[:12]}")
    print(f"{args.seconds:.0f} s per command, deterministic "
          f"(no obs noise, no action delay), joint friction "
          f"{'ON' if not args.no_friction else 'OFF'}\n")
    print(f"{'command':<12} {'x (m)':>7} {'y (m)':>7} {'rad/s':>7} "
          f"{'speed m/s':>9} {'pelvis z':>9} {'fell':>8}")
    peak = np.zeros(W.NUM_JOINTS)
    for label, cmd in cmds:
        r = run(cmd, args.seconds, friction=not args.no_friction)
        peak = np.maximum(peak, r["torque_peak"])
        fell = "-" if r["fell_at"] is None else f"{r['fell_at']:.1f}s"
        print(f"{label:<12} {r['x']:>7.2f} {r['y']:>7.2f} {r['yaw_rate']:>7.3f} "
              f"{r['speed']:>9.2f} {r['z_mean']:>9.3f} {fell:>8}")

    print("\npeak |torque| vs the motor's rated limit, worst joint per group:")
    seen = set()
    for i, n in enumerate(W.JOINT_NAMES):
        frag = next(f for f, *_ in W.GAIN_GROUPS if f in n)
        if frag in seen:
            continue
        seen.add(frag)
        same = [j for j, m in enumerate(W.JOINT_NAMES) if frag in m]
        _, _, effort, _, _ = W.gains_for(n)
        worst = float(np.max(peak[same]))
        print(f"  {frag:<16} {worst:6.1f} / {effort:5.1f} N*m "
              f"({100 * worst / effort:5.1f} %)"
              + ("   AT THE CLAMP" if worst >= effort * 0.995 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
