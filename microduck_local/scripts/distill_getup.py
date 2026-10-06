"""Distil openhorizon-labs' 375-input get-up into OUR 78-input contract.

    uv run --with onnxruntime python scripts/distill_getup.py --run-name asimov-getup-distil

Why distillation and not more RL. Eight PPO chains, CEM twice, MPPI three ways
and ~50 M steps all failed, and every failure traces to exploration or reward
design: the `standing` gate paid 0.0 for 9.1 M steps because the state was never
sampled, the free reward is monotone in height so nothing pays for leaving a
stable pose to launch, and three ladder designs failed because their rungs are
not holdable. Distillation has none of those: no reward, no gate, no ladder, no
exploration. It is supervised regression onto a teacher that is available AT
EVERY STATE.

And the teacher is proven HERE: their policy driven through OUR position
actuators in OUR env stands **8/8** from the demonstration's own fallen poses
(0.045-0.167 m -> 0.619 m, upright +1.00). That retires every structural
conclusion in the ledger — they described our route, not the robot.

Why DAgger and not plain cloning. Replaying the demonstration's joint path
open-loop through our servos gets **0/8**: the motion is not open-loop stable, so
the instant the student drifts, a recording is the wrong answer for where it
actually is. Iteration 0 collects on the TEACHER's states; every later iteration
rolls the STUDENT out and asks the teacher what it would have done there, which
is the only part that fixes that.

THE OPEN QUESTION this answers. The teacher reads 375 inputs — five stacked
frames — and the student reads 78, one frame, which is Menlo's deployable
contract. If one frame does not carry enough to reproduce a 1.2 s ballistic
get-up, the student plateaus at some action error and falls, and that plateau IS
the measurement nobody has: what the history buys. Either outcome is worth
having.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
THEIRS = HERE / ".cache/openhorizon_getup"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--dagger-iters", type=int, default=4)
    ap.add_argument("--episodes-per-iter", type=int, default=30)
    ap.add_argument("--steps", type=int, default=260, help="control steps an episode")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(THEIRS))
    import mujoco
    import run_policy_mujoco as R

    from microduck_local import distill
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    spec = json.loads((THEIRS / "policies/getup_policy_spec.json").read_text())
    env = S.get("asimov1").env_class("getup_clip")(
        domain_rand=False, obs_noise=False, seed=a.seed)
    m, d = env.model, env.data
    pol = R.Policy(str(THEIRS / "policies/getup.onnx"), spec, m)

    # THE PERMUTATION. Their action order and ours must agree or every label is
    # a different robot's. A wrong permutation still produces actions and still
    # moves the robot; it just falls over (AGENTS.md).
    assert list(pol.names) == list(W.JOINT_NAMES), (
        "their action joint_order is not ours — relabel before trusting any of "
        f"this: theirs[:3]={list(pol.names)[:3]} ours[:3]={list(W.JOINT_NAMES)[:3]}")

    decim = round(1.0 / spec["rate_hz"] / m.opt.timestep)
    act_id = np.array([m.actuator(n).id for n in W.JOINT_NAMES])
    torso = m.body("waist_yaw_link").id
    rng = np.random.default_rng(a.seed)

    def teacher_action() -> np.ndarray:
        """Their q_target, expressed in OUR action convention."""
        return (pol.target - W.DEFAULT_POSE) / W.ACTION_SCALE

    OBS, ACT = [], []
    student = None
    for it in range(a.dagger_iters):
        n_over, n_tot, stood = 0, 0, 0
        for ep in range(a.episodes_per_iter):
            env.reset(seed=int(rng.integers(1 << 30)))
            i = int(rng.integers(len(env._clip_starts)))
            d.qpos[:] = env._clip_starts[i]
            d.qvel[:] = 0.0
            d.xfrc_applied[:] = 0.0
            mujoco.mj_forward(m, d)
            pol.reset()
            obs = env._get_obs()
            for k in range(a.steps):
                pol.step(d)                      # teacher sees the true state
                lab = teacher_action()
                OBS.append(obs.copy()); ACT.append(lab.copy())
                n_tot += 1
                n_over += int(np.abs(lab).max() > env._act_clip)
                if student is None:              # iteration 0: TEACHER drives
                    # `last_action` occupies obs[55:78], and `_get_obs` reads it
                    # off the ENV. Stepping the physics by hand without writing
                    # it left all 23 slots at zero for every sample — measured
                    # in the normalizer as var 1e-8, mean 0.0, 23 of 23 dead —
                    # so the student learned a 55-float observation and then met
                    # 78 floats at deployment, 23 of them up to +/-12. The clone
                    # stood 23/24 when driven with the slot still zeroed and
                    # 0/24 the moment it carried a real action. Keep the env's
                    # bookkeeping in the order `walk_env.step` does it: the obs
                    # at step k carries the action taken at k-1.
                    env.prev_action = env.last_action
                    env.last_action = lab.astype(np.float32).copy()
                    for _ in range(decim):
                        d.qfrc_applied[:] = 0.0
                        d.ctrl[act_id] = pol.target
                        mujoco.mj_step(m, d)
                    obs = env._get_obs()
                else:                            # DAgger: the STUDENT drives
                    sa, _ = student.predict(obs, deterministic=True)
                    obs, _r, term, trunc, _i = env.step(np.asarray(sa, np.float32))
                    if term or trunc:
                        break
            z = float(d.xpos[env.trunk_body_id][2])
            u = float(d.xmat[torso].reshape(3, 3)[2, 2])
            stood += int(z > 0.55 and u > 0.9)
        who = "teacher" if student is None else "student"
        print(f"[iter {it}] collected on the {who}'s states: {n_tot} pairs, "
              f"{stood}/{a.episodes_per_iter} ended standing; "
              f"labels past the env's +/-{env._act_clip:g} clip: "
              f"{100*n_over/max(n_tot,1):.1f} %",
              flush=True)

        # float32: `distill.fit` builds torch tensors straight from these and
        # a float64 label raises "Found dtype Double but expected Float".
        obs_a = np.asarray(OBS, np.float32)
        act_a = np.asarray(ACT, np.float32)
        out = HERE / "runs" / (a.run_name if it == a.dagger_iters - 1
                               else f"{a.run_name}-it{it}")
        # `twist_cmd` is zero for a get-up on purpose — nothing is commanded.
        # Every OTHER flat slot is a harness bug, and `dead_slots` raises on it.
        mse = distill.fit(obs_a, act_a, out, epochs=a.epochs, seed=a.seed,
                          robot="asimov1", task="getup_clip",
                          allow_dead=("twist_cmd",))
        print(f"[iter {it}] fitted {len(obs_a)} pairs -> {out.name}  "
              f"action mse {mse:.6f} rad^2", flush=True)
        from stable_baselines3 import PPO
        student = PPO.load(str(out / "model"), device="cpu")

    print(f"\ndone: runs/{a.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
