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
import pathlib
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
    ap.add_argument("--action-clip", type=float, default=13.0,
                    help="the collection env's clip. 13.0, not the spec's 4.0: "
                         "the teacher exceeds 4.0 on 15.5 %% of its frames and "
                         "0.0 %% past 13.0, so a 4.0 env truncates the student "
                         "rollouts while the labels it is fitted to do not "
                         "exist inside that clip. teach-asimov_getup_clip-0e31e5 "
                         "— the only run on this task that stands — used 13.0.")
    ap.add_argument("--post-stand-only", action="store_true",
                    help="on STUDENT iterations keep only the samples from "
                         "after the robot first stood. Plain DAgger aggregates "
                         "everything and HURT: 73 k mostly-floor student states "
                         "raised aggregate MSE 0.000176 -> 0.000470 and rise "
                         "fell 24/24 -> 20/24. The failure is not on the floor: "
                         "the clone's action error is 0.000702 rad^2 on the "
                         "rise and 0.000064 on the hold, and 29 of 36 episodes "
                         "stand and THEN fall. This collects where it breaks.")
    ap.add_argument("--pin-student", action="store_true",
                    help="roll out --student-from on EVERY student iteration "
                         "instead of the freshly fitted clone. Without this, "
                         "iterations 2+ collect on the clone they just fitted "
                         "— which on the first targeted run stood 1/40 and "
                         "9/40, so DAgger went straight back to aggregating a "
                         "policy that cannot stand (clone rise 16/24 against "
                         "the donor's 35/36). The distribution worth "
                         "correcting is the DEPLOYED one, and it does not "
                         "change between iterations.")
    ap.add_argument("--student-from", default=None,
                    help="seed the first student-driven iteration from an "
                         "existing run's model instead of the fresh clone. The "
                         "distribution that matters is the DEPLOYED policy's, "
                         "and the fine-tuned policy is not the clone.")
    a = ap.parse_args()

    sys.path.insert(0, str(THEIRS))
    import mujoco
    import run_policy_mujoco as R

    from microduck_local import distill
    from microduck_local.robots import asimov_walk as W
    from microduck_local.robots import spec as S

    spec = json.loads((THEIRS / "policies/getup_policy_spec.json").read_text())
    # THE COLLECTION ENV MUST CARRY THE STUDENT'S OWN FLAGS. Built with bare
    # defaults it is a different env from the one the student trained in, and
    # the student then fails in it for reasons that have nothing to do with
    # the policy: teach-asimov_getup_clip-0e31e5 scores 34/36 rose and 26/36
    # still up under its recorded flags, and stood **0 of 8** here before this
    # was fixed — which, with --post-stand-only, silently collected NOTHING.
    # `getup_report.py` states the rule: evaluating a policy under flags it did
    # not train with does not look like an error, it looks like a result.
    env_kw = dict(domain_rand=False, obs_noise=False, seed=a.seed)
    if a.student_from:
        _meta = json.loads(
            (HERE / "runs" / a.student_from / "run.json").read_text())
        for _k, _v in (_meta.get("env_kwargs") or {}).items():
            if _k in ("track_tol", "history", "force_lag", "getup_label",
                      "getup_blend", "getup_assist", "ref_speed", "start_phase"):
                env_kw[_k] = _v
        print(f"collection env carries {a.student_from}'s flags: "
              f"{ {k: v for k, v in env_kw.items() if k not in ('seed',)} }",
              flush=True)
    env_kw["action_clip"] = a.action_clip
    env = S.get("asimov1").env_class("getup_clip")(**env_kw)
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

    def load_student(run_dir: pathlib.Path):
        """A policy driven the way DEPLOYMENT drives it.

        `PPO.load(...).predict(obs)` feeds the net RAW observations, and
        `distill.fit` trains the student on the NORMALISED view
        (`export_onnx` bakes the stats in later). Measured on
        teach-asimov_getup_clip-0e31e5, which stands 26/36 under verify_run:

            PPO.predict on raw obs                      stood 0/8
            policy.onnx (normaliser baked in)           stood 8/8
            PPO.predict + VecNormalize stats by hand    stood 8/8

        So every student rollout this script has ever collected was taken on
        an unnormalised policy — states no deployment visits. That is almost
        certainly why plain DAgger "hurt" (aggregate MSE 0.000176 -> 0.000470,
        rise 24/24 -> 20/24): it was aggregating a broken policy's states, not
        a student's. DAgger has not actually been tested on this task.
        """
        import pickle

        from stable_baselines3 import PPO as _PPO
        net = _PPO.load(str(run_dir / "model"), device="cpu")
        with open(run_dir / "vecnormalize.pkl", "rb") as fh:
            vn = pickle.load(fh)
        mu, va, cl, eps = vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs, vn.epsilon

        def predict(o):
            n = np.clip((o - mu) / np.sqrt(va + eps), -cl, cl).astype(np.float32)
            return net.predict(n, deterministic=True)[0]
        return predict

    OBS, ACT = [], []
    student = None
    if a.student_from:
        print(f"iteration 0 collects on the TEACHER as always — it is the base "
              f"the clone is fitted from, and a post-stand filter applied to a "
              f"student iteration 0 can keep almost nothing. {a.student_from} "
              f"drives from iteration 1.", flush=True)
    for it in range(a.dagger_iters):
        n_over, n_tot, stood, n_kept = 0, 0, 0, 0
        for ep in range(a.episodes_per_iter):
            env.reset(seed=int(rng.integers(1 << 30)))
            if student is None:
                # TEACHER rollout: the physics is stepped by hand below, no
                # env cut can fire, so picking the start directly is harmless
                # and this is the path that produced the working clone.
                i = int(rng.integers(len(env._clip_starts)))
                d.qpos[:] = env._clip_starts[i]
                d.qvel[:] = 0.0
                d.xfrc_applied[:] = 0.0
                mujoco.mj_forward(m, d)
            # STUDENT rollout: do NOT override qpos. `reset` already lands on a
            # clip start AND sets the reference phase to match it; writing a
            # different start in afterwards leaves the two desynchronised, the
            # track_tol cut fires on step 1, and 8 episodes yielded 8 samples
            # with the student "never standing". It stands 26/36 under its own
            # flags — the harness was the whole story.
            pol.reset()
            obs = env._get_obs()
            stood_yet = False
            for k in range(a.steps):
                pol.step(d)                      # teacher sees the true state
                lab = teacher_action()
                # TARGETED DAgger. On a student rollout the floor states are
                # the ones that diluted the set last time; the failure lives
                # AFTER the stand. Teacher-driven iteration 0 is always kept
                # whole — it is the base the clone is built from.
                zz = float(d.xpos[env.trunk_body_id][2])
                uu = float(d.xmat[torso].reshape(3, 3)[2, 2])
                stood_yet = stood_yet or (zz > 0.55 and uu > 0.9)
                keep = (student is None) or (not a.post_stand_only) or stood_yet
                if keep:
                    OBS.append(obs.copy()); ACT.append(lab.copy())
                    n_kept += 1
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
                    sa = student(obs)
                    obs, _r, term, trunc, _i = env.step(np.asarray(sa, np.float32))
                    # Roll THROUGH the curriculum cuts, exactly as
                    # `verify_run._score_with` does, and stop only once the
                    # robot has stood and then gone down. The track_tol cut is
                    # a training device that "does not exist on a robot";
                    # honouring it here ended episodes after ~13 steps, so the
                    # student "never stood" and a post-stand filter collected
                    # nothing — from the very policy that stands 26/36.
                    if (term or trunc) and bool(getattr(env, "_stood", False)):
                        break
            z = float(d.xpos[env.trunk_body_id][2])
            u = float(d.xmat[torso].reshape(3, 3)[2, 2])
            stood += int(z > 0.55 and u > 0.9)
        who = "teacher" if student is None else "student"
        if a.post_stand_only and student is not None:
            print(f"[iter {it}] post-stand filter kept {n_kept} of {n_tot} "
                  f"samples ({100*n_kept/max(n_tot,1):.1f} %)", flush=True)
            if n_kept == 0:
                print(f"[iter {it}] *** the student never stood — the filter "
                      "kept NOTHING and this iteration adds no data ***",
                      flush=True)
        print(f"[iter {it}] collected on the {who}'s states: {n_tot} pairs, "
              f"{stood}/{a.episodes_per_iter} ended standing; "
              f"labels past the env's +/-{env._act_clip:g} clip: "
              f"{100*n_over/max(n_tot,1):.1f} %",
              flush=True)

        # float32: `distill.fit` builds torch tensors straight from these and
        # a float64 label raises "Found dtype Double but expected Float".
        obs_a = np.asarray(OBS, np.float32)
        act_a = np.asarray(ACT, np.float32)
        if obs_a.ndim != 2 or len(obs_a) < 256:
            raise SystemExit(
                f"[iter {it}] only {len(obs_a)} samples aggregated — refusing "
                "to fit. With --post-stand-only this means the student almost "
                "never stood, so there is nothing to correct AT the stand; "
                "collect more episodes or drop the filter. (Fitting anyway "
                "crashes inside dead_slots with a scalar variance, which is "
                "how this was found.)")
        out = HERE / "runs" / (a.run_name if it == a.dagger_iters - 1
                               else f"{a.run_name}-it{it}")
        # `twist_cmd` is zero for a get-up on purpose — nothing is commanded.
        # Every OTHER flat slot is a harness bug, and `dead_slots` raises on it.
        mse = distill.fit(obs_a, act_a, out, epochs=a.epochs, seed=a.seed,
                          robot="asimov1", task="getup_clip",
                          allow_dead=("twist_cmd",))
        print(f"[iter {it}] fitted {len(obs_a)} pairs -> {out.name}  "
              f"action mse {mse:.6f} rad^2", flush=True)
        if a.student_from and (it == 0 or a.pin_student):
            # the distribution that matters is the DEPLOYED policy's, and the
            # fine-tuned policy is not the fresh clone
            student = load_student(HERE / "runs" / a.student_from)
        else:
            student = load_student(out)

    print(f"\ndone: runs/{a.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
