"""Search the FLOOR phase against the trained stand's OWN success.

    uv run python scripts/mppi_handoff.py --start pike \
        --policy runs/teach-asimov_getup_ref-1039ef-s8/policy.onnx

Why this objective and not height. There is no feet-only equilibrium anywhere
below standing: of 700 leg configurations, 106 hold 1 s and exactly ONE holds 6 s
— at 0.612 m, which IS standing. So the get-up cannot be staged. The floor phase
has to deliver the robot, in one dynamic move, INTO THE BASIN of a controller
that can only hold itself ~0.6 m up, and scoring that phase by its own pelvis
height asks it for a thing that does not exist. Every search before this one
(CEM twice, MPPI zero-centred, MPPI hold-centred, 8 PPO chains) maximised height
and stopped in the same place: 4 mm above the pike.

What the basin wants is measured: ROOT PITCH dominates and the leg pose barely
matters. At 34 deg of trunk pitch `1039ef-s8` reaches 0.577-0.596 m from EVERY
configuration between `fold`'s joints and the `pike`'s; at 86 deg it reaches 0.46
from all of them and ends inverted. Going from `fold`'s legs to the `pike`'s
moves the peak 19 mm.

So a sample is scored by what the POLICY does after the search hands off to it:
`--handoff` steps of searched action, then the ONNX drives to the end of the
horizon, and the score is the height the CONTROLLER reaches. That is the floor
phase's real job description, and nothing has ever optimised it.
"""

from __future__ import annotations

import argparse

import numpy as np


def make_env(start: str):
    from microduck_local.robots import spec as S
    return S.get("asimov1").env_class("getup_ref")(
        domain_rand=False, obs_noise=False, start_phase=start,
        track_tol=0.6, force_lag=True, free=True,
        getup_label="prone")


class Snap:
    """A MuJoCo state, saved and restored — MPPI replays from one state."""

    __slots__ = ("qpos", "qvel", "act", "time")

    def __init__(self, d):
        self.qpos, self.qvel = d.qpos.copy(), d.qvel.copy()
        self.act = d.act.copy() if d.act.size else None
        self.time = float(d.time)

    def into(self, m, d, mujoco):
        d.qpos[:] = self.qpos
        d.qvel[:] = self.qvel
        if self.act is not None and d.act.size:
            d.act[:] = self.act
        d.time = self.time
        mujoco.mj_forward(m, d)


def smooth_noise(rng, k, h, nu, sigma, beta=0.7):
    """Temporally CORRELATED noise: white jitter averages to nothing over a
    horizon and never produces a push."""
    eps = rng.normal(0.0, sigma, size=(k, h, nu))
    out = np.empty_like(eps)
    acc = np.zeros((k, nu))
    for t in range(h):
        acc = beta * acc + (1.0 - beta) * eps[:, t]
        out[:, t] = acc
    return out / max(1.0 - beta, 1e-6) * (1.0 - beta)


def run(start: str, iters: int, k: int, horizon: int, sigma: float,
        lam: float, seed: int, verbose: bool = True,
        policy: str | None = None, handoff: int = 40, settle: int = 50,
        objective: str = "rise"):
    import mujoco

    env = make_env(start)
    m, d = env.model, env.data
    tb = env.trunk_body_id
    nu = env.action_space.shape[0]
    clip = float(env.robot.action_clip)
    foot_g = [g for g in range(m.ngeom)
              if int(m.geom_contype[g]) and m.geom_bodyid[g]
              and "ankle_roll" in m.body(int(m.geom_bodyid[g])).name]

    sess = None
    if policy:
        import onnxruntime as ort
        sess = ort.InferenceSession(policy, providers=["CPUExecutionProvider"])

    env.reset(seed=seed)
    start_snap = Snap(d)
    z0 = float(d.xpos[tb][2])
    rng = np.random.default_rng(seed)
    # CENTRE THE SEARCH ON HOLDING THE START POSE, not on zero. Zero action
    # commands DEFAULT_POSE — standing — because the contract is
    # `ctrl = DEFAULT_POSE + ACTION_SCALE * action` with `use_default_offset`.
    # At the pike that is up to 1.649 rad (95 deg) from the joint the reset just
    # settled, so a zero-centred mean searches inside a whole-body lunge: 2 s of
    # action 0 collapses every floor rung to z 0.009. The runs of 2026-10-04
    # that reported "never lifted" from `rock_back`, `pike` and `quadruped` were
    # all centred there, and two of the three came back with a PEAK BELOW THE
    # START while `pike`'s best scored under its own starting score — a search
    # that could not match standing still, which is the signature of having no
    # neutral action to improve from. Holding the pose is a known constant:
    # `(pose - DEFAULT_POSE) / ACTION_SCALE`, clipped.
    hold = np.clip((np.array([d.qpos[m.joint(n).qposadr[0]]
                             for n in env.robot.joint_names])
                    - np.asarray(env.robot.default_pose)) / float(
                        np.asarray(env.robot.action_scale).max()), -clip, clip)
    mean = np.tile(hold, (horizon, 1)).astype(np.float64)

    def rollout(seq, record=False):
        start_snap.into(m, d, mujoco)
        total, best, trace = 0.0, 0.0, []
        obs = env._get_obs()
        for t in range(horizon):
            if sess is not None and t >= handoff + settle:
                break          # scored already; the rest is wasted inference
            if sess is not None and t >= handoff:
                # HANDED OFF: the controller drives, and what it achieves from
                # here is the score. The searched prefix is only judged by the
                # state it delivers.
                a = sess.run(None, {"obs": obs.reshape(1, -1).astype(np.float32)}
                             )[0][0].astype(np.float32)
            else:
                a = np.clip(seq[t], -clip, clip).astype(np.float32)
            d.xfrc_applied[:] = 0.0
            obs = env.step(a)[0]
            z = float(d.xpos[tb][2])
            u = float(-env._projected_gravity()[2])
            down = any(d.contact[i].geom1 in foot_g or d.contact[i].geom2 in foot_g
                       for i in range(d.ncon))
            if objective == "upright":
                # UPRIGHTNESS ALONE, with a foot down. The basin is governed by
                # trunk pitch and not by height — at 34 deg the trained stand
                # recovers from every leg configuration, at 86 deg from none —
                # so the question this objective asks is the only one left:
                # can the trunk get upright from here AT ALL? Height is
                # deliberately not in it; every previous search had height in
                # the objective and they all stopped at the same place.
                r = (u + 1.0) * 0.5 if down else 0.0
            else:
                r = (z / 0.639) * (u + 1.0) * 0.5 if down else 0.0
            total += r
            best = max(best, r)
            if record:
                trace.append((z, u, down))
        return total + 3.0 * best, best, trace

    history = []
    for it in range(1, iters + 1):
        noise = smooth_noise(rng, k, horizon, nu, sigma)
        seqs = mean[None, :, :] + noise
        costs = np.array([rollout(s)[0] for s in seqs])
        w = np.exp((costs - costs.max()) / max(lam, 1e-6))
        w /= w.sum()
        mean = np.einsum("k,khn->hn", w, seqs)
        score, best_r, _ = rollout(mean)
        history.append(best_r)
        if verbose and (it % 5 == 0 or it == 1):
            print(f"  it {it:>3}  best {objective} {best_r:.3f}   "
                  f"(the start scores {z0/0.639*0.5:.3f}, `fold` 0.670)",
                  flush=True)
    score, best_r, trace = rollout(mean, record=True)
    zs = [t[0] for t in trace]
    return dict(mean=mean, best=best_r, peak_z=max(zs), start_z=z0,
                end_z=zs[-1], end_u=trace[-1][1], trace=trace, history=history,
                # the SCORE'S OWN NAME travels with it. Printing "best z*u" for
                # an uprightness run is the mislabelled-metric mistake that
                # inverted a result earlier the same day.
                objective=objective)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default=None,
                    help="the trained stand that takes over at --handoff")
    ap.add_argument("--objective", default="rise", choices=("rise", "upright"),
                    help="`rise` = height x uprightness (what every earlier "
                         "search used); `upright` = uprightness alone, which "
                         "asks whether the trunk can come up at all")
    ap.add_argument("--settle", type=int, default=50,
                    help="control steps the POLICY gets after the handoff; the "
                         "score is what it reaches in them. The rest of the "
                         "horizon is wasted inference — at 200 samples that is "
                         "24k single-row ONNX calls an iteration.")
    ap.add_argument("--handoff", type=int, default=40,
                    help="control steps of SEARCHED action before the policy drives")
    ap.add_argument("--start", default="pike")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--samples", type=int, default=240)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="npy for the best sequence")
    args = ap.parse_args()

    print(f"MPPI from `{args.start}`: {args.samples} samples x {args.iters} "
          f"iterations, {args.horizon}-step ({args.horizon*0.02:.1f} s) horizon")
    out = run(args.start, args.iters, args.samples, args.horizon,
              args.sigma, args.lam, args.seed, policy=args.policy, handoff=args.handoff, settle=args.settle, objective=args.objective)
    print(f"\npelvis {out['start_z']:.3f} -> peak {out['peak_z']:.3f} "
          f"-> end {out['end_z']:.3f}   u end {out['end_u']:+.2f}")
    print(f"best {out['objective']} along the rollout: {out['best']:.3f}   "
          f"(`fold` is 0.670, `stand` 0.951)")
    rose = out["peak_z"] > out["start_z"] + 0.03
    print("VERDICT:", "THE PELVIS ROSE" if rose else
          "never lifted — same wall every other search hit")
    if args.save:
        np.save(args.save, out["mean"])
        print(f"saved the action sequence to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
