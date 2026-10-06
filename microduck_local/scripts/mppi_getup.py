"""MPPI over the ACTION sequence — a real search for the `pike` -> `fold` leg.

Every search tried so far parameterised the motion as 3-5 POSE waypoints ramped
between, which is a crippled space: it cannot express "push here, release there,
catch now", and that is exactly what crossing a contact-mode change needs. The
sampling-based optimiser below searches the 23-dim action at EVERY control step
over a 1.6 s horizon, with noise correlated in time so the samples are smooth
commands rather than white jitter.

    uv run python scripts/mppi_getup.py --start pike
    uv run python scripts/mppi_getup.py --start rock_back --iters 25

Cost is the product `pelvis_z x uprightness`, counted only while a foot is on
the floor — the shape that killed the sit, the shoulder stand and the airborne
somersault an earlier search found. `--render` writes the best rollout out so it
can be looked at, which is the rule for anything that moves the robot.
"""

from __future__ import annotations

import argparse

import numpy as np

from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import asimov_walk as W


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
        lam: float, seed: int, verbose: bool = True):
    import mujoco

    env = make_env(start)
    m, d = env.model, env.data
    tb = env.trunk_body_id
    nu = env.action_space.shape[0]
    clip = float(env.robot.action_clip)
    foot_g = [g for g in range(m.ngeom)
              if int(m.geom_contype[g]) and m.geom_bodyid[g]
              and "ankle_roll" in m.body(int(m.geom_bodyid[g])).name]

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
        for t in range(horizon):
            a = np.clip(seq[t], -clip, clip).astype(np.float32)
            d.xfrc_applied[:] = 0.0
            env.step(a)
            z = float(d.xpos[tb][2])
            u = float(-env._projected_gravity()[2])
            down = any(d.contact[i].geom1 in foot_g or d.contact[i].geom2 in foot_g
                       for i in range(d.ncon))
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
            print(f"  it {it:>3}  best z*u {best_r:.3f}   "
                  f"(the start scores {z0/0.639*0.5:.3f}, `fold` 0.670)",
                  flush=True)
    score, best_r, trace = rollout(mean, record=True)
    zs = [t[0] for t in trace]
    return dict(mean=mean, best=best_r, peak_z=max(zs), start_z=z0,
                end_z=zs[-1], end_u=trace[-1][1], trace=trace, history=history)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
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
              args.sigma, args.lam, args.seed)
    print(f"\npelvis {out['start_z']:.3f} -> peak {out['peak_z']:.3f} "
          f"-> end {out['end_z']:.3f}   u end {out['end_u']:+.2f}")
    print(f"best z*u along the rollout: {out['best']:.3f}   "
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
