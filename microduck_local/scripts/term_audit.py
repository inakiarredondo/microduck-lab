"""AUDIT A REWARD ECONOMY BEFORE YOU TRAIN ON IT.

One command for the four checks that cost ~45 runs on the Asimov get-up,
each of which was done by hand, late, after the damage:

1. **Is the term DEAD here?** A monotone best-so-far term is identically zero
   wherever the policy drives its quantity backwards, which is exactly where
   a reverse curriculum puts it. `arm_push` earned **0.00 per episode at
   w=150**, `tuck` moved its own quantity 6 mm at w=450, `rotate` earned 0.5 %
   of positives at w=200. No weight saves a term reading ~0 AT THE RUNG.

2. **Is the term DOMINANT?** `rotate_work` at w=150 took 739 of an 873-point
   episode for ONE roll onto the back, which the policy then banked by lying
   still. A term over ~50 % of positives IS the objective, so check it is the
   one you meant.

3. **Is the TASK'S OWN objective being drowned?** With my proxies on,
   `stand+rise+progress` — the terms that define the task — were **2.0 %** of
   the positive budget. The policy was optimising my shaping, not the task.

4. **Does DOING NOTHING pay?** The frozen policy earned **+872.74** an
   episode; under the env's own objective it scores **-124.23**. If lying
   still is profitable, no training curve means anything. Episode length is
   printed with it, because a rung that ends in 8 steps cannot be taught at
   all (every Asimov rung below `fold` did).

Read off `env.reward_sums`, never a second `_compute_reward()` call — that
reads ZERO for every monotone term and correctly for per-step ones, which is
what once made a working term look dead.

    uv run python scripts/term_audit.py <run> [--rung route:0.00,0.12]
                                              [--set tuck2=80 --set rise_work=150]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: Terms whose value IS the task, as opposed to shaping for it. If these are
#: a small share of the positive budget, the economy is measuring something
#: other than the deliverable.
OBJECTIVE = ("stand", "rise", "progress", "standing", "up", "track")


def _load(run):
    sp = importlib.util.spec_from_file_location("_ga", HERE / "getup_acceptance.py")
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)
    return GA.load(run), GA


def _run_eps(env, pol, eps, seed0=0):
    """Sum `reward_sums` over episodes, and the episode lengths."""
    tot, lens = {}, []
    for s in range(seed0, seed0 + eps):
        o, _ = env.reset(seed=s)
        if pol is not None and hasattr(pol, "reset"):
            pol.reset()
        n = 0
        for _k in range(int(env.max_steps)):
            a = (np.zeros(env.action_space.shape, np.float32) if pol is None
                 else pol(o))
            o, _r, t, tr, _i = env.step(a)
            n += 1
            if t or tr:
                break
        lens.append(n)
        for k, v in (getattr(env, "reward_sums", None) or {}).items():
            tot[k] = tot.get(k, 0.0) + float(v)
    return {k: v / eps for k, v in tot.items()}, float(np.median(lens))


#: Where each term's WEIGHT lives on the env, so the audit can divide it out.
#:
#: Without this the audit could only report the WEIGHTED value, and its
#: "DEAD here" note then asserted something it had not checked: on
#: 2026-10-21 it printed `joint_vel_penalty -0.22  DEAD here — no weight
#: saves it` for a term whose weight is the class constant `W_JOINT_VEL`
#: (0.002). At `joint_vel_cost=9` the SAME rollouts price it at **-968**, so
#: a weight saved it completely. The robot had been jiggling at a median 326
#: deg/s while standing — against the shipped walker's 3 deg/s — and the one
#: instrument pointed at it said the charge was beyond rescue.
#:
#: "The quantity is identically zero here" and "the weight is three orders
#: of magnitude too small" are opposite diagnoses and were printing the same
#: sentence. A term is only unsaveable if the UNWEIGHTED quantity is ~0.
#: The largest weight worth considering. Above this a term is not being
#: tuned, it is being made the objective.
_SANE_WEIGHT = 1000.0

_WEIGHT_OF = {
    "stand": "W_STAND", "rise": "W_RISE", "standing": "W_STANDING",
    "progress": "W_PROGRESS", "head_up": "_w_head_up",
    "joint_vel_penalty": "_joint_vel_cost",
    "over_clip_penalty": "_over_clip_cost",
    "rise_vel_penalty": "_rise_vel_cost",
    "arm_effort_penalty": "_w_arm_effort",
    "action_rate_penalty": "W_ACTION_RATE",
    "home": "_home_pay", "calm_penalty": "_calm_cost",
    "balance_penalty": "_balance_cost", "limit_penalty": "_limit_cost",
    "span_penalty": "_span_cost", "swing": "_swing_pay",
    "rotate": "_rotate_pay", "tuck": "_tuck_pay",
    "roll": "_roll_pay", "on_back": "_roll_pay",
    "arm_push": "_w_arm_push", "arm_work": "_w_arm_work",
    "tuck_work": "_w_tuck_work", "tuck2": "_w_tuck2",
    "rotate_work": "_w_rotate_work", "rise_work": "_w_rise_work",
}


def _ramp_to_trained(env, run: str) -> float:
    """Price the RAMPED penalties where the TRAINER saw them, not at step 0.

    `_penalty_scale()` reads `_lifetime_steps` and walks
    `_PENALTY_STAGES` -- 0.05 at a fresh env, reaching **1.00** only after
    100,000 steps on THAT env instance. A freshly built audit env therefore
    priced every ramped penalty at **5 %** of its training value, silently.

    That cost a 20 M-step run on 2026-10-21: `joint_vel_penalty` was sized
    from an audit reading **-480 an episode** and the trainer actually
    charged **-9,601**, about 65 % of the policy's whole positive budget --
    a 20x error in the one number the launch was chosen on. The term was
    read correctly, at a scale nothing had asked for.

    A run with `envs` workers over `steps` total sees `steps / envs` per
    env, so that is the lifetime this sets. Returns the resulting scale so
    the caller can print it: a SHORT run really does train mostly at the
    gentle end, and hiding that behind a hardcoded 1.0 would be the same
    fault in the other direction.
    """
    import json

    try:
        rj = json.loads((Path("runs") / run / "run.json").read_text())
        per_env = int(rj.get("steps", 0)) // max(int(rj.get("envs", 1)), 1)
    except Exception:
        per_env = 0
    env._lifetime_steps = max(per_env, getattr(env, "_lifetime_steps", 0))
    try:
        return float(env._penalty_scale())
    except Exception:
        return float("nan")


def _weight(env, term: str) -> float | None:
    """The term's scalar weight, or None when this table does not know it."""
    attr = _WEIGHT_OF.get(term)
    if attr is None:
        return None
    w = getattr(env, attr, None)
    try:
        return float(w)
    except (TypeError, ValueError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--rung", default=None,
                    help="start_rung to audit AT (e.g. route:0.00,0.12). "
                         "The default is whatever the run recorded.")
    ap.add_argument("--label", default=None, help="getup_label instead of a rung")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="override an env kwarg, repeatable")
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args(argv)

    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs

    kw = dict(replay_kwargs(Path("runs") / a.run))
    task = kw.pop("task", None)
    for k in ("start_phase", "free", "track_tol", "ref_speed", "force_lag",
              "arrival", "cut_back"):
        kw.pop(k, None)
    if a.rung:
        kw["start_rung"] = a.rung
    if a.label:
        kw.pop("start_rung", None)
        kw["getup_label"] = a.label
    for kv in a.set:
        k, _, v = kv.partition("=")
        try:
            kw[k] = float(v)
        except ValueError:
            kw[k] = v
    kw.update(fell_cut=False, max_episode_s=8.0)
    task = task or ("getup" if kw.get("start_rung") else "getup_ref")
    env = S.get("asimov1").env_class(task)(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    pol, _ = _load(a.run)
    scale = _ramp_to_trained(env, a.run)

    where = kw.get("start_rung") or kw.get("getup_label") or "(recorded)"
    print(f"{a.run}   task={task}   at {where}   {a.episodes} eps")
    print(f"  ramped penalties priced at _penalty_scale() = {scale:.2f} "
          f"(the ramp this run reached; a fresh env is 0.05)")

    terms, ep_len = _run_eps(env, pol, a.episodes)
    pos = sum(v for v in terms.values() if v > 0) or 1.0
    print(f"\n  {'term':<22s}{'per episode':>13s}{'% of positives':>16s}"
          f"{'weight':>9s}{'unweighted':>12s}  note")
    saveable = []
    for k in sorted(terms, key=lambda x: -abs(terms[x])):
        v = terms[k]
        share = 100.0 * v / pos if v > 0 else 0.0
        w = _weight(env, k)
        raw = (v / w) if (w not in (None, 0.0)) else None
        note = ""
        if abs(v) < 0.01 * pos:
            # THE DISTINCTION THIS COLUMN EXISTS FOR. A term worth nothing
            # because its QUANTITY is zero here cannot be rescued; one worth
            # nothing because its WEIGHT is tiny is rescued by a number.
            # "Saveable" asks whether a SANE weight reaches 5 % of the
            # positives — not whether the unweighted quantity happens to be
            # large. `over_clip_penalty`'s quantity is -147 against 15,300 of
            # positives, which fails a 1 %-of-positives test and is rescued
            # completely by a weight of 5.
            if raw is not None and abs(raw) * _SANE_WEIGHT >= 0.05 * pos:
                note = (f"UNDERWEIGHTED — live quantity, weight {w:g}; "
                        f"x{0.05 * pos / abs(v):.0f} reaches 5 % of positives")
                saveable.append(k)
            elif raw is None and w == 0.0:
                note = "OFF — weight is 0.0"
            else:
                note = "DEAD here — no weight saves it"
        elif share > 50.0:
            note = "DOMINANT — this IS the objective"
        ws = f"{w:9.4g}" if w is not None else f"{'?':>9s}"
        rs = f"{raw:12.2f}" if raw is not None else f"{'-':>12s}"
        print(f"  {k:<22s}{v:13.2f}{share:15.1f}%{ws}{rs}  {note}")
    if saveable:
        print(f"\n  SAVEABLE BY A WEIGHT: {', '.join(saveable)}")

    obj = sum(terms.get(k, 0.0) for k in OBJECTIVE if terms.get(k, 0.0) > 0)
    print(f"\n  THE TASK'S OWN TERMS {OBJECTIVE}")
    print(f"    {obj:.1f} of {pos:.1f} positives = {100 * obj / pos:.1f} %"
          + ("   <-- DROWNED by shaping" if obj < 0.10 * pos else ""))

    # THE FLOOR. Zero action under the SAME economy.
    env2 = S.get("asimov1").env_class(task)(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    _ramp_to_trained(env2, a.run)      # the floor must see the SAME economy
    zero, zero_len = _run_eps(env2, None, a.episodes)
    znet, pnet = sum(zero.values()), sum(terms.values())
    print(f"\n  DO-NOTHING FLOOR (zero action, same economy)")
    print(f"    zero action {znet:9.2f} per episode, median {zero_len:.0f} steps")
    print(f"    this policy {pnet:9.2f} per episode, median {ep_len:.0f} steps")
    if znet >= pnet:
        print("    *** THE FLOOR WINS — the curve is measuring the economy ***")
    if znet > 0:
        print("    *** LYING STILL IS PROFITABLE — something must charge per step ***")
    if zero_len < 20:
        print(f"    *** {zero_len:.0f}-STEP EPISODES — nothing can be taught here; "
              "fix the termination or the start ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
