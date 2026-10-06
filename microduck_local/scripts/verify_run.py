"""Score a finished run the way it has to be scored, so a training curve
cannot be mistaken for a result.

    uv run python scripts/verify_run.py runs/<run>            # every checkpoint
    uv run python scripts/verify_run.py runs/<run> --episodes 72
    uv run python scripts/verify_run.py runs/<run> --json out.json

EVERY GUARD HERE IS A MISTAKE THAT WAS MADE, on 2026-10-05/06, on this one
task, and each was reported as a finding before being caught:

1. **A training curve read as a result.** `ep_rew` climbed 216 -> 2942 and
   `ep_len` 33 -> 223 across a 2 M-step run while the deterministic rise fell
   24/24 -> 15/24. The reward was paying a fallen robot more than a standing
   one, and every training number improved as the policy got worse.
   -> This scores the DETERMINISTIC policy and never reads `ep_rew`.

2. **The last checkpoint scored as the run's outcome.** 4.5 M scored 4/12 and
   5.0 M scored 0/12; `policy.onnx` is exported from the last one, so the
   artefact on disk was the worse policy. Another chain's best stage was its
   first of four.
   -> Scores EVERY checkpoint, names the best, and says loudly when the best
   is not the last.

3. **A stale export scored as a fresh one.** The scoring scratch directory was
   keyed by step count alone, so run 3 was scored from run 2's exports and
   reported run 2's numbers to two decimal places.
   -> Keyed by (run, step), and the policy's content hash is printed so two
   identical policies are impossible to miss. That is how a byte-identical
   A/B was caught: an added penalty had been a dead hook.

4. **Evaluating under the curriculum's own cuts.** Honouring the training
   env's `track_err` and stall terminations scored a rise in progress as a
   failure: 12/24 against 23/24 for the same rollouts at the buzzer. A cut
   fired in 24 of 24 episodes at a median of step 352, mid-rise.
   -> Runs to the buzzer. The FALL check is kept (that one is real); the
   curriculum's cuts are reported as "cut fired" and not scored.

5. **Evaluating under flags the run did not train with.** `replay_kwargs`
   exists for this and was missing five knobs for a day, including
   `action_clip` — a replay of a 13.0-clip policy at the inherited 4.0 cannot
   command two thirds of the motion.
   -> Builds the env from `replay_kwargs(run)`, prints what it used, and fails
   if the run recorded nothing.

6. **A reward economy nobody priced.** Doing nothing out-earned rising 2775 to
   788 for a whole 2 M-step run.
   -> Prices the zero-action policy in the same env, every time.

7. **Numbers quoted past what the instrument can resolve.** 44/72 vs 40/72 was
   reported as an improvement; paired it was p = 0.424.
   -> Prints the episode count and the smallest difference it can see, and
   compares against the run's donor PAIRED on the same episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))


def _checkpoints(run: Path) -> list[int]:
    ck = run / "checkpoints"
    if not ck.is_dir():
        return []
    return sorted(int(p.stem.split("_")[1])
                  for p in ck.glob("model_*_steps.zip"))


def _export(run: Path, step: int | None, work: Path) -> Path:
    """Export one checkpoint to a (run, step)-keyed directory. Never reuses."""
    from microduck_local.export_onnx import export

    tag = "final" if step is None else str(step)
    out = work / run.name / tag
    out.mkdir(parents=True, exist_ok=True)
    if step is None:
        shutil.copy(run / "model.zip", out / "model.zip")
        shutil.copy(run / "vecnormalize.pkl", out / "vecnormalize.pkl")
    else:
        ck = run / "checkpoints"
        shutil.copy(ck / f"model_{step}_steps.zip", out / "model.zip")
        shutil.copy(ck / f"model_vecnormalize_{step}_steps.pkl",
                    out / "vecnormalize.pkl")
    shutil.copy(run / "run.json", out / "run.json")
    onnx = out / "policy.onnx"
    if onnx.exists():
        onnx.unlink()            # a stale artefact is not a fresh one
    export(out, onnx)
    return onnx


def _make_env(run: Path, deployment: bool, seed: int):
    """The env the policy TRAINED under, plus the deployment conditions.

    `obs_noise` and `action_delay` are deliberately not in `replay_kwargs` — a
    replay wants the clean motion. A VERIFICATION wants the opposite: a stand
    that needs zero sensor noise and zero latency is not a stand that ships,
    and this robot's own env says so (`AsimovGetupRefEnv.LAG_ENV`: every
    episode without the 20 ms delay held 7.76 s and every episode with it fell
    at 1.3-2.0 s, perfectly separated).
    """
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs

    kw = dict(replay_kwargs(run))
    if not kw:
        raise SystemExit(
            f"{run} recorded no env_kwargs — there is no honest way to score "
            "it. Runs written before `train_env_kwargs` are in this state; "
            "retrain rather than guess the flags.")
    task = kw.pop("task", None)
    rj = json.loads((run / "run.json").read_text())
    robot = rj.get("robot") or "microduck"
    body = S.get(robot)
    cls = body.env_class(task) if task else body.env_class(body.default_task)
    kw.setdefault("max_episode_s", None)
    kw = {k: v for k, v in kw.items() if v is not None}
    return cls(domain_rand=False, obs_noise=bool(deployment),
               action_delay=bool(deployment), seed=seed, **kw), kw


def _score(onnx: Path | None, run: Path, episodes: int, deployment: bool,
           seeds=(0, 1, 2)):
    """Rise, hold and the cuts, for a policy or (onnx=None) for ZERO ACTION."""
    import onnxruntime as ort

    sess = (ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            if onnx is not None else None)
    per_seed = max(1, episodes // len(seeds))
    rose = atbuz = cut = fell_n = 0
    holds, peaks, rets, n = [], [], [], 0
    used = None
    for sd in seeds:
        env, used = _make_env(run, deployment, sd)
        stand_z = float(getattr(env, "_stand_z", None)
                        or getattr(env, "stand_z", 0.639))
        for ep in range(per_seed):
            obs, _ = env.reset(seed=sd * 1000 + ep)
            best = float(env.data.xpos[env.trunk_body_id][2])
            streak = cur = 0
            last = False
            total = 0.0
            fired = fell = False
            for _ in range(int(env.max_steps)):
                if sess is None:
                    a = np.zeros(env.nj, np.float32)
                else:
                    a = sess.run(
                        None, {"obs": obs.reshape(1, -1).astype(np.float32)}
                    )[0][0]
                obs, r, term, trunc, _i = env.step(np.asarray(a, np.float32))
                total += float(r)
                z = float(env.data.xpos[env.trunk_body_id][2])
                u = float(-env._projected_gravity()[2])
                best = max(best, z)
                # The stand gate, off the MODEL's own standing height rather
                # than a hard 0.55 — a number carried across bodies is a
                # number that describes the wrong robot.
                last = (z > 0.86 * stand_z) and (u > 0.85)
                cur = cur + 1 if last else 0
                streak = max(streak, cur)
                if (term or trunc) and not fired:
                    fired = True
                    # A FALL is real; a curriculum cut is not. `_stood` is what
                    # arms the fall check, so a termination after standing is
                    # the robot going down and the episode is genuinely over.
                    # A termination BEFORE it ever stood is `track_err`, the
                    # stall deadline or the floor cut — curriculum devices that
                    # do not exist on a robot, and honouring them scored a rise
                    # in progress as a failure (12/24 against 23/24 for the
                    # same rollouts).
                    if bool(getattr(env, "_stood", False)):
                        fell = True
                        break
                    continue
            rose += int(streak > 0)
            atbuz += int(last)
            cut += int(fired and not fell)
            fell_n += int(fell)
            holds.append(streak / 50.0)
            peaks.append(best)
            rets.append(total)
            n += 1
    return {
        "n": n, "rose": rose, "at_buzzer": atbuz, "cut_fired": cut,
        "fell_after_standing": fell_n,
        "hold_median_s": float(np.median(holds)),
        "hold_max_s": float(np.max(holds)),
        "peak_median_m": float(np.median(peaks)),
        "return_median": float(np.median(rets)),
        "env_kwargs": used,
    }


def _score_with(onnx: Path, run: Path, kwargs: dict, episodes: int,
                seeds=(0, 1, 2)):
    """`_score`, with the env kwargs given rather than read off the run."""
    import onnxruntime as ort

    from microduck_local.robots import spec as S

    rj = json.loads((run / "run.json").read_text())
    body = S.get(rj.get("robot") or "microduck")
    task = (json.loads((run / "run.json").read_text()).get("task")
            or body.default_task)
    cls = body.env_class(task)
    kw = {k: v for k, v in kwargs.items() if v is not None and k != "task"}
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    per = max(1, episodes // len(seeds))
    rose = buz = 0
    holds, peaks, n = [], [], 0
    for sd in seeds:
        env = cls(domain_rand=False, obs_noise=True, action_delay=True,
                  seed=sd, **kw)
        for ep in range(per):
            obs, _ = env.reset(seed=sd * 1000 + ep)
            best = float(env.data.xpos[env.trunk_body_id][2])
            cur = streak = 0
            last = False
            for _ in range(int(env.max_steps)):
                a = sess.run(
                    None, {"obs": obs.reshape(1, -1).astype(np.float32)})[0][0]
                obs, _r, t, tr, _i = env.step(np.asarray(a, np.float32))
                z = float(env.data.xpos[env.trunk_body_id][2])
                u = float(-env._projected_gravity()[2])
                best = max(best, z)
                last = z > 0.55 and u > 0.85
                cur = cur + 1 if last else 0
                streak = max(streak, cur)
                if (t or tr) and bool(getattr(env, "_stood", False)):
                    break
            rose += int(streak > 0)
            buz += int(last)
            holds.append(streak / 50.0)
            peaks.append(best)
            n += 1
    return {"n": n, "rose": rose, "at_buzzer": buz,
            "hold_median_s": float(np.median(holds)),
            "hold_max_s": float(np.max(holds)),
            "peak_median_m": float(np.median(peaks))}


def _sign_test(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided sign test on the discordant pairs."""
    bo = sum(1 for x, y in zip(a, b) if (not x) and y)
    ao = sum(1 for x, y in zip(a, b) if x and not y)
    m = ao + bo
    if m == 0:
        return bo, ao, 1.0
    k = min(ao, bo)
    p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)
    return bo, ao, p


def _mde(n: int, p: float = 0.5) -> float:
    """Roughly the smallest difference in proportion this n can see (95/80)."""
    return 2.8 * float(np.sqrt(max(p * (1 - p), 0.01) / max(n, 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--episodes", type=int, default=36,
                    help="total episodes per policy (split across 3 seeds)")
    ap.add_argument("--clean", action="store_true",
                    help="score WITHOUT obs noise and the action delay too")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=None)
    a = ap.parse_args()

    run = a.run if a.run.is_dir() else a.run.parent
    work = a.work or (HERE / ".cache" / "verify")
    steps = _checkpoints(run)
    if not steps and not (run / "model.zip").is_file():
        raise SystemExit(f"{run}: no checkpoints and no model.zip")

    rows = []
    print(f"verifying {run.name}   {len(steps)} checkpoint(s) + final, "
          f"{a.episodes} episodes each, WITH obs noise and the action delay")
    base = _score(None, run, a.episodes, deployment=True)
    print(f"  env from replay_kwargs: {base['env_kwargs']}")
    print(f"  zero-action baseline  : return {base['return_median']:.1f}, "
          f"rose {base['rose']}/{base['n']}   "
          f"<- every number below must beat this\n")

    seen: dict[str, str] = {}
    hdr = (f"{'checkpoint':>11} {'rose':>8} {'up at end':>10} {'hold med':>9} "
           f"{'hold max':>9} {'peak m':>8} {'return':>9} {'cut':>6}  hash")
    print(hdr)
    for step in list(steps) + [None]:
        onnx = _export(run, step, work)
        h = hashlib.md5(onnx.read_bytes()).hexdigest()[:8]
        r = _score(onnx, run, a.episodes, deployment=True)
        r["step"] = step
        r["hash"] = h
        rows.append(r)
        dup = seen.get(h)
        seen[h] = "final" if step is None else str(step)
        tag = "final" if step is None else f"{step:,}"
        print(f"{tag:>11} {r['rose']:4d}/{r['n']:<3d} {r['at_buzzer']:6d}/{r['n']:<3d} "
              f"{r['hold_median_s']:8.2f}s {r['hold_max_s']:8.2f}s "
              f"{r['peak_median_m']:8.3f} {r['return_median']:9.1f} "
              f"{r['cut_fired']:3d}/{r['n']:<3d}  {h}"
              + (f"  == {dup} (IDENTICAL POLICY)" if dup else ""))

    # THE VERDICT, and the three things that have to be said out loud.
    print()
    best = max(rows, key=lambda r: (r["at_buzzer"], r["rose"],
                                    r["hold_median_s"]))
    last = rows[-1]
    bn = "final" if best["step"] is None else f"{best['step']:,}"
    print(f"BEST: {bn}  -> {best['at_buzzer']}/{best['n']} up at the buzzer, "
          f"{best['rose']}/{best['n']} rose, hold {best['hold_median_s']:.2f}s")
    mde = _mde(best["n"])
    gap = (best["at_buzzer"] - last["at_buzzer"]) / max(best["n"], 1)
    if best is not last and gap > mde:
        print(f"  *** THE BEST CHECKPOINT IS NOT THE LAST ONE, by more than "
              f"this eval can resolve. policy.onnx is exported from the final "
              f"model ({last['at_buzzer']}/{last['n']} up), so the artefact on "
              f"disk is NOT the number above. Do not --pick this run without "
              f"saying which checkpoint the claim belongs to.")
    elif best is not last:
        print(f"  the best row is not the last one, but by "
              f"{100 * gap:.0f} points against a resolution of "
              f"{100 * mde:.0f} — that is a tie, not a better checkpoint. "
              f"policy.onnx (the final model) is a fair artefact here.")
    if best["return_median"] <= base["return_median"]:
        print(f"  *** THE BEST POLICY DOES NOT BEAT DOING NOTHING "
              f"({best['return_median']:.1f} vs {base['return_median']:.1f}). "
              f"The reward economy is inverted; the run measured the economy, "
              f"not the learning.")
    print(f"  resolution: {best['n']} episodes sees differences of about "
          f"{100 * mde:.0f} percentage points or more. A smaller gap than that "
          f"is not a result.")
    if best["fell_after_standing"]:
        print(f"  {best['fell_after_standing']}/{best['n']} episodes STOOD and "
              f"then fell. That is a real fall, not a scoring artefact — the "
              f"gap is the hold, not the rise.")
    if best["cut_fired"] > 0.25 * best["n"]:
        print(f"  note: a CURRICULUM cut (track_err / stall / floor) fired "
              f"before standing in {best['cut_fired']} of {best['n']} "
              f"episodes. Scored to the buzzer anyway — those do not exist on "
              f"a robot.")

    # Paired against the donor, on the same episodes.
    rj = json.loads((run / "run.json").read_text())
    donor = rj.get("init_from")
    if donor and Path(donor).is_dir() and (Path(donor) / "model.zip").is_file():
        d = Path(donor)
        print(f"\npaired against its donor {d.name}:")
        d_onnx = _export(d, None, work)
        # Scored under THIS run's flags, not the donor's. A staged chain's
        # stages have different start distributions by design — stage 3 of the
        # Asimov get-up starts on the crouch plateau and stage 4 on a mix — so
        # scoring each under its own env compares two different TASKS and
        # invites exactly the wrong read ("the donor held 2.02 s and this one
        # 0.77 s" was the donor being asked an easier question).
        dr = _score(d_onnx, run, a.episodes, deployment=True)
        print(f"  donor  {dr['at_buzzer']}/{dr['n']} up, hold "
              f"{dr['hold_median_s']:.2f}s, rose {dr['rose']}/{dr['n']}")
        print(f"  this   {best['at_buzzer']}/{best['n']} up, hold "
              f"{best['hold_median_s']:.2f}s, rose {best['rose']}/{best['n']}")
        print(f"  (both on THIS run's env. A gap under {100 * mde:.0f} points "
              "is nothing at this episode count; for a real test run --json on "
              "both and pair the per-episode outcomes.)")

    # THE DELIVERABLE, when the run trained on something easier than it.
    #
    # A staged chain's last stage can start anywhere: the Asimov get-up's
    # final stage starts on a MIX of the floor and two holdable plateaus, and
    # scoring it under its own recorded env reported **27/36 rose, 8/36 up** —
    # numbers carried by the episodes that began already upright on the crouch
    # plateau. From the floor alone the same policy was **17/36 rose, 0/36 up**.
    # Both are true; only the second is the task. A verifier that prints the
    # training env's number as the headline is the same fault it exists to
    # catch, one level up.
    rung = (best["env_kwargs"] or {}).get("start_rung") or ""
    if rung:
        print(f"\nTHE DELIVERABLE — the same checkpoint from the FLOOR only "
              f"(it trained on start_rung={rung!r}, which includes easier "
              f"starts):")
        onnx = _export(run, best["step"], work)
        kw = dict(best["env_kwargs"] or {})
        kw["start_rung"] = ""
        f = _score_with(onnx, run, kw, a.episodes)
        print(f"  {f['rose']}/{f['n']} rose, {f['at_buzzer']}/{f['n']} up at "
              f"the buzzer, hold median {f['hold_median_s']:.2f}s "
              f"(max {f['hold_max_s']:.2f}s), peak {f['peak_median_m']:.3f} m")
        if f["at_buzzer"] * 2 < best["at_buzzer"]:
            print(f"  *** the training env's {best['at_buzzer']}/{best['n']} is "
                  f"NOT the deliverable. Quote {f['at_buzzer']}/{f['n']}.")

    if a.clean:
        print("\nthe same best checkpoint WITHOUT noise or delay:")
        onnx = _export(run, best["step"], work)
        c = _score(onnx, run, a.episodes, deployment=False)
        print(f"  {c['at_buzzer']}/{c['n']} up, hold {c['hold_median_s']:.2f}s, "
              f"peak {c['peak_median_m']:.3f} m")
        if c["at_buzzer"] > best["at_buzzer"]:
            print("  *** it needs the clean conditions to stand. A stand that "
                  "needs zero latency is not a stand that ships.")

    if a.json:
        a.json.write_text(json.dumps(
            {"run": run.name, "baseline": base, "rows": rows}, indent=1,
            default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
