"""LAUNCH A GET-UP FINE-TUNE WITH NO INHERITED KNOBS.

    uv run python scripts/getup_launch.py <donor-run> <steps> KNOB=VALUE ...

An `initFrom` job has no stage, so `TrainingJob._job_env` starts from the
behaviour's FINAL stage's knobs and lets the request override. That is
documented and deliberate — and on 2026-10-10 it silently added
`home_pay=30, balance_cost=40, calm_cost=10` to a run meant to isolate
`ground` and `limit`, turning a clean experiment into the everything-at-once
configuration that had already failed. The run recorded it correctly; nobody
read `run.json` until 2 M steps in.

So this sends the COMPLETE knob set every time: every
`MICRODUCK_ASIMOV_GETUP_*` the env knows about, defaulted to its own OFF
value, with only what you name overridden. Omission becomes impossible
rather than merely discouraged.

It prints the full dict before posting, and reads the knobs back off
`run.json` afterwards — the AGENTS.md rule, executed rather than quoted.

AND IT DIFFS AGAINST THE DONOR, because OFF-by-default has a mirror failure
that cost a run the same day: a launch meant to add `handoff_pay` to s11
named four knobs and thereby turned OFF the four that MADE s11
(`ground 6, balance 40, home 30, limit 50`). Five knobs moved in an
experiment written up as one. A knob the donor had ON that this launch would
switch OFF is printed in capitals and REFUSED unless you pass `--allow-drop`.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# The REF env, not the base one: it is what every `getup_ref` fine-tune
# actually constructs, and it declares five knobs of its own (`START`,
# `TRACK_TOL`, `REF_SPEED`, `LAG`, `FREE`). Scanning only the base class
# meant this launcher sent the base knobs completely and let the REF knobs
# -- including `START`, which chooses the rungs and is the single most
# important knob in a curriculum fine-tune -- be inherited from the recipe's
# final stage, silently. That is the exact failure this file exists to stop,
# on the one knob where it matters most. `AsimovGetupRefEnv` subclasses
# `AsimovGetupEnv`, so every base knob is still covered.
from microduck_local.robots.asimov_env import AsimovGetupRefEnv as E


#: EVERY knob this env reads, mapped to the value the env itself would use
#: if nothing were set. Computed, not typed: a hand-written list is a list
#: that goes stale the next time a term is added, and this file exists
#: precisely because an omitted knob is invisible.
def _defaults() -> dict[str, str]:
    import os

    names = [a for a in dir(E) if a.endswith("_ENV")]
    saved = {}
    for a in names:
        k = getattr(E, a)
        saved[k] = os.environ.pop(k, None)
    try:
        env = E(domain_rand=False, obs_noise=False, seed=0)
        attr = {
            "CLIP_ENV": env._act_clip, "ROTATE_ENV": 0.0, "TUCK_ENV": 0.0,
            "SPAN_COST_ENV": 0.0, "SPAN_TARGET_ENV": env._span_target,
            "GROUND_ENV": 0.0, "BALANCE_ENV": 0.0, "HOME_ENV": 0.0,
            "LIMIT_ENV": 0.0, "PUSH_ENV": 0.0, "CALM_ENV": 0.0,
            "SWING_ENV": 0.0, "SPAN_ENV": 0.0, "LAUNCH_SPAN_ENV": 0.0,
            "FALL_DEPTH_ENV": env._fall_depth, "ASSIST_ENV": env._assist,
            "BLEND_ENV": 0.0, "ARM_PUSH_ENV": env._w_arm_push,
            "ARM_EFFORT_ENV": env._w_arm_effort, "ARM_WORK_ENV": env._w_arm_work, "TUCK_WORK_ENV": env._w_tuck_work, "TUCK2_ENV": env._w_tuck2, "ROTATE_WORK_ENV": env._w_rotate_work, "RISE_WORK_ENV": env._w_rise_work, "HEAD_ENV": env._w_head_up,
            "START_RUNG_ENV": "", "PROLOGUE_ENV": "", "POSE_ENV": "supine",
            "HANDOFF_ENV": 0.0,
            "ROLL_ENV": 0.0,
            "FELL_CUT_ENV": 1,
            "PUSH_EVERY_ENV": 0.0,
            "RISE_VEL_ENV": 0.0,
            # WHICH ACTUATOR, and its OFF value is "xml" because that is what
            # every get-up in this repo trained under: position servos at the
            # URDF's continuous effort with NO speed dependence. "dc" is the
            # published speed-torque envelope (robots/asimov_motor). This is
            # the knob that decides whether a run is a claim about Menlo's
            # robot or about a stronger imaginary one, so it goes through the
            # same no-omission machinery as a reward weight.
            "MOTOR_ENV": "xml",
            # Added 2026-10-19. `ATTITUDE`, `FOLD` and `ARM_UNLOAD` are
            # reward/assist terms and OFF is 0.0. The span axis and band
            # take the env's OWN defaults ("both", unbanded) rather than a
            # typed guess. `HISTORY` is the dangerous one: its OFF value is
            # **1 frame**, not 0, so a donor trained at 5 dropping to 1 is
            # not a "drop to OFF" and slid past both diffs below -- while
            # feeding a 78-float observation to a 390-float policy. See the
            # `moved` check.
            "ATTITUDE_ENV": 0.0, "FOLD_ENV": 0.0, "ARM_UNLOAD_ENV": 0.0,
            "SPAN_AXIS_ENV": env._span_axis,
            "SPAN_BAND_ENV": "",
            "HISTORY_ENV": env._history,
            # The REF env's own five. `FREE` is a flag so OFF is 0; the rest
            # take the env's own defaults rather than a typed guess.
            "START_ENV": ",".join(env._start_phases),
            "TOL_ENV": env._track_tol,
            "SPEED_ENV": env._ref_speed,
            "LAG_ENV": int(bool(env._force_lag)),
            "FREE_ENV": 0,
            "ARRIVAL_ENV": 0.0,
            "CUT_BACK_ENV": (env.CUT_BACK_Z if env._cut_back is None
                             else env._cut_back),
            # Added 2026-10-21, and NEITHER OFF value is zero -- the same
            # trap as `HISTORY`. Both knobs override a CLASS CONSTANT
            # (`W_JOINT_VEL` 0.002, `W_OVER_CLIP` 0.2), so "off" means the
            # constant and a launch that sent 0.0 would silently DELETE a
            # charge rather than leave it alone. They are read off the env
            # for the same reason every other entry here is.
            "JOINT_VEL_ENV": env._joint_vel_cost,
            "OVER_CLIP_ENV": env._over_clip_cost,
        }
        # FAIL LOUDLY on a knob this map has not been told about. The map was
        # described as "computed, not typed" and was in fact typed: when
        # `HANDOFF_ENV` was added it was silently dropped from every launch,
        # and the whole point of this file is that an omitted knob is
        # invisible. A missing entry is now a crash, not a default.
        missing = [a for a in names if a not in attr]
        if missing:
            raise SystemExit(
                "getup_launch does not know the OFF value for: "
                + ", ".join(sorted(missing))
                + " — add it to `_defaults`, because a knob this file does "
                  "not send is one the run inherits from the recipe's final "
                  "stage without anybody noticing.")
        return {getattr(E, a): str(attr[a]) for a in names if a in attr}
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


DEFAULTS = _defaults()


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


#: Each `*_ENV` knob and the `env_kwargs` key `Body.train_env_kwargs` records
#: it under. The two names differ (`SWING` -> `swing_pay`, `SPAN` ->
#: `max_span`, `HEAD` -> `head_up`), and the first version of `donor_knobs`
#: guessed with a suffix list that silently missed `head_up` and would have
#: mis-resolved `SPAN`. Guessing is what this file exists to stop, so this is
#: explicit and a missing entry raises.
RECORD_KEY = {
    "ARM_EFFORT_ENV": "arm_effort", "ARM_PUSH_ENV": "arm_push",
    "ASSIST_ENV": "getup_assist", "BALANCE_ENV": "balance_cost",
    "BLEND_ENV": "getup_blend", "CALM_ENV": "calm_cost",
    "CLIP_ENV": "action_clip", "FALL_DEPTH_ENV": "fall_depth",
    "GROUND_ENV": "ground_cost", "HANDOFF_ENV": "handoff_pay",
    "HEAD_ENV": "head_up", "HOME_ENV": "home_pay",
    "JOINT_VEL_ENV": "joint_vel_cost",
    "OVER_CLIP_ENV": "over_clip_cost",
    "LAUNCH_SPAN_ENV": "launch_span", "LIMIT_ENV": "limit_cost",
    "POSE_ENV": "getup_label", "PROLOGUE_ENV": "prologue",
    "PUSH_ENV": "push_bw", "ROTATE_ENV": "rotate_pay",
    "SPAN_COST_ENV": "span_cost", "SPAN_ENV": "max_span",
    "SPAN_TARGET_ENV": "span_target", "START_RUNG_ENV": "start_rung",
    "SWING_ENV": "swing_pay", "TUCK_ENV": "tuck_pay",
    "ROLL_ENV": "roll_pay", "FELL_CUT_ENV": "fell_cut",
    "PUSH_EVERY_ENV": "push_every",
    "RISE_VEL_ENV": "rise_vel_cost",
    "MOTOR_ENV": "actuator_force",
    "ATTITUDE_ENV": "getup_attitude", "FOLD_ENV": "fold_pay",
    "ARM_UNLOAD_ENV": "arm_unload", "SPAN_AXIS_ENV": "span_axis",
    "SPAN_BAND_ENV": "span_band", "HISTORY_ENV": "history",
    "START_ENV": "start_phase", "TOL_ENV": "track_tol",
    "SPEED_ENV": "ref_speed", "LAG_ENV": "force_lag", "FREE_ENV": "free",
    "ARRIVAL_ENV": "arrival", "CUT_BACK_ENV": "cut_back",
    "ARM_WORK_ENV": "arm_work", "TUCK_WORK_ENV": "tuck_work", "TUCK2_ENV": "tuck2", "ROTATE_WORK_ENV": "rotate_work", "RISE_WORK_ENV": "rise_work",
}


#: `RECORD_KEY` keyed by the ENV VAR instead of the attribute name.
RECORD_KEY_BY_VAR = {getattr(E, a): k for a, k in RECORD_KEY.items()
                     if hasattr(E, a)}


def donor_knobs(donor: str) -> dict[str, str]:
    """What the DONOR actually trained with, keyed the same way as DEFAULTS.

    The OFF-by-default rule above fixed silent INHERITANCE and created its
    mirror image: on 2026-10-10 a run meant to add `handoff_pay` to s11 also
    turned OFF the four terms that MADE s11 (`ground 6, balance 40, home 30,
    limit 50`), because they were not named. Five knobs moved in an
    experiment described, in writing, as one. Defaults stay OFF — but a knob
    the donor had ON and this launch is switching OFF is now printed in
    capitals and refused without `--allow-drop`.
    """
    rj = Path(donor) / "run.json"
    if not rj.is_file():
        rj = Path("runs") / Path(donor).name / "run.json"
    if not rj.is_file():
        return {}
    ek = json.loads(rj.read_text()).get("env_kwargs") or {}
    out: dict[str, str] = {}
    for a in (x for x in dir(E) if x.endswith("_ENV")):
        if a not in RECORD_KEY:
            raise SystemExit(
                f"getup_launch does not know which run.json key {a} is "
                "recorded under — add it to RECORD_KEY. A knob missing here "
                "is a knob the donor diff below CANNOT see, which is the "
                "whole point of this function.")
        k = RECORD_KEY[a]
        if k in ek:
            out[getattr(E, a)] = str(ek[k])
    return out


def _numeric(v: str) -> bool:
    """Is this knob's value a NUMBER? A knob whose OFF value is a word is
    categorical, and `_is_off` has no opinion worth having about it."""
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


def _is_off(v: str) -> bool:
    # `run.json` stores booleans as Python bools, so a donor knob that is
    # OFF arrives here as the STRING "False" -- which is non-empty and was
    # therefore read as ON, reporting every boolean-off knob as a drop. A
    # false alarm is not harmless in a tool whose only control is
    # `--allow-drop`: it teaches you to pass the override, and the override
    # hides the real drops too.
    if str(v).strip().lower() in ("false", "none"):
        return True
    if str(v).strip().lower() == "true":
        return False
    try:
        return float(v) == 0.0
    except ValueError:
        return not v


def _accepted_kwargs(cls) -> set[str]:
    """Every keyword `cls.__init__` accepts, walking the MRO.

    A `**kwargs` is NOT a licence to pass anything: it forwards to
    `super().__init__`, and the chain ends at `MicroduckWalkEnv`, which names
    every parameter it takes. So the union of the NAMED parameters down the
    MRO is the real accepted set — which is why
    `AsimovGetupEnv(arrival=0.0)` raises from `MicroduckWalkEnv.__init__`
    rather than being absorbed. An earlier version of this helper bailed out
    to `set()` on the first `**kwargs` it saw and so ruled nothing out, which
    made it inert on the exact class it was written for.

    Returns an empty set only when nothing could be read, so a caller can
    treat "empty" as "do not filter".
    """
    import inspect

    out: set[str] = set()
    for klass in cls.__mro__:
        if klass is object:
            continue
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            sig = inspect.signature(init)
        except (TypeError, ValueError):
            continue
        for name, prm in sig.parameters.items():
            if prm.kind is prm.VAR_KEYWORD or name in ("self", "args"):
                continue
            out.add(name)
    return out


def _preflight_did_not_run(ex: Exception) -> None:
    """Say loudly that the safety check did NOT happen.

    It used to print `preflight skipped: <70 chars>` among forty lines of
    knobs, which reads like a step that passed. On 2026-10-21 that line
    carried `unexpected keyword argument 'arriva` -- the message truncated
    mid-word -- and a launch went out with no economy check at all. A check
    that can silently not-run is worse than no check, because its absence
    looks like its success.
    """
    print("")
    print("  !!!! PREFLIGHT DID NOT RUN -- THIS LAUNCH IS UNCHECKED !!!!")
    print(f"  {type(ex).__name__}: {ex}")
    print("  The economy was NOT scored: nothing here says the floor loses, "
          "that the episodes are long enough to teach, or that the task's "
          "own terms are not drowned. Fix the harness or score it by hand "
          "(scripts/term_audit.py) before trusting the run.")
    print("")


def preflight(donor: str, env: dict[str, str], episodes: int = 6) -> list[str]:
    """Refuse a launch whose ECONOMY is already known to be untrainable.

    Four checks, each of which cost real runs on the Asimov get-up before it
    was done by hand, late:

    * **the floor wins** — zero action out-earns the donor under this economy,
      so the curve measures the economy, not the policy (`ep_rew` once climbed
      216 -> 2942 while the policy learned to lie still);
    * **lying still is profitable** — a frozen policy scored **+872.74** an
      episode, 739 of it one term paying for a single roll it then banked;
    * **the episodes are too short to teach** — every Asimov rung below `fold`
      ends in **7-9 steps**, and seven reward designs were aimed at a reward
      that was never the problem;
    * **the task's own terms are drowned** — with shaping stacked on,
      `stand`+`rise`+`progress` were **2.0 %** of the positive budget.

    Returns the list of reasons to refuse; empty means go. Scored on the
    DONOR, under the economy this launch is about to post, at the band it is
    about to train on — which is the only configuration that answers the
    question.
    """
    import importlib.util as _il

    spec = _il.spec_from_file_location("_ta", Path(__file__).with_name("term_audit.py"))
    TA = _il.module_from_spec(spec)
    sys.modules["_ta"] = TA
    spec.loader.exec_module(TA)

    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs

    kw = dict(replay_kwargs(Path("runs") / Path(donor).name))
    task = kw.pop("task", None)
    for k in ("start_phase", "free", "track_tol", "ref_speed", "force_lag",
              "arrival", "cut_back"):
        kw.pop(k, None)
    # The economy THIS LAUNCH posts, translated from env vars to kwargs.
    for var, val in env.items():
        key = RECORD_KEY_BY_VAR.get(var)
        if key is None:
            continue
        try:
            kw[key] = float(val)
        except (TypeError, ValueError):
            kw[key] = val
    kw.update(fell_cut=False, max_episode_s=8.0)
    task = task or ("getup" if kw.get("start_rung") else "getup_ref")
    # DROP WHAT THIS TASK'S ENV CANNOT ACCEPT, AND SAY SO. Because this file
    # sends the COMPLETE knob set, it also sends the five that exist only on
    # `AsimovGetupRefEnv` — and on 2026-10-21 translating them back into
    # kwargs handed `arrival=0.0` to a base-task env, which raised
    # `unexpected keyword argument 'arrival'`. The `except` below then
    # swallowed it as "a harness problem" and the launch proceeded with NO
    # preflight at all, which is the one outcome this function exists to
    # prevent: a safety check that silently does nothing is worse than none,
    # because the printed line reads like a pass.
    accepts = _accepted_kwargs(S.get("asimov1").env_class(task))
    if accepts:
        dropped = sorted(k for k in kw if k not in accepts)
        for k in dropped:
            kw.pop(k)
        if dropped:
            print(f"   preflight: {task} does not take {dropped} — "
                  "not scored (they reach the trainer, which uses the same "
                  "env, so they are no-ops there too)")
    try:
        e1 = S.get("asimov1").env_class(task)(
            domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
        e2 = S.get("asimov1").env_class(task)(
            domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
        pol, _ = TA._load(Path(donor).name)
    except Exception as ex:                      # never block a launch on a
        _preflight_did_not_run(ex)                      # harness problem
        return []

    try:
        terms, ep_len = TA._run_eps(e1, pol, episodes)
        zero, zero_len = TA._run_eps(e2, None, episodes)
    except Exception as ex:
        # Scoring can fail for reasons that are nothing to do with the
        # economy -- a donor whose observation width does not match the env
        # this launch builds, most often. Still never fatal, and still never
        # quiet: see `_preflight_did_not_run`.
        _preflight_did_not_run(ex)
        return []
    pos = sum(v for v in terms.values() if v > 0) or 1.0
    obj = sum(terms.get(k, 0.0) for k in TA.OBJECTIVE if terms.get(k, 0.0) > 0)
    pnet, znet = sum(terms.values()), sum(zero.values())
    where = (kw.get("start_rung") or kw.get("start_phase")
             or kw.get("getup_label") or "(recorded)")

    print("")
    print(f"PREFLIGHT on {Path(donor).name} at {where}, {episodes} eps:")
    print(f"   donor {pnet:9.2f}/ep over {ep_len:.0f} steps | "
          f"zero action {znet:9.2f} over {zero_len:.0f} steps | "
          f"task's own terms {100 * obj / pos:.1f}% of positives")
    bad = []
    if znet >= pnet:
        bad.append(f"THE FLOOR WINS: zero action {znet:.1f} >= donor {pnet:.1f} "
                   "— the curve would measure the economy")
    if znet > 0:
        bad.append(f"LYING STILL PAYS: zero action earns {znet:+.1f}/episode "
                   "— something must charge per step")
    if zero_len < 20:
        bad.append(f"{zero_len:.0f}-STEP EPISODES under zero action — nothing "
                   "can be taught; fix the termination or the start")
    if obj < 0.10 * pos:
        bad.append(f"OBJECTIVE DROWNED: stand+rise+progress are "
                   f"{100 * obj / pos:.1f}% of positives — the policy would "
                   "optimise the shaping, not the task")
    hot = [(k, 100 * v / pos) for k, v in terms.items() if v > 0.50 * pos]
    for k, share in hot:
        bad.append(f"{k} is {share:.0f}% of positives — it IS the objective; "
                   "confirm that is what you meant")
    return bad


def band_top_is_solved(donor: str, rung: str) -> str | None:
    """Refuse a `route:LO,HI` band whose TOP sits below the solved frontier.

    Measured twice, the second time after the first had been written down: a
    band topping out below the frontier collapses EVERY band to 0/20 while
    peak heights stay high — it keeps rising and stops holding, because the
    finish is never practised. `0.24,0.32` and `0.20,0.34` were fine;
    `0.16,0.32` and `0.12,0.28` each cost a full run. Width is irrelevant.
    """
    if not str(rung or "").startswith("route:") or "," not in str(rung):
        return None
    try:
        hi = float(str(rung).split(":", 1)[1].split(",")[1])
    except (IndexError, ValueError):
        return None
    # The frontier is wherever the donor still stands. Cheap proxy: the band
    # the donor was last trained on, which `run.json` records.
    rj = Path("runs") / Path(donor).name / "run.json"
    if not rj.is_file():
        return None
    prev = str((json.loads(rj.read_text()).get("env_kwargs") or {})
               .get("start_rung") or "")
    if not prev.startswith("route:") or "," not in prev:
        return None
    try:
        prev_hi = float(prev.split(":", 1)[1].split(",")[1])
    except (IndexError, ValueError):
        return None
    if hi < prev_hi - 1e-9:
        return (f"BAND TOP DROPS: {rung} tops out at {hi:.2f} where the donor "
                f"trained to {prev_hi:.2f}. A band whose top is below the "
                "solved frontier collapses every band to 0/20 — the finish "
                "stops being practised. Measured twice.")
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    # `--allow-drop` must NAME the knobs it drops. A blanket override is how
    # `home_pay` went 30 -> 5 -> 0 -> 10 across three launches without anyone
    # deciding to: the donor diff reported it every time and a bare
    # `--allow-drop` suppressed the whole report. The result was a get-up that
    # jiggles at 2.56 rad/s while standing and sits 0.183 rad off the idle
    # pose, against 0.021 and 0.036 for the policy that holds still — a
    # regression nobody chose, caught by a human watching the video.
    allow_names: set[str] = set()
    allow_all = False
    for a_ in list(argv):
        if a_ == "--allow-drop":
            allow_all = True
        elif a_.startswith("--allow-drop="):
            allow_names |= {x.strip().upper() for x in
                            a_.split("=", 1)[1].split(",") if x.strip()}
    argv = [a for a in argv
            if a != "--allow-drop" and not a.startswith("--allow-drop=")]
    allow_drop = bool(allow_names) or allow_all
    _argv_had_nopre = "--no-preflight" in argv
    argv = [a for a in argv if a != "--no-preflight"]
    # The RECIPE is chosen by the request text, and this was hardcoded to
    # "turn not sprawl" -- so fine-tuning a `getup_ref` donor filed the run
    # under a different behaviour, with that recipe's final stage as the
    # inheritance base. Name the recipe explicitly.
    text = "get up by reference"
    argv2 = []
    for a in argv:
        if a.startswith("--text="):
            text = a.split("=", 1)[1]
        else:
            argv2.append(a)
    argv = argv2
    donor, steps = argv[0], int(argv[1])
    env = dict(DEFAULTS)
    for kv in argv[2:]:
        k, _, v = kv.partition("=")
        k = k if k.startswith("MICRODUCK_") else f"MICRODUCK_ASIMOV_GETUP_{k.upper()}"
        if k not in env:
            print(f"unknown knob {k!r}; known: {sorted(env)}")
            return 2
        env[k] = v
    print("POSTING the COMPLETE knob set (nothing inherited):")
    for k, v in sorted(env.items()):
        print(f"   {k:42} {v!r}")

    # DIFF AGAINST THE DONOR. See `donor_knobs`.
    donor_env = donor_knobs(donor)
    dropped = {k: v for k, v in donor_env.items()
               if not _is_off(v) and _is_off(env.get(k, "")) }
    # ...AND THE CATEGORICAL KNOBS, which `_is_off` cannot speak for. It
    # answers "is this zero", so for a knob whose OFF value is a WORD
    # (`MOTOR` = "xml", `POSE` = "supine", `START_RUNG`, `PROLOGUE`) a launch
    # that changes the donor's value away from it is not "turning it off" and
    # slipped through the check entirely. That is the same silent-drop bug
    # from the third side: a donor trained under the dc speed-torque envelope
    # would be reverted to the flat clamp by any later launch that simply did
    # not mention it, and the only symptom would be a get-up that got easier.
    # A differing value is reported whichever direction it moves, because for
    # a categorical knob there is no "less".
    changed = {k: (v, env.get(k, "")) for k, v in donor_env.items()
               if k not in dropped and not _numeric(DEFAULTS.get(k, "0"))
               and str(v) != str(env.get(k, ""))}
    # ...AND THE NUMERIC KNOBS THAT MERELY MOVED. `dropped` only sees a fall
    # to zero, so a knob whose OFF value is NOT zero could be changed
    # silently. `HISTORY` makes that concrete: OFF is 1 frame, so a donor at
    # 5 arrives here as "1", which is neither off nor categorical, and the
    # warm start would feed a 78-float observation to a 390-float policy with
    # no warning at all. Reported in both directions, because for a knob the
    # donor's behaviour depends on there is no safe direction.
    # `not _is_off(v)` matters: without it this fired on every knob the
    # launch TURNS ON from the donor's 0.0, which is how an experiment adds a
    # term — so adding `span_cost=4` was refused as if it endangered the
    # donor. `dropped` already covers ON -> OFF, so this clause only needs to
    # speak for a knob the donor actually HAD on and this launch moves.
    moved = {k: (v, env.get(k, "")) for k, v in donor_env.items()
             if k not in dropped and k not in changed
             and _numeric(DEFAULTS.get(k, "0")) and _numeric(v)
             and not _is_off(v)
             and float(v) != float(env.get(k, "0") or 0)}
    if dropped or changed or moved:
        print("\n*** THIS LAUNCH MOVES KNOBS THE DONOR TRAINED WITH ***")
        for k, (dv, ev) in sorted(moved.items()):
            print(f"   {k:42} donor {dv!r} -> {ev!r}")
        for k, v in sorted(dropped.items()):
            print(f"   {k:42} donor {v!r} -> OFF")
        for k, (dv, ev) in sorted(changed.items()):
            print(f"   {k:42} donor {dv!r} -> {ev!r}")
        print("   The donor's behaviour may DEPEND on these. Name them to keep")
        print("   them, or NAME them in --allow-drop if moving them IS the")
        print("   experiment.")
        moved_names = {k.rsplit("_", 0)[0].replace(
            "MICRODUCK_ASIMOV_GETUP_", "").replace("MICRODUCK_ASIMOV_", "")
            for k in list(dropped) + list(changed) + list(moved)}
        unnamed = moved_names - allow_names
        if allow_all and unnamed:
            print("\n   *** A BARE --allow-drop SUPPRESSES ALL OF THESE AT ONCE ***")
            print("   That is how `home_pay` went 30 -> 5 -> 0 across three")
            print("   launches without anyone deciding to, and the get-up ended")
            print("   up jiggling at 2.56 rad/s. Name them instead:")
            print(f"       --allow-drop={','.join(sorted(unnamed))}")
            return 3
        if unnamed:
            print(f"\n   NOT NAMED in --allow-drop: {','.join(sorted(unnamed))}")
            return 3
    skip_pre = _argv_had_nopre
    reasons = []
    bt = band_top_is_solved(donor, env.get(getattr(E, "START_RUNG_ENV", ""), ""))
    if bt:
        reasons.append(bt)
    if not skip_pre:
        reasons += preflight(donor, env)
    if reasons:
        print("\n*** PREFLIGHT REFUSES THIS LAUNCH ***")
        for r in reasons:
            print(f"   - {r}")
        print("   Fix the economy, or pass --no-preflight if you mean it.")
        return 4

    got = post("http://127.0.0.1:8788/teach",
               {"text": text, "robot": "asimov1",
                "initFrom": donor, "steps": steps, "env": env})
    job = got.get("job") or {}
    run = job.get("runName")
    print(f"\nrun {run} | {job.get('status')}")
    if not run:
        print(got.get("message", "")[:200])
        return 1
    # READ IT BACK. The rule exists because a run that records something else
    # is a different experiment under the right name.
    rj = Path("runs") / run / "run.json"
    for _ in range(30):
        if rj.is_file():
            break
        time.sleep(2)
    if rj.is_file():
        ek = json.loads(rj.read_text()).get("env_kwargs", {})
        # Derived, not typed: the FIRST version of this list omitted
        # `handoff_pay`, so the knob the run existed to test was the one knob
        # the readback could not show. That is the bug this file is about.
        print("\nread back off run.json:")
        for k in sorted(ek):
            if k.endswith(("_pay", "_cost", "_bw", "_target")):
                print(f"   {k:14} {ek.get(k)}")
        # AND REFUSE if a knob we SENT did not land. Printing was not enough:
        # on 2026-10-12 `ROLL=100` was sent, silently dropped by a third
        # kwargs branch nobody had edited, and the run trained without the
        # term it existed to test. A readback that only prints relies on
        # somebody reading it.
        # A knob THIS TASK'S env does not declare is a legitimate no-op, not
        # a lost experiment. Because this file sends the complete set, a base
        # `getup` run always carries the five that live only on
        # `AsimovGetupRefEnv`, and on 2026-10-21 all four of those with a
        # record key were reported as "the run is training a DIFFERENT
        # experiment under this name. Stop it." on a run that was correct in
        # every knob. An alarm that cries wolf on every launch is one people
        # learn to scroll past, which is how the REAL instance of this
        # (`ROLL=100`, silently dropped) would get missed next time.
        from microduck_local.robots import spec as _S
        task_of_run = (ek.get("task")
                       or ("getup" if ek.get("start_rung") else "getup_ref"))
        try:
            accepts = _accepted_kwargs(_S.get("asimov1").env_class(task_of_run))
        except Exception:
            accepts = set()
        missing, na = [], []
        for k, v in env.items():
            if _is_off(v) or k not in RECORD_KEY_BY_VAR:
                continue
            key = RECORD_KEY_BY_VAR[k]
            if key in ek:
                continue
            (na if (accepts and key not in accepts) else missing).append(k)
        if na:
            print(f"\n   not applicable to task {task_of_run!r} (no-ops, "
                  f"correctly absent): {sorted(na)}")
        if missing:
            print("\n*** THESE KNOBS WERE SENT AND NOT RECORDED ***")
            for k in sorted(missing):
                print(f"   {k}  ->  expected run.json key "
                      f"{RECORD_KEY_BY_VAR[k]!r}, absent")
            print("   The env never received them: the run is training a "
                  "DIFFERENT experiment under this name. Stop it.")
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
