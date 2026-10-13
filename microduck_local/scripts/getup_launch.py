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

from microduck_local.robots.asimov_env import AsimovGetupEnv as E


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
            "ARM_EFFORT_ENV": env._w_arm_effort, "HEAD_ENV": env._w_head_up,
            "START_RUNG_ENV": "", "PROLOGUE_ENV": "", "POSE_ENV": "supine",
            "HANDOFF_ENV": 0.0, "SWING_ENV": 0.0, "ROLL_ENV": 0.0,
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
    "LAUNCH_SPAN_ENV": "launch_span", "LIMIT_ENV": "limit_cost",
    "POSE_ENV": "getup_label", "PROLOGUE_ENV": "prologue",
    "PUSH_ENV": "push_bw", "ROTATE_ENV": "rotate_pay",
    "SPAN_COST_ENV": "span_cost", "SPAN_ENV": "max_span",
    "SPAN_TARGET_ENV": "span_target", "START_RUNG_ENV": "start_rung",
    "SWING_ENV": "swing_pay", "TUCK_ENV": "tuck_pay",
    "ROLL_ENV": "roll_pay",
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


def _is_off(v: str) -> bool:
    try:
        return float(v) == 0.0
    except ValueError:
        return not v


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    allow_drop = "--allow-drop" in argv
    argv = [a for a in argv if a != "--allow-drop"]
    donor, steps = argv[0], int(argv[1])
    env = dict(DEFAULTS)
    for kv in argv[2:]:
        k, _, v = kv.partition("=")
        k = k if k.startswith("MICRODUCK_") else f"MICRODUCK_ASIMOV_GETUP_{k.upper()}"
        if k not in env:
            print(f"unknown knob {k!r}; known: {sorted(env)}")
            return 2
        env[k] = v
    named = {kv.partition("=")[0] for kv in argv[2:]}
    print("POSTING the COMPLETE knob set (nothing inherited):")
    for k, v in sorted(env.items()):
        print(f"   {k:42} {v!r}")

    # DIFF AGAINST THE DONOR. See `donor_knobs`.
    dropped = {k: v for k, v in donor_knobs(donor).items()
               if not _is_off(v) and _is_off(env.get(k, "")) }
    if dropped:
        print("\n*** THIS LAUNCH TURNS OFF KNOBS THE DONOR TRAINED WITH ***")
        for k, v in sorted(dropped.items()):
            print(f"   {k:42} donor {v!r} -> OFF")
        print("   The donor's behaviour may DEPEND on these. Name them to keep")
        print("   them, or pass --allow-drop if removing them IS the experiment.")
        if not allow_drop:
            return 3
    got = post("http://127.0.0.1:8788/teach",
               {"text": "turn not sprawl", "robot": "asimov1",
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
        missing = [k for k, v in env.items()
                   if not _is_off(v) and k in RECORD_KEY_BY_VAR
                   and RECORD_KEY_BY_VAR[k] not in ek]
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
