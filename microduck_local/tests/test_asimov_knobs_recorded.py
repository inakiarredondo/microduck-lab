"""Every Asimov env knob must be RECORDED onto the run, not just readable.

`AGENTS.md`: *"Add every new knob there, or the run is unreproducible and the
loop above cannot evaluate it correctly."* `asimov_walk.train_env_kwargs` is
that place, and this file fails the build when a knob is missing from it.

It is not a hypothetical. On 2026-10-15 `getup_attitude` was added to
`AsimovGetupEnv`, trained with at 120 for a whole 8.4 M-step chain, and
recorded in `run.json` as **None** — because the knob was added to the env and
not to the recorder. Nothing errors: the env reads its own environment, trains
fine, and leaves no record of what it trained on. The damage is downstream —
`scripts/getup_acceptance.py` and `scripts/score_gap_chain.py` rebuild eval
envs from `replay_kwargs(run)`, so an unrecorded knob means every later
evaluation silently scores the policy WITHOUT the scaffold it trained with, and
`robots/moss.py` already carries the same rule from the same failure.

The mapping from env var to constructor kwarg is DERIVED FROM `__init__`'s
source rather than guessed from the name, because guessing gets it wrong in
both directions: `POSE_ENV` is recorded as `getup_label`, `MOTOR_ENV` as
`actuator_force`, `PUSH_ENV` as `push_bw`, and `CLIP_ENV` is a NAME COLLISION —
`AsimovGetupEnv.CLIP_ENV` is `…_ACTION_CLIP` (recorded `action_clip`) while
`AsimovGetupClipEnv.CLIP_ENV` is `…_CLIPS` (recorded `clip_file`). Keying on
the env-var STRING instead of the attribute name is what makes that resolvable.
"""
from __future__ import annotations

import inspect
import re

import pytest

#: Knobs read straight from the environment rather than through a
#: `kwarg if kwarg is not None else os.environ.get(...)` ctor fallback, so the
#: source regex cannot pair them. Each is named BY HAND in
#: `train_env_kwargs`, and the recorded key is asserted here so a rename
#: cannot quietly orphan one.
READ_DIRECTLY = {
    "MICRODUCK_ASIMOV_GETUP_CLIPS": "clip_file",
    "MICRODUCK_ASIMOV_GETUP_FELL_CUT": "fell_cut",
    "MICRODUCK_ASIMOV_MOTOR": "actuator_force",
    "MICRODUCK_ASIMOV_GETUP_POSE": "getup_label",
    "MICRODUCK_ASIMOV_GETUP_START": "start_phase",
    # The action clip is read by a different ctor idiom, so the source regex
    # does not pair it. It IS recorded, on every branch.
    "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "action_clip",
}

#: Knobs a given task legitimately does not record, with the reason. Kept
#: deliberately SMALL and explicit: an entry here is a claim that the knob
#: cannot affect that task, and the cheap way to be wrong is to add one to
#: silence a failure. Empty until a real case appears.
EXEMPT: dict[str, set[str]] = {}

TASKS = ("getup", "getup_ref", "getup_clip")


def _env_to_kwarg() -> dict[str, str]:
    """Map each `MICRODUCK_*` var to the ctor kwarg it backs, from source."""
    from microduck_local.robots import asimov_env as A

    src = inspect.getsource(A)
    pat = re.compile(r"(\w+)\s+if\s+\1\s+is\s+not\s+None(?:[^;]{0,200}?)"
                     r"os\.environ\.get\(\s*self\.(\w+_ENV)", re.S)
    out = dict(READ_DIRECTLY)
    for kwarg, attr in pat.findall(src):
        # Resolve the ATTRIBUTE to its env-var STRING on whichever class
        # declares it — the attribute name alone collides (see the docstring).
        for cls in (A.AsimovGetupClipEnv, A.AsimovGetupRefEnv,
                    A.AsimovGetupEnv, A.AsimovWalkEnv):
            var = getattr(cls, attr, None)
            if isinstance(var, str) and var.startswith("MICRODUCK_"):
                out.setdefault(var, kwarg)
                break
    return out


def _declared(cls) -> dict[str, str]:
    """`{env var: attribute}` for every knob this env class declares."""
    out = {}
    for attr in dir(cls):
        if not attr.endswith("_ENV"):
            continue
        var = getattr(cls, attr, None)
        if isinstance(var, str) and var.startswith("MICRODUCK_"):
            out[var] = attr
    return out


def _env_classes():
    from microduck_local.robots import asimov_env as A

    return {"getup": A.AsimovGetupEnv, "getup_ref": A.AsimovGetupRefEnv,
            "getup_clip": A.AsimovGetupClipEnv}


def _recorded(task: str) -> set[str]:
    from microduck_local.robots import spec as S

    class Args:
        pass

    a = Args()
    a.task = task
    return set(S.get("asimov1").train_env_kwargs(a).keys())


def test_the_mapping_itself_is_not_empty():
    """A/B GUARD. If the regex stops matching, every assertion below passes
    vacuously — which is the shape of a green test that checks nothing. This
    pins the mapping's size so a refactor of `__init__` fails here loudly
    instead of silently disarming the file."""
    m = _env_to_kwarg()
    assert len(m) >= 30, f"only {len(m)} env->kwarg pairs derived: {sorted(m)}"
    # The three added on 2026-10-15, which are why this file exists.
    for var, kwarg in (
        ("MICRODUCK_ASIMOV_GETUP_ATTITUDE", "getup_attitude"),
        ("MICRODUCK_ASIMOV_GETUP_FOLD_PAY", "fold_pay"),
        ("MICRODUCK_ASIMOV_GETUP_ARM_UNLOAD", "arm_unload"),
    ):
        assert m.get(var) == kwarg, f"{var} should back '{kwarg}', got {m.get(var)}"


@pytest.mark.parametrize("task", TASKS)
def test_every_declared_knob_is_recorded_on_the_run(task):
    """The whole point: a knob the env reads must reach `run.json`."""
    m = _env_to_kwarg()
    cls = _env_classes()[task]
    rec = _recorded(task)
    exempt = EXEMPT.get(task, set())
    missing = []
    for var, attr in sorted(_declared(cls).items()):
        kwarg = m.get(var)
        if kwarg is None:
            missing.append(f"{attr} ({var}): no ctor kwarg found — add it to "
                           f"READ_DIRECTLY with the key it is recorded under")
        elif kwarg not in rec and kwarg not in exempt:
            missing.append(f"{attr} ({var}) -> kwarg '{kwarg}' is NOT in "
                           f"train_env_kwargs('{task}')")
    assert not missing, (
        f"{len(missing)} Asimov knob(s) readable but unrecorded for task "
        f"'{task}' — a run using one cannot be reproduced or re-scored:\n  "
        + "\n  ".join(missing))


@pytest.mark.parametrize("task", TASKS)
def test_a_set_knob_actually_lands_in_the_recorded_kwargs(task, monkeypatch):
    """Recording the KEY is not enough — the VALUE has to travel.

    `train_env_kwargs` reads the environment, so this sets a knob and asserts
    the recorded dict carries it. A key present with a hardcoded default would
    pass the test above and still lose the experiment, which is the
    `misspelled-knob-silent-experiment` failure.
    """
    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_ATTITUDE", "77.0")
    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_FOLD_PAY", "13.0")
    monkeypatch.setenv("MICRODUCK_ASIMOV_GETUP_ARM_UNLOAD", "3.0")
    rec = _recorded(task)
    kw = {}
    from microduck_local.robots import spec as S

    class Args:
        pass

    a = Args()
    a.task = task
    kw = S.get("asimov1").train_env_kwargs(a)
    for key, want in (("getup_attitude", 77.0), ("fold_pay", 13.0),
                      ("arm_unload", 3.0)):
        assert key in rec, f"{key} not recorded for '{task}'"
        assert float(kw[key]) == want, (
            f"{key} recorded as {kw[key]!r}, not the {want} that was set — "
            f"the key is there but the value does not travel")
