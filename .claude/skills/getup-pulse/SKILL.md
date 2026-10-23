---
name: getup-pulse
description: >-
  Check a LONG training chain mid-flight, every few million steps, against the
  previous chain at the same point — and stop it early when it is not
  tracking. Scores the live checkpoint on rises/stays/hold beside the
  zero-action floor, in seconds, without waiting for the run to end. Use
  during any multi-stage Asimov get-up chain, and whenever someone watching
  the lab says "it still doesn't look like it's getting up". Trigger on:
  "is the chain working", "pulse the training", "check it mid-run", "should I
  stop this run", "is it on track".
---

# getup-pulse — do not let a 23 M-step chain run blind

`train-loop` blocks until a run ENDS. `watch-training` renders frames for a
human to read. Neither answers the question that decides whether to keep
spending: **is this chain on track, right now?**

On 2026-10-10 a from-scratch get-up chain ran **12 M steps** while Jonathan
watched it roll around the lab and asked three times why it was not getting
up. The honest answer needed a measurement that did not exist, so the chain
kept running. Twice.

```bash
uv run --with onnxruntime python scripts/getup_pulse.py 127.0.0.1:8788 12
```

Prints the live checkpoint's `rises / stays / best hold / return` beside the
ZERO-ACTION floor for the same env. Takes seconds. Run it **every 2-3 M
steps** of any chain longer than one stage.

## The rule this exists to enforce

**A number is only "on track" against the PREVIOUS chain at the same step
count.** "0/24 rises" is meaningless alone — it was also true of the chain
that later reached 24/24. Score the reference:

```bash
uv run python scripts/getup_acceptance.py <old-chain-stage-N>
```

| what you see | what it means |
|---|---|
| live beats the reference at the same stage | keep going |
| live ties the reference | keep going, pulse again in 2 M |
| live is behind by a stage or more | STOP. The chain is not slow, it is different. |

Measured on 2026-10-10: the TURN chain rose **4/24 by its stage 6**; the
from-scratch WHOLE chain rose **0/24 at stage 5 and 0/12 part-way through
stage 6**. That is a stage behind, and it was the signal to restructure
rather than spend the last 4 M steps.

## Why a chain goes behind — the trap this found

Constraints that apply DURING the rise (`ground`, `limit`) can remove the
exploration route to the skill itself. The hop is ugly, but on this robot it
is how the policy first discovers that leaving the floor is possible; charge
it from step 1 and the rise may never form at all. Constraints that are gated
on already standing (`balance`, `home`, `calm`) cost the rise nothing and are
safe to carry from the beginning.

So the ordering that follows from both failures:

* retrofitting a constraint onto a FINISHED policy fails when the thing it
  forbids has become load-bearing (forcing the arms neutral on that policy:
  13/24 -> **0/24**);
* applying it from step 1 fails when it blocks the route to the skill;
* **apply it while the skill is still plastic** — after the rise exists, and
  while a later term is still reshaping it.

## Stopping a chain

```bash
curl -s -X POST http://127.0.0.1:8788/teach/stop -H 'content-type: application/json' -d '{}'
```

Say the step count and the reference number you stopped against, so the next
person can tell a judged stop from an impatient one.
