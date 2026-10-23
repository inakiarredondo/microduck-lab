---
name: reward-term
description: Add, weight, or debug a reward term without learning the same four lessons again. Use BEFORE writing a new term, before changing a weight, when a term "does nothing", when a policy games an objective, and when a training curve rises while the deliverable stays at 0. Trigger on: "add a reward term", "the reward isn't working", "it's gaming the reward", "what weight should this be", "why is this term zero", "the policy learned to cheat", "tune the economy".
---

# Adding a reward term

Forty-five runs on the Asimov get-up produced one rule and four ways to break
it. Read the rule, then run the one command.

## The rule: SIGNED and COUPLED

Every term sits in one of four quadrants. **Three of them fail.**

|  | **uncoupled** (one axis) | **coupled** (the product) |
|---|---|---|
| **monotone** (best-so-far) | cannot start; gets banked | cannot start |
| **signed** (per-step rate) | starts, then gets gamed | **this one** |

* **MONOTONE** terms are *identically zero wherever the policy drives their
  quantity backwards* — which is exactly where a reverse curriculum puts it.
  They cannot INITIATE the motion they exist to pay for. Measured:
  `arm_push` **0.00/episode at w=150**; `tuck` moved its own quantity **6 mm
  at w=450**; `rotate` earned **0.5 % of positives at w=200**. Three for
  three. (They have a second failure this repo already records: they pay for
  TOUCHING a value, never for HOLDING it.)
* **UNCOUPLED** terms are dense enough to start, and then get spent on the
  axis they do not price. `tuck_work` (CoM-to-feet MIDPOINT) bought the
  **0.917 m splits** into a state nothing recovers from. `rotate_work`
  (uprightness alone) bought a roll onto the **BACK** — uprightness +0.90 at
  trunk 0.078 — paying **739 of an 873-point episode** for one roll it then
  banked by lying still. `arm_work` (load x dz) bought a **LEAN**: floor peak
  0.449 -> 0.477 while uprightness at that peak fell +0.34 -> +0.23.
* **SIGNED AND COUPLED** is the one that worked. `rise_work` is the per-step
  rate of the env's own `rise = height x max(upright, 0)`: dense, so it can
  start the motion; coupled, so a lean scores nothing (the product falls) and
  a supine roll scores nothing (height ~ 0); farm-resistant by CANCELLATION
  (a closed cycle integrates to zero) rather than by monotonicity. It took
  the floor bands from **0/20 to 16-20/20** ever-standing.

**Price the PAIR, not an axis of it.** A stand needs trunk 0.62 m AND
uprightness +0.85 TOGETHER. If your deliverable is a conjunction, the term
must be a product — otherwise the policy buys the cheap half.

## Before you add anything: is the objective already there?

The env usually computes the right quantity and nobody pays its RATE.
`stand`/`rise` paid the LEVEL of the product (`W_STAND=8`, `W_RISE=2`) and
nothing paid its rate, so there was no dense signal for improving it from
below. **Grep the terms dict first.** Four knobs in this env were already
present and simply dead on the subclass (`span_penalty`, `arm_effort`,
`arm_push`, then `tuck`/`rotate`), and a duplicate key silently ZEROED one
(a dict literal evaluates both values; a monotone getter banks on the first).

## The one command

```bash
uv run python scripts/term_audit.py <run> --rung route:0.00,0.12 --set my_term=50
```

It answers, at the rung you will actually train on:

* **why is it near zero**, in three flavours that used to print one sentence:
  `OFF` (weight is 0.0), `DEAD` (the UNWEIGHTED quantity is ~0 here, so no
  weight helps) and `UNDERWEIGHTED` (live quantity, small weight — with the
  multiplier that reaches 5 % of positives). The audit prints the weight and
  the value with the weight DIVIDED OUT, because it once labelled
  `joint_vel_penalty` *"DEAD here — no weight saves it"* at **-0.22** an
  episode when its weight was the class constant **0.002** and a weight of 7
  priced the same rollouts at **-968**. Five terms moved out of DEAD when the
  column was added, `calm_penalty` among them;
* **is it DOMINANT** (> 50 %) — then it IS the objective, confirm you meant that;
* **is the TASK's own objective drowned** — `stand+rise+progress` were once
  **2.0 %** of the budget while the policy optimised my shaping;
* **does DOING NOTHING pay** — a frozen policy scored **+872.74**/episode, and
  episode length is printed because a rung that ends in **7-9 steps** cannot
  be taught at all.

`scripts/getup_launch.py` now runs this automatically and refuses the launch.

## Price it at the ramp the TRAINER sees

`_penalty_scale()` starts at **0.05** and reaches 1.00 only after 100,000
steps on that env instance, so a freshly built scoring env under-reports
every RAMPED penalty by 20x. `joint_vel_penalty` audited at **-480 an
episode** while the trainer charged **-9,601** — 65 % of the whole positive
budget — and a 20 M-step run was sized on the wrong number. `term_audit`
now sets `_lifetime_steps` to `steps / envs` from the run's own `run.json`
and prints the scale it used. Unramped POSITIVES are unaffected, which is
exactly why the bad table looked plausible.

## If the deliverable is a per-step property of a HELD state, price the plant

Before paying for stillness, smoothness or softness, measure what the
do-nothing policy does to that quantity. Asimov's jiggle: zero action
standing reads **0.000 rad/s, 0 deg/s limb max** — so the plant is already
still and the twitch is the CONTROLLER. Then check whether it is doing work:
jiggle with the CoM outside its support polygon (1.878) vs inside (1.796)
was the same, so it was not balancing either. At that point no weight helps —
`joint_vel=7` moved limb rate 326 -> 298 deg/s while charging 65 % of the
budget. **The answer was a second controller**: handing to the vendor's
walker once up took jiggle **1.609 -> 0.032 rad/s** with 24/24 rises and
24/24 still standing. Same lesson as the HOLD, re-learned for 20 M steps.

## Does the quantity you mean actually get read?

Two checks that each cost a 20 M-step run:

**Grep for the term that integrates YOUR quantity, not the one with your
name.** Asked to stop a robot jiggling, `home` and `calm` are the knobs whose
names fit, and 20 M steps of `home 30, calm 10` moved the jiggle **2.47 ->
2.19** against a 0.021 target. `_home_gain` pays joint POSITION and
`_calm_charge` prices TRUNK speed; limb speed is `joint_vel_penalty` and
nothing else. A name is not evidence about what a term measures.

**Diff the terms dicts down the class hierarchy — then ask which
divergences were DECISIONS.** A charge written inline in a base
`_compute_reward` is absent from every subclass that builds its own dict: two
were, for **209 of 449** get-up runs, after the same bug had been found and
commented FOUR times by reading dicts one at a time. One loop found eleven
divergences. But the first fix hoisted **four** terms and two of them were
deliberate exclusions — `arm_effort_penalty` is kept off the tracking tasks
because `quadruped` and `pike` need the elbows at **58-66 % of rated
torque**, so it charged the only phase that lifts the robot, and
`rise_vel_penalty` is gated on `not up`, which IS the rise. Two exact-set
tests caught it; without them the blanket fix would have re-priced the
lifting phase of every tracking run.

So: **hoist a charge to every task only if it cannot price the task.**
`joint_vel_penalty` is gated on `up` (zero for the whole rise) and
`over_clip_penalty` is exactly zero inside the clip — those are safe.
Anything that can touch the motion stays on the task that measured it, and
the allowlist in `test_every_getup_task_charges_the_ROBOTS_own_costs` carries
the measured reason per name, rather than being inferred from whatever
currently passes.

## Then ship a farm test

Any new term needs a test that an exploit does NOT collect it, A/B'd against
the version you rejected. `test_arm_work_is_signed_and_cannot_be_farmed_by_oscillation`
is the shape: it pins BOTH a large per-step churn AND a small net, because
asserting only the net also passes for a dead term. Measured, a bounce
captures **100 % of its churn** under `max(dz, 0)` and **under 25 %** signed.

## Reading a term's value

Read `env.reward_sums`. A second `_compute_reward()` call reads **zero** for
every monotone term (they mutate their own baseline) and correctly for
per-step ones — which once made a working term look dead. The tell is that
*all* the monotone terms read zero and none of the per-step ones do.

## Related

`AGENTS.md` — "Three for three", "a monotone best-so-far term", "a duplicate
key in a dict literal", "price the do-nothing policy's EPISODE LENGTH".
