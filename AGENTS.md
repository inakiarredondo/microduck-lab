# Microduck local-training workspace

This repo is a community harness for training and watching RL policies for the
[Microduck](https://pollen-robotics.com/microduck) — Pollen Robotics' ~25 cm,
~800 g bipedal robot (14 Dynamixel XL330 servos, IMU, 50 Hz control) — **on an
ordinary Mac, no CUDA GPU required**. It is not affiliated with Pollen Robotics.

The workspace is four side-by-side checkouts. This repo tracks two of them; the
two upstream Pollen repos are cloned next to them (they are in `.gitignore`):

| Repo | What it is | Stack |
|---|---|---|
| `microduck_local/` | **This repo.** Local CPU-MuJoCo + Stable Baselines 3 PPO prototyping harness — same 61-obs contract and MJCF as microduck_rl. Includes `duck-lab`, the streaming backend for duck-viewer, and the 🎓 teach-a-trick training loop. | Python 3.12 / uv |
| `duck-viewer/` | **This repo.** Next.js + react-three-fiber browser viewer — many policies/checkpoints walking side by side, live over WebSocket from `duck-lab`. | Next.js / TS |
| `microduck/` | Upstream: the robot's onboard software, shipped ONNX policies, docs. Clone from `pollen-robotics/microduck`. | Rust workspace |
| `microduck_rl/` | Upstream: the official GPU training stack (MuJoCo Warp + mjlab + PPO), BAM actuator sim2real recipe, ONNX export. Clone from `pollen-robotics/microduck_rl`. **This is the sim2real recipe; this repo is the prototyping loop.** | Python 3.12 / uv |

## Setup

One command does all of the below (upstream clones at the pinned shas, the
shipped policies from the Hub, `uv sync`, `npm install`, a smoke test) on a
Mac or Linux:

```bash
git clone <this repo> microduck-workspace && cd microduck-workspace && ./scripts/setup.sh
```

By hand:

```bash
git clone <this repo> microduck-workspace && cd microduck-workspace
git clone https://github.com/pollen-robotics/microduck
git clone https://github.com/pollen-robotics/microduck_rl
# The contract, golden-bit and symmetry tests are measured against specific
# upstream models and policies — the same shas CI pins (.github/workflows/tests.yml).
# (The golden bits are per platform, tests/goldens/; a platform without a
# recording skips them — record yours with MICRODUCK_RECORD_GOLDENS=1.)
git -C microduck_rl checkout badc4e7ffe5507fd7acb1a21487bd2925c1afe5a
git -C microduck checkout 2c61dcc1f03440541cdc0729f7a375b2a9ea3005
cd microduck_local && uv sync            # needs https://docs.astral.sh/uv/
# The shipped policies left the microduck repo for the Hub on 2026-09-03
# (ef4becf); the pinned sha above still vendors them, any later one does not.
# This Hub revision is byte-identical to the vendored set — see setup.sh.
uv run hf download pollen-robotics/microduck-policies \
  alpha_walking.onnx alpha_stand.onnx alpha_sitstand.onnx alpha_ground_pick.onnx \
  ball_kick_left.onnx ball_kick_right.onnx roller.onnx roller_crouch.onnx roulade.onnx \
  --revision 088524a64e2557dc453256b6071dbb9d23888802 --local-dir ../microduck/policies --quiet
cd ../duck-viewer && npm install
```

`microduck_local` finds the MJCF models in `../microduck_rl` (override with
`MICRODUCK_RL_DIR`) and the shipped reference policies in
`../microduck/policies/` (downloaded from
[pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)
by `setup.sh`; upstream no longer vendors them).

## Read the repo-local docs first — they are authoritative

- `microduck_local/README.md` — what the harness is/is not for, every command,
  the measured performance story.
- `microduck_local/AGENTS.md` — **the training playbook for agents**:
  invariants, reward-design rules, verification discipline. Read it before
  touching rewards, observations, or training code.
- `duck-viewer/README.md` — viewer architecture and the GPU pitfalls already hit.
- `microduck_rl/AGENTS.md` (upstream) — the full sim2real playbook the local
  harness mirrors.
- `.claude/skills/render-rollout/SKILL.md` — how to *look* at what a policy
  actually does (works as plain documentation for any agent, not just Claude).
- `.claude/skills/add-a-robot/SKILL.md` — how to bring SOMEBODY ELSE'S robot
  in: the licence check, the pinned fetch, the ladder from "on the stage" to
  "it trains", the number that settles each step, and the eight traps hit
  doing it for real. Read it before adding a body or touching `robots/`.
- `.claude/skills/record-world/SKILL.md` — how to *record* a world scenario
  (living room, playroom, soccer pitch) to video + a contact sheet + an events
  log, headless under a seed, and read what the ducks did. Debug the `/sim`
  page with this, not by describing what a browser tab looked like.
- `docs/mars-roadmap.md` — the plan for a THIRD body, Innate's MARS (a
  wheeled base with a 6-DoF arm): what is measured, the `Body`/`RobotSpec`
  split that makes the next robot a registry entry, and the phases with
  the number that settles each. Read it before adding any robot.
- `docs/roadmap.md` — the working list of experiments: what to run next, the
  command for each, and the number that would settle it. Read it before
  starting anything open-ended, and **write the answer back into the item**
  when you finish one — a negative result is worth as much as a positive one,
  and this is where the next person finds out it was already tried.

## What runs where

- **A second body:** the Unitree G1 (29 joints, 99-obs) trains, exports,
  renders and appears in the lab beside the ducks — `robots/spec.py` is the
  seam, `uv run fetch-g1` the setup. A lab contract, not a sim2real one.
- **Somebody else's robot:** Menlo Research's open-source
  [Asimov 1](https://github.com/menloresearch/asimov-1) (1.2 m, 32 kg, 23
  joints, 78-obs) is a level-1 body here — `uv run fetch-robot asimov1`. Its
  model is GPL-2.0 / CERN-OHL-S-2.0 and its policy BSD-3-Clause, so BOTH are
  fetched and neither is vendored. The proof the contract is theirs and not
  invented is that their own shipped policy walks in this repo's MuJoCo
  (0.44 m/s at a 0.5 command, no falls). Nothing here has run on an Asimov.
  **It also GETS UP** — `asimov-getup-h5-dag`, 40/48 rises and 28/48 still
  standing at the buzzer, inside Menlo's published speed-torque envelope
  (`robots/asimov_motor`, `actuator_force="dc"`). Two caveats that belong in
  the same breath: it reads **390 floats** (5 stacked frames of their 78), so
  it will not hot-swap or load in their viewer and needs an on-device ring
  buffer — the 5-frame history is LOAD-BEARING, not a convenience, and one
  frame is 0/48 — and it **sprawls 0.944 m** on the way up, inherited from the
  openhorizon-labs demonstration it was cloned from (which sprawls 0.955 m).
  A narrow get-up needs a narrow demonstration, not a better clone.
- **Any Mac (tuned on Apple Silicon), CPU-only:** everything in this repo —
  training (`train-walk`, `train-behavior`), eval, ONNX export, rendering,
  the lab + viewer. Linux works too (set `MUJOCO_GL=egl` for offscreen
  rendering); the MPS update path is Mac-only and auto-disables elsewhere.
- **Linux / cloud CPU boxes:** the same commands, under a different torch
  thread policy that `machine.py` detects (cores read from CPU affinity and
  the cgroup quota, not `os.cpu_count()`, so a container gets its real
  budget). Mac behavior is unchanged by it — see "Cloud and Linux
  training" in `microduck_local/README.md` for the measurements and the
  `--compare-profiles` A/B.
- **Needs a CUDA GPU:** the upstream `microduck_rl` MuJoCo Warp training —
  the final sim2real step once a behavior prototyped here is worth it.
- **Runs on the robot:** ONNX exported by `export-walk` is drop-in compatible
  with the deployment contract, but ship policies retrained on the official
  stack — see "sim2real honesty" in `microduck_local/AGENTS.md`.

## Command crib

```bash
# --- microduck_local (run from microduck_local/) ---
uv run --with pytest pytest tests/            # contract tests — run before training
uv run train-walk --envs 32 --steps 3_000_000 --run-name my-run
uv run export-walk runs/my-run && uv run eval-walk runs/my-run/policy.onnx
uv run train-behavior one_leg                 # teachable tricks (behaviors/)
uv run train-brain --run-name follow-v6 --steps 2_000_000 --variety \
    --title "Follower v6" --description "what it tests, and later what it found" --group shipped-followers
uv run describe-brain p-n256-s31 --title ... --description ... --group capacity   # name a BRAIN after the fact
uv run describe-run <run> --title "Front kick (G1)" --note "8/8 hold 20 s, apex 0.62 m" --pick  # name a WALK/TASK run
uv run describe-run <run> --backfill          # derive its title/description from run.json
uv run render-rollout --policy runs/my-run/policy.onnx --behavior stand --out /tmp/rr
uv run record-world pitch-2v2 --seconds 30 --out /tmp/rw   # world video + sheet + events.txt
uv run record-world playroom --brain d0=tidy --camera follow:d0 --seconds 60 --out /tmp/rw-tidy
uv run machine-facts                          # cores + thread profile for THIS machine
uv run bench-envs                             # the right --envs for THIS machine
uv run bench-envs --compare-profiles          # mac profile vs linux/cloud profile, interleaved

# SOMEBODY ELSE'S robot (Menlo's Asimov 1) — see .claude/skills/add-a-robot/
uv run fetch-robot                            # every body this install knows
uv run fetch-robot asimov1                    # MJCF + meshes + both licences
uv run python -m microduck_local.robots.asimov_walk --fetch   # their policy
uv run python scripts/probe_mjcf_body.py asimov1 --out /tmp/p # LOOK at a body
uv run python scripts/probe_asimov_walk.py    # does THEIR policy walk HERE?
uv run python scripts/probe_asimov_getup_ref.py --verify   # the authored get-up:
                                              # every waypoint's 6 s hold, then the
                                              # servos ramped through it with NO policy
uv run record-world scenarios/asimov-walkabout.json --camera follow:p0 \
    --seconds 45 --out /tmp/rw-asimov        # the level: it walks, ducks follow

# THE REAL ACTUATOR ENVELOPE (robots/asimov_motor) and the get-up that fits it
uv run python -m microduck_local.robots.asimov_motor          # the published table
uv run python -m microduck_local.robots.asimov_motor --fetch  # openhorizon's release
uv run python scripts/probe_actuator_ab.py <run> --episodes 48    # xml vs dc, paired
uv run python scripts/score_getup_dc.py <run> [<run> ...]         # the DEPLOYMENT env
uv run python scripts/probe_history_ablation.py                   # is history load-bearing?
uv run python scripts/term_audit.py <run> --rung route:0.00,0.12   # AUDIT THE ECONOMY
                                              # before training: dead terms, dominant
                                              # terms, is the TASK's own objective
                                              # drowned, and does doing nothing pay
uv run python scripts/probe_openhorizon_lab.py --episodes 24      # their policy, our env
uv run python scripts/harvest_getup_clips.py --our-env --episodes 40 \
    --out .cache/asimov_1/getup_clips_dc.npz    # demos OUR physics produced
uv run python scripts/distill_getup.py --run-name <name> \
    --history 5 --actuator dc --episodes-per-iter 250   # the clone that works
uv run python scripts/distill_getup.py --run-name <name> \
    --history 5 --actuator dc --student-from <clone>    # DAgger: it buys the HOLD

# a SECOND robot (the Unitree G1) — see microduck_local/README.md
uv run fetch-g1                               # MJCF + meshes + walker.onnx into .cache/
uv run bench-walk --robot g1                  # it costs ~2x the duck per step
uv run distill --robot g1 --teacher .cache/unitree_g1/walker.onnx --run-name g1-clone
uv run train-walk --robot g1 --envs 32 --init-from runs/g1-clone --run-name g1-walk
uv run export-walk runs/g1-walk               # obs[1,99] -> actions[1,29]
uv run duck-lab --checkpoints runs/my-run    # + viewer below → watch it in the browser
uv run duck-lab --world playroom              # world mode: rooms, sensors, brains (the /sim page)
uv run eval-tidy --seeds 3 --seconds 300      # Track 12 benchmark; trace-tidy / walker-facts to debug it
uv run eval-tidy --robot mars --seeds 3 --seconds 300   # the same room, tidied by MARS's ARM (brain tidy_arm)
uv run eval-pitch --seeds 4 --seconds 300     # soccer benchmark: two chase brains, shipped kicks, goals
uv run eval-pitch --seeds 12 --per-side 3 --out runs/poacher.jsonl --tag poacher   # resumable: re-run the SAME
                                              # command after any interruption and it continues (--seed0 extends)

# --- duck-viewer (run from duck-viewer/) ---
npm run dev                                   # then open the printed localhost URL
npm test                                      # vitest: the canvas arithmetic (lib/*.test.ts)

# --- upstream microduck_rl (run from microduck_rl/, GPU) ---
uv run train <TASK> --env.scene.num-envs 64 --agent.max_iterations 5   # smoke test
uv run scripts/infer_policy.py --walking ../microduck_local/runs/my-run/policy.onnx
```

## Conventions that matter

- **A new robot is fetched, never vendored, at a pinned sha — and its licence
  is checked before anything else.** A revision is a different robot (Asimov's
  pinned one is the commit that aligned its joint ranges to the hardware), the
  bytes are verified against the listing's git blob sha1, and the LICENCE file
  travels into the cache and gets printed, because a licence that stayed on
  GitHub is a licence nobody read. The full route is the `add-a-robot` skill.
- **An actuator model is a claim about somebody's hardware, and a torque limit
  with no SPEED term is the strongest claim in the file.** Asimov's envs clamped
  the position servos at Menlo's URDF `<limit effort=>` — correct, per-group,
  read from the vendor — and applied it flat, with no speed dependence at all.
  That is a robot which is weak standing still and unlimited once moving, so
  the cheapest route through any reward is to fling everything: our get-up
  exceeded Menlo's published ratings on **82 % of rise frames, max 5.29x**, and
  under the real envelope went **48/48 -> 0/48** (paired, same seeds, p =
  0.0000). The real model is a speed-torque curve — `peak_torque` near stall
  falling to zero at the rated speed — and it is PUBLISHED
  (`robots/asimov_motor`, mjlab's `dc_motor_clip` via openhorizon-labs), so it
  is copied rather than invented: `kp`/`kd`/`armature`/`friction` agree with
  Menlo's `env.yaml` on all 23 joints, `vlim` is the URDF's, and
  `peak_torque` is theirs and recorded as theirs. Two corollaries worth more
  than the fix. **A crude limit is worse than none**: the first version bled
  torque off above the rating without ever granting the 3x peak below it, so it
  over-penalised in one direction and under-modelled in the other, and the
  "0/24" it produced had to be retracted even though the re-measurement agreed.
  And **the envelope is rarely the obstacle** — openhorizon's policy rises
  24/24 inside the same limits, in our MuJoCo, so "this body cannot" was again
  a statement about our route. Before training a motion against a torque limit,
  ask what the limit does at SPEED, and price the do-nothing policy inside it:
  under `dc` our donor returned **-671** against a zero-action floor of
  **+255**, so a warm-start run there would have learned to lie still.
- **Never invent a number somebody published, and never trust a number nobody
  measured.** If a robot ships a policy, its observation layout is published —
  mirror it, cross-check two independent publications of it, and then prove it
  by running THEIR policy in OUR env. A wrong permutation still produces
  actions and still moves the robot; it just falls over, and every measurement
  after it describes a different robot.
- Policies are hot-swapped on the robot behind a **shared 61-dim obs /
  14-action ONNX contract** — never change the obs layout per-task; zero-pad
  unused command slots.
- Always deploy ONNX from `export-walk` / upstream `scripts/export.py` — they
  bake the obs normalizer in. Never hand someone a raw checkpoint.
- **A robot's mesh dump carries the model's OWN normals**, and its vertex
  lattice is read off the model's tessellation, not inherited
  (`robots/g1.extract_visual_scene`, `robots/mjcf_body.visual_vert_scale`).
  Inventing normals in the browser is what made every Menagerie body, and
  MOSS, arrive melted — 14-78 % of face corners more than 15° off what MuJoCo
  draws. The G1 and MARS still take the invented ones because their material
  tables were tuned against them; everything else ships the real ones. Traps 9
  and 10 of `.claude/skills/add-a-robot/SKILL.md` have the measurements.
- **To get two colours on one link, cut the SOURCE STL — never
  `model.mesh_vert`.** `mesh_vert` is MuJoCo's RECENTRED copy of the mesh, so
  writing those vertices into a new file subtracts the recentre a second time
  and the halves land where the original never was. Done that way on Asimov
  (2026-10-04) it sank his torso 18 cm, ate the neck column and left the head
  floating; four "corrections" were then measured against the wrong premise
  and all four failed. Split the triangles of the file on disk, as
  `mars.split_base_mesh` has always done for MARS's tyres, and both halves
  land within **4e-6 mm** of the uncut mesh with no position compensation at
  all. Pick the cut KIND from the part: a connected
  shell when it is its own piece (a visor, a tyre), a face-normal test when
  it is a face of a curved surface (a sole), a plane only when neither fits
  — **a plane across a curve serrates**, which is what put magenta teeth on
  Asimov's soles. A raised boss on a curved shell (his knee disc) is none of
  the three and defeated six attempts.
- **A cut's thresholds are in the SOURCE file's frame, which is not the
  compiled one.** Asimov's foot spans z −34..+23 mm on disk; a threshold read
  off `mesh_vert` was −74 mm, selected zero triangles and made `walk_spec()`
  raise — mid-session, with a trainer running. `cut_part` guards against an
  empty or total selection for exactly this reason, and caches on the source
  mesh's mtime AND this module's, so editing a threshold actually re-cuts.
- **Compare geometry at `mj_resetData` (qpos0), never at "keyframe 0".**
  `walk_spec` and friends add their own STAND keyframe, so resetting both
  models to keyframe 0 compares two different POSES and reports movement in
  links nobody touched. That confounded the first diagnosis of the bug above
  by 56 mm. `tests/test_asimov_look.test_the_look_moves_no_geometry_at_all`
  is the shape to copy.
- **A vendor's link name is not the part — an MJCF link is named for the joint
  ABOVE it.** `knee_link` is the segment BELOW the knee: the SHIN. On
  2026-10-09 a grep of body names for "shin"/"calf" returned nothing on Asimov
  and was reported as "this robot has no shin collision geom", then used to
  argue a half-kneel was impossible. The shin is there — a 90 x 40 mm capsule
  on `knee_link` — and so are the thighs (`hip_yaw_link`). Ask
  `asimov_getup_ref.part_of` / `floor_parts`, which derive the part from the
  kinematic chain (the joint a body hangs from, to its own child joint) and
  are pinned by `test_the_vendor_link_names_mean_what_the_CHAIN_says`; every
  render and report prints PARTS, never link names. The real reason that robot
  cannot kneel is the KNEE, not the ankle — **corrected 2026-10-16**. Folding
  BOTH knees to the limit and settling with MuJoCo's own contacts, the FOOT
  takes the floor at every ankle angle from -20 to +90 deg: 4.5x the shipped
  range changes nothing. Sweeping knee flexion instead, the SHIN takes the
  floor from **120 deg** and the robot has **85.9** — it is **34 deg short**,
  and the ankle is irrelevant (20 deg and 45 deg give identical contacts at
  every knee angle). The earlier ankle attribution came from a `geom_rbound`
  ranking whose error (up to 63 mm) was an order of magnitude larger than the
  1-11 mm margins it reported; the same probe, rebuilt on contacts, reverses
  it. `scripts/probe_kneel_ankle.py`.
- **Train from the TASK's own states, not from whatever cache already plugs
  into the curriculum machinery.** On 2026-10-08 the Asimov get-up — prone,
  push-up, knees, pike, stand, every pose of it already settled, verified and
  0.215 m narrow — was trained instead from a cache of FALLING states, for one
  reason: `start_rung="falling"` was already wired up and the authored
  waypoints needed a different task. Four stages and 7.5 M steps later the
  policy was indistinguishable from zero action at every depth, because 21 of
  its 24 start states are the robot already horizontal with downward momentum.
  **Reachable is not recoverable:** "a state the robot demonstrably passes
  through" justifies that it can be REACHED, never that anything can recover
  from it, and a start nothing recovers from produces no successes and
  therefore no gradient. Before choosing a start distribution, score the
  do-nothing policy ON IT; if nothing succeeds from those states, the
  curriculum has no anchor and no reward will give it one.
- **Price the do-nothing policy's EPISODE LENGTH, not just its score — a rung
  that ends in eight steps cannot be taught anything.**
  `AsimovGetupRefEnv` cuts on `trunk_z < _anchor_z - CUT_BACK_Z` with
  `CUT_BACK_Z = 0.15` measured from the height the episode STARTED at. Its
  rungs below `fold` are not equilibria, so they sink and hit that cut in
  **7-9 steps (0.14-0.18 s)** every episode, forever — and zero action's
  `peak z` EQUALS its `start z` at prone, plant_forearms, draw_knees_in,
  quadruped, pike and rock_back, so the do-nothing policy never lifts the
  robot a millimetre from any of them. Two chains spent 2.37M and 2.96M steps
  scoring **0/24** on those rungs while `fold` held 24/24. **The sprawl was
  manufactured by the cut**: under zero action the span at `rock_back` is a
  narrow 0.214 m, while the trained policy reaches 0.530 there and 0.731 at
  pike — because flinging the legs is the only thing that can move the reward
  before the episode ends. Seven reward designs were aimed at a reward that
  was never the problem. One `print` of the zero-action episode length would
  have said so; under ~20 steps, stop and fix the termination or the start.
- **A monotone best-so-far term also never pays for APPROACHING its target —
  so it cannot START the motion it exists to pay for.** This file already
  records that such a term "pays for TOUCHING a value, never for HOLDING it";
  the third face of the same coin is that it is identically **zero on the
  whole region below the episode's start**, which is precisely where a
  reverse curriculum puts the policy. `arm_push`
  (`load x max(_trunk_up_speed(), 0)`, gated on new-best trunk height) earned
  **0.00 per episode at w=150** from `pike`, because the robot only sinks.
  The whole budget there was `track` 85.3 + `contact` 73.0 over 109 steps —
  and `contact` is collected by LYING THERE, so the do-nothing policy took
  ~100 % of the reward while the term that pays for getting up read zero.
  The fix is a different farm defence, not a weaker one: `arm_work` pays
  SIGNED work (`load x dz`), dense from step one and zero round any closed
  cycle. A/B'd against the rejected `max(dz, 0)` version, a bounce captures
  **100 % of its churn** with the clamp and under 25 % signed. Read a new
  term's value off `env.reward_sums` AT THE RUNG before trusting it.
- **Three for three: every monotone best-so-far term in the Asimov get-up is
  zero exactly where it is needed, and the signed version is the fix.**
  `arm_push` (new-best height) earned **0.00/episode at w=150** from `pike`
  because the robot sinks; `tuck` (best CoM-to-feet midpoint) moved its own
  quantity **6 mm at w=450**; `rotate` (best uprightness) earned **19.2 of
  3733 positives, 0.5 %, at w=200** while the policy was climbing to 0.61 m
  FACE DOWN (uprightness -0.33 to -0.46 at peak height) — a monotone term
  cannot see a quantity being driven backwards. The signed analogue of the
  same quantity reads **-862.7 at w=30** on those rollouts: 45x the
  magnitude, correctly negative. Signed work is farm-resistant by
  CANCELLATION (a closed cycle integrates to zero) rather than by
  monotonicity, so it keeps the density without reopening the oscillation
  exploit. `arm_work`, `tuck2`, `rotate_work`. Before weighting a monotone
  term, read it off `env.reward_sums` AT THE RUNG; if it is ~0 there, no
  weight will save it.
- **A duplicate key in a dict literal silently ZEROES a monotone term.**
  Python evaluates BOTH values for a duplicate key and keeps the last, so
  adding a second `"arm_push"` to a terms dict that already had one made the
  surviving entry read 0.0 — `_trunk_up_speed()` banks the new-best height on
  the first call and returns 0.0 on the second. Nothing errors and the knob
  still records its weight. The same trap as "a stateful reward can only be
  read where it is SUMMED", from the authoring side.
  `test_the_arm_reward_cannot_be_farmed_by_flailing` caught it; grep the dict
  for the key before adding it.
- **A term reads zero for THREE different reasons, and only one of them is
  unfixable — an instrument that prints one sentence for all three sent a
  whole session down the wrong road.** `scripts/term_audit.py` classified a
  term by its WEIGHTED per-episode value and labelled anything under 1 % of
  the positives `DEAD here — no weight saves it`. That sentence is a claim
  about every possible weight, made from a measurement at one weight. On
  2026-10-21 it printed exactly that for `joint_vel_penalty` at **-0.22 an
  episode** against 14,690 of positives; the term's weight is the class
  constant `W_JOINT_VEL` = **0.002**, and at `joint_vel_cost=7` the SAME
  rollouts price it at **-968**, so a weight saved it completely. Meanwhile
  the robot stood there twitching at a median **326 deg/s** of limb rate —
  against the shipped walker's **3 deg/s** in the same posture — and the one
  instrument aimed at that charge reported it beyond rescue. The fix is a
  column, not a threshold: divide the weight out and print the UNWEIGHTED
  quantity, then say `OFF` (weight is 0.0), `DEAD` (the quantity itself is
  ~0, no weight helps) or `UNDERWEIGHTED` (live quantity, and here is the
  multiplier that reaches 5 % of positives). Re-run on the same rollouts it
  had already scored, the audit moved **five** terms out of DEAD, including
  `calm_penalty`, whose "no weight saves it" I had quoted earlier the same
  day as grounds for abandoning calm as a lever. **Before concluding a term
  cannot matter, read its value with the weight divided out.**
- **Nothing in `home` or `calm` can see a limb jiggle, because neither
  measures limb speed.** Asked to make the Asimov settle after its get-up,
  the obvious knobs were the two with the right names, and `home 30, calm 10`
  moved the jiggle **2.47 -> 2.19** (target 0.021) over 20 M steps.
  `_home_gain` pays joint POSITION against the default pose and
  `_calm_charge` prices TRUNK speed, so a limb twitching at punch speed under
  a perfectly still torso is invisible to both — and `home` was already the
  second-largest term in the economy at **4737 an episode, 32 % of
  positives**, so there was no headroom there anyway. Limb speed has exactly
  one charge in this file, `joint_vel_penalty`, gated on `up` so it cannot
  eat the rise. **Name the quantity you want changed, then grep for the term
  that integrates THAT quantity** — a knob whose name matches your intent is
  not evidence it reads what you mean.
- **Two charges that belong to the ROBOT were summed on one task out of
  three, for 209 of 449 get-up runs — and the two that LOOK like their
  siblings must stay out.** `joint_vel_penalty` and `over_clip_penalty` were
  written inline in `AsimovGetupEnv._compute_reward` and nowhere else, so
  `AsimovGetupRefEnv` and `AsimovGetupClipEnv` — every `getup_ref` and
  `getup_clip` run — summed neither. Both subclass dicts already carried a
  comment about this happening a FOURTH time (`span_penalty`, `arm_effort`,
  `arm_push`, `tuck`); these were the fifth, and a per-dict comment is
  evidently not a fix. `_regulariser_terms(up)` returns them from one place,
  and `test_every_getup_task_charges_the_ROBOTS_own_costs` diffs each
  subclass's summed keys against the parent's. **The first version of that
  helper returned FOUR terms and was wrong**: `arm_effort_penalty` is
  excluded from the tracking tasks by measurement — `quadruped` and `pike`
  are unreachable on this body without the elbows at **58-66 % of rated
  torque**, so it charged the robot for the only phase that lifts it, and the
  torque envelope is a phase-4 measurement rather than a reward term — and
  `rise_vel_penalty` is the same objection by construction, being gated on
  `not up`, which IS the rise. The ref env's two exact-set tests caught both;
  without them a blanket "add the missing regularisers" would have re-priced
  the lifting phase of every tracking run. So the rule is narrower than it
  first looked: **a charge may be hoisted to every task only if it cannot
  price the task** — `joint_vel_penalty` is gated on `up` and so is zero for
  the whole rise, `over_clip_penalty` is exactly zero inside the clip. Hoist
  those; leave anything that can touch the motion on the task that measured
  it, and put the measured reason in the allowlist rather than inferring the
  allowlist from whatever currently passes. **Diff the terms dicts across the
  class hierarchy rather than reading them one at a time**: the sweep cost
  one loop and found eleven divergences where reading had found one each of
  the previous four times — then the tests told me which of the eleven were
  decisions. And it is worth saying which way this one cut: the jiggle
  Jonathan reported was NOT caused by it, since that run trained on the base
  task where both were present. The omission was real, serious and
  unrelated; being right about a bug is not the same as having found the
  cause.
- **A safety check that cannot run must not print like one that passed.**
  `scripts/getup_launch.py`'s preflight scores the economy and refuses four
  known-untrainable shapes. Because the launcher sends the COMPLETE knob set,
  it handed `arrival=0.0` to a base-task env that does not declare it, and
  the `except` printed `preflight skipped: ...unexpected keyword argument
  'arriva` — truncated mid-word, one line among forty of knobs — then
  launched. The run was in fact fine, which is worse: the habit survives
  intact until the launch where it is not. Three repairs, all of them
  general: ask the env class what kwargs it ACCEPTS (`_accepted_kwargs`
  unions the named parameters down the MRO — a `**kwargs` is not a licence,
  it forwards to `MicroduckWalkEnv`, which names everything) and print what
  was dropped; make a failure LOUD (`!!!! PREFLIGHT DID NOT RUN — THIS
  LAUNCH IS UNCHECKED !!!!`) while still never blocking a launch; and stop
  the readback alarm crying wolf, because the same complete knob set made it
  report four correctly-absent ref-only knobs as *"the run is training a
  DIFFERENT experiment under this name. Stop it."* on a run whose every knob
  was right. **An alarm that fires on every launch is one people learn to
  scroll past**, which is how the real instance of it — `ROLL=100`, silently
  dropped — gets missed next time.
- **A knob the OFF-by-default launcher cannot see is a knob the recipe still
  supplies.** `scripts/getup_launch.py` scanned only `AsimovGetupEnv`, so the
  five knobs declared on `AsimovGetupRefEnv` — `START`, `TRACK_TOL`,
  `REF_SPEED`, `LAG`, `FREE` — were inherited from the recipe's FINAL stage
  silently, `START` included, which chooses the rungs and is the most
  important knob in a curriculum fine-tune. And its two diffs could not see a
  numeric knob whose OFF value is not zero: `HISTORY` is **1**, so a donor at
  5 dropping to 1 was invisible while feeding a 78-float observation to a
  390-float policy. `run.json` also stores booleans as `False`, which
  `_is_off` read as ON. All fixed; on the next real launch the tool caught
  **eleven** moving knobs, including that `HISTORY` drop and `MOTOR dc ->
  xml`. When a safety tool reports nothing, check it can see the field.
- **Grep before you BUILD, not only before you give up.** The rule below was
  written for "cannot be done" and the expensive half is the opposite: two new
  curricula were designed, tested and trained while `asimov_getup_ref` — eight
  stages over the authored waypoints, each adding one earlier keyframe and
  REHEARSING the later ones, at the clip where `fold` is actually stable —
  already sat in `behaviors/asimov_tasks.py` implementing exactly the
  requested design. "Does a recipe for this already exist?" costs one grep and
  was skipped for ~20 M training steps.
- **One definition of success, in the library, called by every scorer.** Ten
  get-up scripts each had their own copy of "did it stand" and they did not
  agree: some asked whether the PEAK trunk height over the episode cleared the
  bar. At `fold` the episode STARTS at 0.523 m and the bar is 0.523 m, so
  every episode passed before doing anything — a whole curriculum was designed
  on a "free rise" of 23/24 that was the starting pose being counted — and a
  trajectory search maximising the same quantity returned a 0.615 m "stand"
  that was the robot face down on its elbows, uprightness -0.61. Height alone
  is never it: `asimov_getup_ref.standing_report` requires the trunk up, the
  robot upright AND nothing but feet on the floor, evaluated on the pose that
  SURVIVES rather than on a maximum, and
  `test_every_get_up_scorer_asks_the_SAME_question` fails the build on a
  private copy.
- **When a call site is wrong, measure the FUNCTION's blast radius before
  patching the caller — and a kwarg another body does not speak is
  DISCARDED, not refused.** The lab's `env_kwargs_for_behavior` returned the
  duck's shape — `behavior_id` + `standing_spawns` — for every behavior, and
  `Duck._make_env` pops both before it reaches a non-duck body. What was left
  was `{}`, which `slot_env` honours by falling back to that body's
  `default_task`: a finished Asimov get-up was re-seated in the WALK env,
  standing, with nothing to get up from, and the only symptom was a robot
  doing nothing. The first fix branched at the one call site that had been
  seen to fail. Sweeping the registry then showed the function returned NO
  TASK for **28 behaviors across all four non-duck bodies** — so the branch
  belongs in the function, where the next caller and the next body get it
  for free. Two cheap habits prevent the whole class: **sweep the registry,
  not the symptom** (one loop over `BEHAVIORS` priced this in seconds and
  turned "a lab bug" into "every non-duck task"), and **make a silent
  discard loud** — `Duck._make_env` now records into `MISROUTED_PREVIEWS`
  and prints, rather than raising, because a raise inside the 50 Hz loop
  costs the whole roster. Held by
  `test_a_preview_env_NAMES_THE_TASK_for_every_body`, which asserts it swept
  at least 20 behaviors so a green result cannot mean "swept nothing".
  **It happened THREE times in one day, so treat it as a class and grep for
  the shape, not the instance.** The shape is *a duck-shaped question asked
  of a non-duck body*, and the giveaway is reading a per-task field off
  `env.behavior` — which only the duck's env has. The other two:
  `handoff_fn` read that way fired on **0 of 1750** lab frames (silent), and
  `handoff_recenter` read the same way ran the duck's **61-obs** walker on
  the 78-obs Asimov and **deleted the body from the stage** (`Got: 78
  Expected: 61`). Both now live on the `Duck` (`Duck.handoff_fn`,
  `Duck.handoff_recenter`), set from the roster entry that knows the robot,
  and the recentre is hard-refused when `self.robot != "microduck"`. When you
  fix one of these, `grep -n 'env\.behavior\|\.behavior\.' ` the module
  before you call it done.
- **A launcher that sends EVERY knob turns omission into an explicit zero —
  so diff it against the parent, not against the env.**
  `scripts/getup_launch.py` was built because an `initFrom` job silently
  INHERITS the recipe's final stage's knobs (`_job_env`), which had added
  `home 30, balance 40, calm 10` to a run meant to isolate two terms. The
  cure sends all 23 knobs defaulted to their OFF value, so omission is
  impossible — and then the handoff retrain, launched with four knobs named,
  silently turned OFF the four terms that MADE its parent (`ground 6,
  balance 40, home 30, limit 50`). **Five knobs moved in an experiment
  described, in writing, as one.** Both failure modes are the same bug from
  opposite ends: the launch's baseline must be the DONOR's recorded
  `env_kwargs`, and any knob this launch turns off that the donor had on is
  worth printing in capital letters before posting.
- **A monotone best-so-far term pays for TOUCHING a value, never for
  HOLDING it — if the deliverable is a held state, something must pay per
  step.** `progress`, `rotate` and `tuck` are all best-so-far: the robot
  banks the gain the first instant it reaches a height, an uprightness or a
  feet-under-mass gap, and from then on may drift for free. That is why the
  Asimov get-up learned to arrive and let go. Measured on
  `teach-asimov_getup_turn-380ec2`: while STANDING by `standing_report` —
  trunk up, upright, nothing but feet down — the centre of mass was OUTSIDE
  its own support polygon **74 % of the time** (median -22 mm, p10 -131 mm),
  and every lost stand ran the same way: uprightness steady at **+0.98**
  while the CoM-to-feet gap grew 200 -> 320 mm. Not a topple — a translation
  off the feet while vertical, which an instantaneous pose test cannot see
  and a best-so-far term has no opinion about.
  `asimov_getup_ref.support_margin` is the missing clause (signed distance
  to the support polygon, built from MuJoCo's CONTACT POINTS, because a pad
  hovering 2 mm clear is not support); `standing_report` now carries
  `com_margin` and `balanced` BESIDE `standing` rather than inside it,
  because folding it into `standing` would silently rewrite every number
  this repo has published. A per-step charge for the failure, gated on
  already being up so it cannot eat the rise, is `GROUND_ENV`'s sibling
  `BALANCE_ENV`.
- **When the deliverable is "and then it keeps going", the answer may be
  ANOTHER CONTROLLER, not another reward term.** Nine reward terms and
  eleven runs went into making the Asimov HOLD the stand it had learned to
  reach, and every one was a null or a regression — because the rise arrives
  at the top with **0.89 m/s** of trunk speed and the stance that was being
  trained absorbs **0.22 m/s**. No per-step charge removes arriving momentum;
  the momentum IS the rise. What worked, same afternoon, was handing the
  robot to the vendor's own shipped walk policy the moment it is up
  (`Behavior.handoff_fn` / `handoff_policy`): over 200 paired episodes,
  **200/200 rises, 180 still up at the buzzer, and 180 of the 181 shoves
  that land survived**. Paying the get-up to arrive in a catchable state
  (`HANDOFF_ENV`) on top of that economy is a measured NULL — the handoff
  itself is the whole win. And the tuning went the
  opposite way to intuition — **earlier is better** (hand off at 0.60 m/s,
  not once calm) — because a walker is an ACTIVE controller and that 0.22
  ceiling was measured on a PASSIVE one. Before pricing a hold, ask what
  already controls this robot in that state and what it would take to reach
  its basin: 128 handoffs says trunk speed is the only strong separator
  (1.19 sd) and that a WIDER stance is caught more often (0.422 vs 0.345 m),
  which also means a gate written from intuition can be the thing in the way.
- **A SWITCH between policies must read what the ROBOT can sense, and its
  thresholds are chosen on a COST, not on agreement.** The Asimov's
  two-policy architecture hangs on one predicate, and that predicate asked
  for world-frame trunk height, MuJoCo's contact list and world-frame linear
  velocity. The robot has none of them — Menlo's 78-obs layout is gyro,
  projected gravity, joint positions, joint velocities and the last action —
  so the selector that made the whole thing work could not have run on
  hardware, which defeats the point of staying inside their contract. Build
  the switch from the OBSERVATION VECTOR and score it end to end: IMU
  uprightness alone reproduced the privileged test on **99.0 %** of 15,458
  frames and still cost **11 of 48 rises**, because a latching one-way
  switch cannot undo a false fire and that 1 % sits exactly on the decision
  boundary. Adding two encoder channels (elbow and ankle, the states that
  fool the IMU — a robot propped on its forearms reads upright) cut
  false-positives **1.04 % -> 0.07 %** and matched the privileged selector
  on every clause, while making overall agreement slightly WORSE. **Frame
  agreement is the wrong objective whenever the two error directions cost
  different amounts**, and for a latching switch they differ by the length
  of an episode.
- **A stateful reward can only be read where it is SUMMED.** `rotate`
  measured as paying 0.000 per episode, which matched a known historical bug
  in that exact term and was therefore believable. It was the instrument:
  calling `env._compute_reward()` a second time per step reads zero for
  every MONOTONE best-so-far term (`rotate`, `tuck`, `progress`) because
  they mutate their own baseline — while per-step terms (`stand`, `rise`)
  read correctly, which is what made a working term look dead. The tell is
  that *all* the monotone terms read zero and none of the per-step ones do.
  Read `env.reward_sums`, never a second call.
- **"Running" is not "progressing", and a test that seeds the value it
  checks proves nothing.** On 2026-10-10 a trainer sat at **0 of 3,000,000
  steps for 16 minutes** while `GET /teach/status` happily reported
  `running: True` — the env's constructor had succeeded and `step` raised on
  the first call, every episode, forever. The bug:
  `getattr(self, "_prev_swing_z", 0.0)` returns **`None`** when the
  attribute EXISTS and is `None`, so a per-episode baseline initialised to
  `None` made `float - None` the first arithmetic of every rollout. Its unit
  test passed because the test SET `_prev_swing_z` to a float before calling
  the method — it was checking the one state the bug cannot occur in. Two
  habits: poll `steps` and require it to be INCREASING, never merely
  `running`; and when a test sets up the attribute under test, delete that
  line and watch it fail (`AGENTS.md`, "A/B new tests against planted
  regressions") — a reset-path bug is invisible to any test that performs
  the reset itself.
- **A stand is not a get-up: score the ROUTE, not only the arrival.**
  `asimov_getup_ref.standing_report` asks the right question about a POSE —
  trunk up, upright, nothing but feet down — and on 2026-10-09 a policy
  passed it 23 times out of 24 by **hopping**. Counted frame by frame, 1.04 s
  of its 3.4 s rise had ZERO floor contacts, in 8 separate flights, the
  lowest geom of a 32 kg robot reaching 105 mm of clearance; 0.00 s airborne
  once it was up. It was reported as a staged get-up for a day. Every term in
  that economy paid for what a bounce PRODUCES — height (`stand`, `rise`,
  `progress`), uprightness (`rotate`), feet under the mass (`tuck`) — and
  none priced the impulse, so the cheapest route through the reward was
  ballistic and the policy took it. **A per-step property of the trajectory
  is not implied by any property of its endpoint**: if the deliverable is a
  SAFE motion, something has to charge per step for the unsafe one, and the
  check has to ask MuJoCo's contact list (`AsimovGetupEnv._on_floor_contacts`
  / `GROUND_ENV`), never a clearance threshold — a 1.2 m robot resting on one
  heel has every other geom well clear of the floor. Jonathan found it from
  the player, by saying he could not see it get up; the frames said
  `on floor: -` and I had not read them.
- **Before deciding a thing cannot be done, grep for it.** This repo had
  been cutting meshes since MARS (`mars.split_base_mesh`), and a day went
  into re-deriving — then wrongly forbidding — something already working ten
  lines away. "Do we already do this?" is the cheapest question here and the
  one most often skipped.
- **A rule in this file is executed by other agents, so a wrong one does
  damage a wrong line of code cannot.** A "never cut a visual mesh" bullet
  written from one failed afternoon was read by a parallel session, which
  dutifully emptied a working table and documented why. Write the MECHANISM
  and the number that settles it, never a prohibition inferred from a single
  failure — and when a rule here turns out to be wrong, fix it before fixing
  the code it misdirected.
- **An instrument agrees with the half of the world it can see — so price the
  do-nothing policy before you believe any curve.** On 2026-10-05/06 the three
  things blocking the Asimov get-up were all instrument faults, not the robot,
  and each metric was internally consistent while blind exactly where the
  failure lived. A distillation loss computed on data whose 23 `last_action`
  floats were constant reported the best clone fidelity this robot had had
  (**0.000228 rad²**) for a policy that stood **0 of 24**. A `track` reward
  priced on joints alone gave a robot lying on the floor with straight legs a
  perfect **10.0 of 10.0**, and because the fall cut arms on `_stood`, not
  standing was never terminated — so doing nothing earned **2775 an episode
  against the riser's 788** and `ep_rew` climbing 216 → 2942 was the policy
  learning to lie still. One zero-action rollout gives the number every training
  curve has to beat; if the floor wins, the curve is measuring the economy.
  **Diff the training data's per-slot variance against the deployed env's**
  before blaming capacity or sample count, and **check what your termination is
  asymmetric about** — a cut armed by success pays failure in episode length.
  All nineteen faults, each with the number that settles it, are in
  `microduck_local/AGENTS.md` under "Nineteen ways a measurement here has lied".
- **A byte-identical A/B means the HOOK is dead, not that the variable does not
  matter — and a knob nothing reads trains a different experiment under the
  right name.** A per-env action clip reached the reset pose but neither `step`
  nor the gym action space (which SB3 clips to FIRST), so widening it changed
  nothing and was written up as "the clip is not the cause"; it had never been
  applied. Three runs were then launched with a `track_tol` knob missing a word
  from its name and trained at the recipe's default while being described, in
  writing, as something else. Grep every site a variable is supposed to reach
  before concluding anything about the variable, and read the knob back off
  `run.json` before describing a run. `/teach` now refuses an unknown
  `MICRODUCK_*` key and names the nearest real one.
- **Compare paired when both arms see the same episodes.** A continuation moved
  the median unbroken stand 5.15 → 6.67 s and was a NULL: 9-vs-5 discordant
  pairs, p = 0.424, **+0.22 ± 0.66 s**. A bimodal outcome's marginal median will
  disagree with the paired test, and the paired test is the one that counts.
  And score at the horizon the deliverable is defined at, with only the
  deployment's own cuts — honouring the curriculum's terminations turned the
  same rollouts from **23/24** into **12/24**.
- **A contact sheet cannot show stillness — it samples at ~1.4 Hz and jiggle
  is a 50 Hz phenomenon.** A finished Asimov get-up was reported as a steady
  hold off captions reading `trunk_z=0.63, tilt=1-2 deg, feet L=1 R=1,
  floor:none` frame after frame; the robot was vibrating at **2.56 rad/s**
  and sitting **0.183 rad** off the idle pose, against **0.021** and
  **0.036** for the policy that holds still. Jonathan saw it in the mp4. The
  cause was a weight nobody decided to change: `home_pay` went **30 -> 5 ->
  0 -> 10** across three launches as collateral while proxies were being
  stripped, and a bare `--allow-drop` suppressed the donor diff that named it
  every time. Two fixes, both shipped: `--allow-drop` must NAME the knobs it
  drops (a bare one is refused and prints the exact list), and
  `scripts/score_bands.py` carries `jiggle` and `off-idle` columns so a
  stillness regression appears in every table. For anything about rate or
  smoothness, watch the video or print a number — a still image is the wrong
  instrument.
- **A RAMPED penalty is priced at 5 % on a fresh env, so an audit of it is
  wrong by 20x.** `_penalty_scale()` walks `_PENALTY_STAGES` off
  `_lifetime_steps` — **0.05** at construction, reaching **1.00** only after
  100,000 steps on THAT env instance — and every scoring tool here builds a
  fresh env. On 2026-10-21 `scripts/term_audit.py` read
  `joint_vel_penalty` at **-480 an episode** and the trainer was charging
  **-9,601**, about 65 % of the policy's entire positive budget. The weight
  for a 20 M-step run was chosen on that number. Nothing was broken: the
  term was read correctly, at a scale nobody had asked for, and the figure
  looked plausible because the UNRAMPED positives (`rise_work`, `home`,
  `stand`) are unaffected and dominated the table. `term_audit` now sets
  `_lifetime_steps` to `steps / envs` from the run's own `run.json` and
  PRINTS the resulting scale, for the zero-action floor too — not a
  hardcoded 1.0, because a short run really does train mostly at the gentle
  end and hiding that would be the same fault reversed. The re-priced table
  reads `joint_vel_penalty -8286`, the largest magnitude in the economy,
  which is the kind of thing that changes a decision.
- **Stillness is the HOLD problem again, and the answer is again another
  CONTROLLER.** This file already records that nine reward terms failed to
  make the Asimov hold a stand and the handoff to Menlo's walker fixed it in
  an afternoon. The same is true of the twitch Jonathan saw at the end of the
  render, and it took a 20 M-step run to re-learn: `joint_vel=7` moved limb
  rate only **326 -> 298 deg/s** while charging 65 % of the budget. Three
  measurements say why no weight was ever going to do it. The PLANT does not
  buzz — zero action standing gives jiggle **0.000** and limb max **0 deg/s**
  on both actuator models, so it is entirely policy-induced. It is not
  CORRECTIVE — jiggle with the CoM outside its support polygon (1.878) and
  inside (1.796) is the same, so it is not balance. And it is not CHEAP — the
  policy paid and kept twitching. One network was being asked to rise fast
  and stand still. Handing to the vendor's walker the moment it is up:
  jiggle **1.609 -> 0.032 rad/s**, off-idle **0.089 -> 0.027**, 24/24 rises,
  24/24 still standing, 24/24 shoves survived, mean joint error vs the
  default pose **1.17 deg**. **Before pricing a per-step property of a held
  state, price the do-nothing policy's value of that property** — if the
  plant reads 0.000, the term is fighting the controller, not the physics.
- **The duck-shaped lookup, a FOURTH time — in the module whose own entry
  says to grep for it.** `render_rollout.handoff_due` read
  `env.behavior.handoff_fn`; `behavior` is a field only the DUCK's env has,
  so on an Asimov, G1, MARS or MOSS it returned None and fell through to the
  BACKFLIP rotation gate (`_bf_rot`), which is also None there — so
  `--handoff` could never fire on any non-duck body. It failed SILENTLY and
  convincingly: the render completed, every caption read `drv=<the trick
  policy>`, and the mp4 showed the unhanded behaviour while the flag said
  otherwise. The env had exposed the right seam the whole time —
  `handoff_ready`, the one definition `asimov_tasks._getup_handoff_due`
  delegates to — and the renderer simply never asked it; it does now, before
  the duck field. The same sweep found `caption_fn`, `markers_fn` and
  `report_fn` read the same way in the same file. And one more trap behind
  it: **two policies in a handoff need not read the same vector.** Asimov's
  get-up takes 390 floats (5 stacked frames, newest last) and the walker 78,
  so the moment the predicate was fixed the render raised `Got: 390
  Expected: 78`. An ONNX driver now feeds each model the width it DECLARES,
  taking the newest frame — which is exactly what the on-device ring
  buffer's head holds.
- **A deliverable attached PER-BEHAVIOR is a deliverable that depends on
  which of seventeen near-identical entries a run happened to name.** The
  Asimov's get-up is only finished when it hands to Menlo's walker and stops
  moving — and `handoff_fn` + `handoff_policy` were declared on exactly ONE
  of the seventeen Asimov get-up behaviors (`asimov_getup_turn`). Every
  working get-up in this repo trained under `asimov_getup` or
  `asimov_getup_whole`, which had neither, and `viz_server.handoff_for`
  returns None unless the behavior declares it. So on the lab's stage those
  runs drove the get-up policy forever. **Jonathan watched the finished
  robot SHAKE while this session's acceptance numbers said it was still** —
  both were right about different programs, because the scoring scripts
  wired the handoff by hand and the lab could not. Measured over 24
  episodes from prone, that gap is worst joint **252 -> 2.3 deg/s** and
  **16.3 -> 5.1 deg** off the default pose, with rises and stays unchanged.
  `_attach_getup_handoff()` now gives it to every get-up behavior that has
  not chosen its own (`Behavior` is `@dataclass(frozen=True)`, so it
  REPLACES and re-registers — assignment there is silently a no-op). Two
  habits: **when a result depends on a wrapper your script supplies, check
  the PRODUCT supplies it too**, and when you find one entry carrying a
  field, sweep the other sixteen.
- **Zero action is the stillest controller this robot has and it cannot
  catch the rise.** Handing the get-up to `action = 0` — which commands the
  default stand — gives the quietest pose in the repo (**worst joint 1.0
  deg/s**, against 2.3 for the walker and 252 for the get-up policy), and it
  **falls 17 of 24** because it is PASSIVE and the rise arrives with
  momentum. Walker first, then fade: 50 steps (1 s) of walker then zero is
  **24/24 rises, 22/24 stays at 0.9-1.9 deg/s**, while fading at 25 steps
  drops to 12/24. So the catch needs an ACTIVE controller for about a
  second and nothing after that — the same 0.22 m/s-vs-0.89 m/s arithmetic
  this file already records for the hold, now with the fade point measured.
- **A visible anomaly in your own render is evidence, not noise.** The same
  floating head appeared in two renders I took and I explained it away as
  camera framing both times; the user caught it. If something looks wrong in
  a picture, measure it before narrating it away.
- **A discrepancy you measured and waved off must be waved off against the
  case you are WORKING ON, not the convenient one.** On 2026-10-04 the robot's
  lowest point was computed from `geom_rbound` — a BOUNDING-SPHERE radius. I
  measured the error, saw it was ~2 mm for the standing poses and up to 63 mm
  for the lying ones, wrote "negligible for the standing poses" and moved on.
  The task was a GET-UP: every pose in it is a lying pose. Measured at the
  actual episode resets it was `plant_forearms` **+50 mm floating** and
  `draw_knees_in` **+52 mm**, while `prone` sat **25 mm BURIED** in the floor
  and `quadruped` and `pike` 12-13 mm — so every episode began either with a
  5 cm fall already in progress or with the solver violently ejecting the
  robot, which is where the 100 %-torque reset transients had been coming from.
  Jonathan found it by looking at the `/sim` stage and asking *"why is it
  floating?"*. The number was in my own output the whole time.
  **Before writing "negligible": name the case it is negligible FOR, check that
  is the case in hand, and if it is not, fix it.** Use MuJoCo's own contact
  distances (`asimov_getup_ref.settle_onto_floor`) rather than any bound on the
  geometry — every bound is wrong somewhere, and `rbound` is wrong by the
  length of the longest capsule.
- **A failed poll is not a result, and a stale artefact is not a fresh one.**
  The same day, a progress watcher returned `{}` on a timed-out request, which
  read as "the job finished" — so it exported a policy from a PARTIAL
  checkpoint (98 k of 300 k steps) and reported it as the rung's outcome. And
  because it exported only `if not policy.onnx.is_file()`, that stale file
  would then have been scored as the final one. Distinguish "no answer" from
  "the answer is no", and re-derive an artefact rather than trusting one that
  is already on disk.
- Prototype here → port the env design to an mjlab cfg → retrain on GPU with
  the official stack. The local harness is for minutes-long iteration loops,
  not for the policy you put on hardware.
- Every trained run carries a `title`, `description` and `group` in its
  `brain.json` (`train-brain --title/--description/--group`, or
  `describe-brain` later). The run name is an identifier and never changes;
  the title is what the `/train` page and the `/sim` brain menu show. Write
  the finding into the description when the experiment resolves — see
  "Every run is a record" in `microduck_local/AGENTS.md`.
- A walk or task run (every `train-walk` / G1 run) carries the same thing in
  `runs/<name>/record.json` — written automatically by the trainer, edited
  with `describe-run`, and shown by the lab's policy palette as the chip's
  title and tooltip. **`--pick` names the stage of a curriculum chain worth
  using**: without it the palette's ▶ and ⤓ point at the LAST stage, which in
  the G1 kick chain was the worst one (3/8 seeds against 8/8 a rung earlier).
  Only a measurement may set it.
- **IF IT MOVES THE ROBOT FOR MORE THAN A MINUTE, IT HAPPENS WHERE JONATHAN
  CAN SEE IT.** Not "training" — anything. The old wording said *training*, and
  on 2026-10-04 that let hours of trajectory searches, timing scans and
  parameter sweeps run inside plain background shells: every one of them
  stepping the physics, none of them visible, and the only thing on the stage
  was finished runs lying on the floor. "It isn't training" is not an
  exemption; the question is whether the robot is moving and whether he can
  watch it.

  | what you are doing | how it becomes visible |
  |---|---|
  | training a policy | `POST /teach` on `127.0.0.1:8788`, which streams to the viewer at `localhost:63317` — never a bare `uv run train-*` |
  | a search, sweep or optimisation over motions | run it as a `/teach` job if it can be posed as one; otherwise cap it at a minute, and **render the best candidate** (`render-rollout`) so there is something to look at |
  | a one-off probe that prints numbers | fine unattended — but if the CONCLUSION is about motion, render it before reporting |
  | a battery or a paired A/B | the stated exception, `MICRODUCK_RUNS_DIR` — say so when you start one |

  And **say where it is running when you start it**, every time, with the URL.
  Two chains were launched correctly through the lab and still never seen,
  because each finished inside one turn and the lab reaps the trainee the
  moment a job ends — which is a thing to warn about up front, not to explain
  afterwards. Full rule in `microduck_local/AGENTS.md`.
- **And POSTing is not the same as being watched.** After launching, confirm
  a real frame carries the trainee (`uv run --with websockets python
  scripts/stage_check.py` — there is no GET for the roster), and give
  the person the URL of the lab you actually launched on (a tab pointing at a
  scratch lab shows an empty room while the job runs fine). A restarted lab
  starts EMPTY and reaps the trainee the moment the job ends, so a chain that
  finishes inside one turn was never seen by anyone.
- **The stage must show the physics the trainer is running.** The lab builds
  preview envs in-process; the trainer is a subprocess that imports the env
  fresh. Edit anything under `robots/`, `behaviors/`, `world/` or `brain/` and
  the stage keeps previewing the code the lab booted with — so restart the
  backend BEFORE you POST, while it is still idle. Both rules, with the
  measurements behind them, in `microduck_local/AGENTS.md`.
- Before claiming anything about a trained policy, **render it and look**
  (`render-rollout`) — reward curves and eval sums have repeatedly lied here.
  The same rule for world mode: before claiming what happens in a room or on
  the pitch, `record-world` it and read `sheet.png` + `events.txt`.
- **Play an authored motion open loop before training it, and check the
  contact set and joint limits before authoring it at all.** Ramping the
  position servos from pose to pose costs two minutes and on Asimov's get-up it
  turned a six-phase skill into ONE unlearned transition — after four runs had
  been spent treating the whole motion as the problem. The same pass found that
  its wrists carry no collision geom (so it plants FOREARMS, and the G1's hand
  effectors would have authored four phases on thin air), that a ±0.35 rad
  ankle means it cannot squat below 81 % of standing, and that it cannot rise
  from supine at all. Judge a "hold" over 6 s: a crouch that sits still for two
  seconds collapses at three. §4b–4c of `.claude/skills/add-a-robot/SKILL.md`.
- If rollouts never contain the skill you're paying for, **fix the physics
  curriculum, not the reward** — an unsampled state's value is never learned.
  The full lesson (and the servo-ladder pattern that solved it) is in
  `microduck_local/AGENTS.md` under "Reward design rules".
- When a new task subclasses an existing one, **retarget an inherited reward
  term, never switch it off** — and check it isn't flat where the policy
  starts. Zeroing a weight because its default target is wrong for the new
  task is the most repeated mistake here (five retrains in one day); it leaves
  a proxy that failure also satisfies, and the robot sags, folds or topples
  into it. Same section of `microduck_local/AGENTS.md`.
