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
  cannot kneel is measured separately and is the ankle: in every legal
  configuration the FOOT touches 72-115 mm before the shin, because a 20 deg
  ankle cannot point the toe out of the way.
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
