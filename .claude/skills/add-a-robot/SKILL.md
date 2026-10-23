---
name: add-a-robot
description: >-
  Bring somebody else's robot into this harness — their MJCF or URDF, their
  shipped policy, their published contract — and climb the levels from "on the
  stage" to "it trains" without inventing anything. The order that worked, the
  number that settles each step, and the 14 traps that were hit doing it
  for real. Use when adding a new body (a MuJoCo Menagerie model, a vendor's
  open-source robot, a second arm), when a fetched model will not build, when
  deciding what a new body must declare, and before changing anything under
  robots/. Trigger on: "add a robot", "bring in <robot>", "import this URDF",
  "new body", "get <robot> into the lab", "why won't this MJCF load",
  "what does a robot need to declare".
---

# Adding a robot

The ladder is `docs/mars-roadmap.md` §7.1: **level 0** on the stage, **1** it
trains, **2** it lives in a room, **3** honest about hardware. A body earns a
level by DECLARING things, and nothing is required that the level does not
need. This file is the route, measured on the four bodies that have done it —
and most sharply on Asimov 1 (2026-10-03), which is the only one whose owners
also published the training config their policy runs under.

**The rule that outranks the rest: never invent a number somebody published,
and never trust a number nobody measured.** Those are different failures and
both are fatal. A guessed observation layout produces a robot that moves and
falls over; an inherited reward window produces a term that never fires.

## 0. Before anything: the licence, and a pinned sha

Look at the licence FIRST — it decides whether the model can live in the tree
at all. Asimov 1 is GPL-2.0 (software) plus CERN-OHL-S-2.0 (hardware), both
strongly reciprocal, and this repo is Apache-2.0 and public, so the model is
**fetched, never vendored**. Its policy is BSD-3-Clause: the two halves of one
robot can be under different licences, and both get checked.

Fetch like `robots/asimov.py` / `robots/menagerie.py` do:

* **pin the revision** — a revision is a different robot. Asimov's pinned sha
  is literally `fix(sim-model): Update urdf and xml motion range to align with
  the hardware limits`; an earlier one has wider joint ranges and would teach a
  policy to ask for travel the servos do not have.
* **verify the bytes** against the listing's git blob sha1 — the only check
  that tells a moved branch, a truncated transfer and a captive portal's login
  page apart from the robot.
* **atomic replace, sha256 manifest** — a parallel worker must never import a
  half-written mesh, and a second fetch then costs no bytes.
* **bring the LICENCE file along and PRINT it.** A licence that stayed on
  GitHub is a licence nobody read.

## 1. Level 0: `MjcfBody`, then LOOK at it

```bash
uv run fetch-robot <id>
uv run python scripts/probe_mjcf_body.py <id> --out /tmp/probe   # then READ the PNG
```

`robots/mjcf_body.MjcfBody.from_mjcf` reads the joints, the default pose, the
spawn keyframe, the visual group, the vertex lattice and the measured stage
pitch off the compiled model. **Settles:** the body builds, and the picture is
the robot its README shows — standing on the floor, not folded into it, not
missing a limb.

Do not skip the picture. Every number `MjcfBody` reads is self-consistent by
construction: a keyframe that spawns the robot underground still yields a
perfectly valid joint table.

**Then look at it in the BROWSER too, because that is a different renderer.**
`probe_mjcf_body.py` is MuJoCo drawing the model with the model's own normals;
the lab ships a mesh dump and three.js draws that. The two disagreed for a year
and nobody checked — see trap 9.

## 2. Level 1: find the published contract, mirror it, cross-check TWO sources

If the robot has a shipped policy, its observation layout is **published, not
inferable**. Asimov's 78 floats put the twist command at `obs[6:9]` (before
the joints, where the duck's and the G1's sit last) and permute the 23 joints
into three slots that interleave left and right — because the slots are
**CAN buses**, not limbs, and each bus has its own sampling latency. Nobody
guesses that.

Mirror the constants into one file with each one citing where it came from
(`robots/asimov_walk.py`), and **cross-check two independent publications of
it** — Asimov's viewer config gives integer index lists, its Hub `env.yaml`
gives joint names, and the MJCF gives a joint order. All three agreeing is the
evidence; any one of them alone is a hope.

## 3. The decisive check: run THEIR policy in OUR env

```bash
uv run python scripts/probe_asimov_walk.py        # the shape to copy
```

**This is the step that makes everything downstream trustworthy, and it is the
one most likely to be skipped.** A wrong permutation still produces actions
from the graph and the robot still moves — it just falls over, and every
measurement taken afterwards describes a different robot.

**Settles:** their policy, in our MuJoCo, walks in the commanded direction at
roughly the commanded speed, deterministically (no obs noise, no action delay
— `AGENTS.md`, "Verify policies deterministically"). Asimov: 0.44 m/s at a 0.5
command, 0.68 at 0.8, ±0.5 rad/s turns, no falls in 10 s on six commands.

Then run the same policy through the ENV (not just a bare driver) and check
the two agree — that is what proves `_get_obs` and the driver write the same
78 floats. Measure in the robot's OWN frame: the env randomises spawn yaw, and
a world-frame displacement reads like a failure when it is a rotation.

## 4. Measure what cannot be read

`docs/mars-roadmap.md` §7.4 lists which `RobotSpec` fields a stranger's model
can supply. These are the ones it cannot, with how Asimov's were taken:

| field | how |
|---|---|
| `ground_tol` | run the shipped policy and record both soles' height difference. Asimov: median 10.3 mm, p90 24 mm → a 1 cm cut splits the gait near its own median |
| `fall_height` | let it go limp from the spawn pose. Asimov: pelvis settles at 0.110 m against 0.639 standing → 0.25 m is clear of both |
| `sole_tol` | the spread of the sole's contact geoms at STAND. Asimov: 0.00000 m, the sole is flat, so any tolerance picks the whole footprint |
| reward windows | **not yet, and do not inherit them.** The duck's air-time window paid on 0 % of G1 swings until re-measured |

A model may ship **`nu = 0`** — Asimov does, because its trainer adds the
actuators in Python. Inject them from the vendor's own gains, and treat that
as the single place the two stacks can silently disagree: a policy trained
against kp 150 and driven at MuJoCo's default flails, and the only symptom is
a robot that falls over.

### 4b. Before authoring any motion, read the CONTACT SET and the LIMITS

A level-3 body gets asked to do something — a get-up, a kneel, a reach — and the
first instinct is to author it from anatomy. Four facts about Asimov, each
measured in minutes, each of which invalidates a motion authored that way:

| what | how it was found | what it ruled out |
|---|---|---|
| **no hands** | list every geom with `contype != 0`. `left_wrist_yaw_link` has a visual mesh and nothing else | every phase that plants a hand. The load-bearing set was pelvis, waist, neck, hip links, the knee (a 0.13 m capsule down the whole shin), 4×5 mm foot spheres per ankle, shoulder rolls and the **elbow** — so it plants FOREARMS |
| **no squat** | `ankle_pitch` range ±0.35 rad, so the shin cannot tilt forward | any pose below 0.517 m on the feet alone — 81 % of standing |
| **no sit-up** | command the pose and read `joint_drift`: 0.35–0.58 rad, at 92 % of rated torque | the whole route that never turns over |
| **no static pose in a 0.22 m band** | settle a grid of candidates for 6 s | a phase state machine over held poses |

The cheapest version of all four: **`model.joint(n).range` and the collision
geom list, before a single pose is drafted.** The G1's effector table names
hands; Asimov has none, and copying it would have authored four phases resting
on thin air.

Two more that cost a measurement each:

* **Judge a "hold" over 6 s, not 2.** A knee-1.00 Asimov crouch sits at 0.511 m
  for two full seconds and collapses at three. A slow collapse and a hold are
  indistinguishable early, and the short window named a "deepest holdable pose"
  that does not hold.
* **A mirrored-joint helper must make a default a NO-OP.** The sign `DEFAULT_POSE`
  uses is not uniform across pairs — `left_knee` +0.45 against right -0.45, but
  `left_hip_pitch` -0.15 against right +0.15, and `left_shoulder_pitch` -0.25
  while `left_elbow` +0.40. One uniform sign swung both arms opposite to the
  default while the legs obeyed, and a crouch that holds read as collapsing to
  0.139 m. Define the kwarg as the LEFT side's value and assert the round trip.

### 4c. Play an authored motion OPEN LOOP before you train it

Ramp the position servos from waypoint to waypoint — no policy, no feedback.
Two minutes, and on Asimov it turned a six-phase get-up into **one** unlearned
transition: the chain follows the reference from `prone` to `rock_back` and
breaks on exactly one leg. Four training runs had already been spent treating
the whole motion as the problem.

Then make the boundary a test asserted from **both** sides: the legs before it
must still succeed, and the leg after it must still fail. A one-sided assertion
goes stale silently the moment the reference gets easier.
`tests/test_asimov_getup_ref.py` is the shape.

## 5. What a level-1 body owes

Beyond `RobotSpec`'s fields — this is the list §7.4 did not have, learned by
implementing them one failure at a time:

`contract()` · `env_class(task)` · `visual_scene()` · `robot_model()` ·
`frames()` · `shipped_policies()` · `ready()` / `fetch()` / `setup_hint()` ·
`attach(spec, prefix, frame)`

`robot_model()` is the cheapest of them and worth declaring first: it is what
lets `tests/test_body_conformance` count a new body's meshes **with no branch
of its own**. `attach()` must strip any floor and lights the model carries, or
a `/sim` room gets two floors.

## 6. Let the conformance suite find the rest

```bash
uv run --with pytest pytest tests/test_body_conformance.py -k <id>
```

It runs the generic cases on every body and the walker cases on legged ones.
On Asimov it caught four things in one pass — a missing `visual_scene`, a
`joint_groups` of the wrong SHAPE (five groups for 23 joints: nothing would
have crashed, the panel would have labelled 23 sliders from a 5-long list),
and two assumptions in the suite itself. Read a failure as a question about
the body first and about the suite second.

## 7. Make it look like the robot — four things, each with a number

A body that trains is not yet a body anyone can watch. `/sim` and the lab
stage draw it from a JSON dump of its meshes, and there are exactly three ways
that goes wrong. All three were hit on Asimov 1 in one afternoon, all three are
invisible to every test in the suite, and the only instrument for any of them
is opening the page (`.claude/skills/sim-smoke`, then hard-reload — a stale
module in the tab showed a robot that no longer existed, twice).

**a. The body owes `visual_scene()`.** Without it `GET /scene/robot/<id>` is a
404 and `/sim` draws the grey capsule placeholder. Build it from the ROBOT
file, not the scene: a scene carries the floor and the skybox and the viewer
draws its own. The whole method is one call —

```python
from .g1 import extract_visual_scene
from .mjcf_body import visual_geom_group, visual_vert_scale

model = mujoco.MjSpec.from_file(str(xml_path())).compile()
group = visual_geom_group(model)          # READ: 2 here, 1 for a URDF import
return extract_visual_scene(model, group=group, normals=True,
                            vert_scale=visual_vert_scale(model, group))
```

— and `@lru_cache` it, because the lab asks per slot and this dump is tens of
megabytes. **Settles:** `curl -s localhost:8788/scene/robot/<id> | head -c 200`
returns bodies, meshes and geoms rather than a 404.

**b. `normals=True`, always, for a CAD model.** The model's own per-vertex
normals (`mesh_normal`) are what MuJoCo draws with; without them the browser
welds by position and averages the faces, which averages across every hard
edge the author split. MEASURED on Asimov as the angle to the authored normal
per face corner: off by more than 15 deg on **62.2 %** of corners, more than
45 deg on 33.4 %, median 24.5 deg. That is the melted look, and it is a
dropped array rather than a resolution problem. Trap 9 has the general case.

**c. `vert_scale` is a RESOLUTION, and it is per robot.** Read it with
`mjcf_body.visual_vert_scale`, never inherit it. Asimov is the cautionary
tale: the first dump hard-coded the G1's millimetre on the reasoning that both
are G1-sized bipeds, which is the wrong axis entirely — the lattice has to
resolve the model's TRIANGLES, not its height. MEASURED, as the share of faces
whose normal the quantisation tilts past 15 deg:

| lattice | faces tilted >15 deg | dump |
|---|---|---|
| 1 mm, inherited from the G1 | **74.0 %** | 26.5 MB |
| 0.1 mm, MARS's | 24.8 % | 27.9 MB |
| 10 um, what `visual_vert_scale` picks | **0.6 %** | 29.3 MB |

His 10th-percentile triangle edge is 0.140 mm, so the helper lands on the
ladder's finest rung and 2.8 MB (+10.6 %) buys three quarters of the robot's
faces off an invented normal.

**d. A look module is what stops a new robot being flat grey.** b and c get
the SHAPE right and do nothing at all about the colour. Asimov's MJCF paints
all 26 geoms one `visual_grey` (rgba 0.7) from a defaults class, and the URDF
beside it has no `<color>` at all, so a correct dump of a correct mesh still
arrives as a mannequin. The fix is a table beside the other three —
`duck-viewer/components/<Robot>Look.tsx`, exporting `<ROBOT>_KINDS`, a
`<robot>PartKind(bodyName, meshName)` and a `use<Robot>Materials(enabled)`
hook — wired into `SimStage.RobotBody`'s `look` and into `personLook()`.

Two things that are easy to get wrong here:

* **Pass the look; do not let it default.** `RobotBody`'s `look` defaulted to
  `"g1"` because the G1 was the only robot that walked a room, so Asimov was
  drawn through the G1's part table, every part of him fell through to its
  pale shell, and the result was a uniform `#d8d8d8` robot that looked
  deliberate. A body with no look of its own wants `"generic"` — its geoms'
  own streamed rgba, which is right for a Menagerie model that paints its
  parts and useless for one that paints them all alike.
* **Split the parts by something you can MEASURE, and say which half is a
  guess.** Asimov's split is the link names, and the evidence is the meshes'
  bounding boxes: every structural shell is longer than 148 mm and every
  actuator housing shorter than 136 mm, a real gap in the model. The COLOURS
  are not measured and the module says so in its header — the model states one
  grey, so nothing in it can tell you what the machine is painted, and
  inventing a brand colour for a robot nobody here has seen is how a viewer
  starts misrepresenting somebody else's product.

**Settles:** the page, side by side with a body that already looks right, and
`components/<Robot>Look.test.ts` asserting every link lands on a kind the
material array actually has — `indexOf` falls back to 0 on a miss, so a typo
in a rule paints a part as a shell and nothing else says a word.

### 7e. Two colours on ONE link: cut the SOURCE mesh

A link gets one colour, until you cut its mesh. The robot is two colours in
places the model is not — a red chest over a black back, a magenta sole under
a black shoe — and the fix is a real one, already in this repo since MARS.
**It cost a day on Asimov by being done from memory instead of from
`mars.split_base_mesh`.** The order that works:

0. **Look for the precedent first.** `grep -rn "split_.*mesh\|_add_cut_geom"
   robots/` finds MARS cutting `base.STL` into shell + tyres. Read the
   helpers, not just the caller — reading `_add_cut_geom` and skipping
   `_read_binary_stl` is what made this look novel when it was not.
1. **Pick the KIND of cut from the part, not from habit.** There are three,
   and `keep(centroids, shells, normals)` gets what each needs:
   * **connected shell** — the part is its own piece of geometry. Asimov's
     visor is one of the head's five shells; MARS's tyres are two of its
     base's 25. Cleanest of the three: the boundary is the model's own.
   * **face normal** — the part is a face of a curved surface. The sole is
     the shoe's downward-facing triangles, and its edge lands on the welt.
   * **plane** — only when neither fits, and only where the surface is
     already going that way. **A plane across a curved surface serrates**:
     `z < -28 mm` through Asimov's sole left a row of magenta teeth visible
     from across the stage, and a 170 mm floor tore the chest cowl. The vest
     uses one plane on top of a shell cut, where the cowl curves under.
   Combine them freely — the vest is shell + plane, the sole is shell +
   normal. If none expresses the part, **stop and say so**: Asimov's bronze
   knee disc is a raised boss on a curved shell and defeated six attempts.
2. **Resolve the SOURCE file** through the MJCF's asset entry and `meshdir`,
   never by guessing from the link name — Asimov's mesh asset is
   `waist_yaw_link.STL` and the file it names is `WAIST_YAW.STL`.
3. **Measure the threshold in THAT file's frame**, and print the extents
   first. A number read off `model.mesh_vert` is in a different frame: a
   −74 mm sole plane missed a foot spanning −34..+23 mm, selected nothing,
   and made `walk_spec()` raise mid-session with a trainer running.
4. **Split the file's triangles by centroid**, write both halves atomically,
   and refuse an empty or total selection — that guard is what turns a
   wrong-frame threshold into a sentence instead of a mystery.
5. **Cache on the source mesh's mtime AND the module's.** The predicate lives
   in the module, so a source-only key silently reuses the old halves and the
   edit looks like it did not apply.
6. **Add two visual geoms**: `contype`/`conaffinity`/`density` 0, material
   cleared, and **no position compensation** — the halves are in the file's
   frame, so MuJoCo places them exactly as it placed the whole mesh.

**Settles, in this order:**

* every visual body's world AABB against the UNCUT model, at `mj_resetData`
  (qpos0) on both sides — 4e-6 mm on Asimov. Not "keyframe 0": a spec that
  adds its own STAND key makes that two different poses and reports movement
  in links nobody touched.
* the physics is untouched: `body_mass`, `body_inertia`, `dof_armature` and
  the collision-geom list identical, and a 1500-step rollout BIT-identical.
* then LOOK at it, beside the vendor's own photograph.

`tests/test_asimov_look.py` is the shape for all three, each case A/B'd
against a planted regression.

## 8. The traps, all hit for real

1. **A model that bundles its own scene.** Asimov's MJCF carries a `floor`
   plane and three lights, so the generated scene added a second `floor` and
   nothing compiled. Level 0's "an MJCF and an id" held only for files shaped
   like Menagerie's until this was fixed generically.
2. **`joint_groups` is one label PER JOINT**, not per group.
3. **A tall body cropped by a width-framed camera.** `lab_spacing_m` is 3.52x
   the widest HORIZONTAL extent — the right quantity for two robots side by
   side, the wrong one for a picture of a 1.2 m biped.
4. **An unfetched body asked for its contract** and took the whole
   `fetch-robot` listing down with it.
5. **Wrapped yaw reads like a stall.** A 0.6 rad/s turn held for 10 s is
   4.79 rad, which prints as −1.49.
6. **A world-frame displacement reads like a drift** when the env randomised
   the spawn yaw. Measure in the robot's own frame.
7. **"Legged bodies topple open-loop" is a fact about the duck and the G1, not
   about legs.** Asimov holds its default pose for 2 s with no policy at all —
   32 kg on 150 N·m/rad servos in a statically stable stance. A test that
   asserts the general version will fail on the next robot, and the failure is
   good news.
8. **Two halves, two licences.** The model and the policy were not under the
   same terms.
9. **The browser is a second renderer, and the mesh dump used to throw away
   half of what it needs** (2026-10-03, found on the Go2 — "this robot looks
   like it's got a messed up model"). MuJoCo draws a CAD mesh with the
   normals the file ships (`mesh_normal`, per vertex); the dump carried only
   positions and faces, so the viewer welded by position and averaged the
   faces to invent normals. MEASURED, as the angle to the authored normal per
   face corner: **off by more than 15 deg on 13.9 % of the Go2's corners,
   55.0 % of the SO-ARM100's, and 78.3 % of MOSS's** (median 37.8 deg). That
   is the melted, dented, cloudy look, and no lattice fixes it. Two things to
   check on a new body:
   * **it ships normals** — `extract_visual_scene(..., normals=True)`, which
     every `MjcfBody` does. A model whose normals are not per-vertex is
     refused rather than bodged, and the viewer invents them as before.
   * **its normals are not inverted.** Real exports carry some: 5.29 % of
     MOSS's vertices point into the solid. MuJoCo lights both sides and never
     shows it; three.js renders them black. `_authored_normals` turns the SIGN
     round against the geometry and keeps the author's direction.
10. **The vertex lattice is a resolution, and the default was one robot's.**
   The dump quantises vertices to integers; 1 mm was taken on a 1.3 m G1 and
   then inherited by everything. On the Go2 it merged 44.8 % of distinct
   vertices and shredded the panel seams and the `Go2` lettering.
   `visual_vert_scale` now reads it off the model's own tessellation — a
   tenth of the **10th-percentile** triangle edge, not the median, because
   shading is normals and a normal comes off one triangle. 18 % more bytes.
   Both halves were A/B'd on the stage: authored normals alone still shred
   the lettering, a fine lattice alone still leaves the shells cloudy.
11. **A correct mesh is still a flat grey robot, and a default look hides
   it.** Traps 9 and 10 are about SHAPE and neither does anything about
   colour: Asimov's MJCF paints all 26 geoms one `visual_grey` from a defaults
   class, so he arrived uniform `#d8d8d8` with his shells, his actuator
   housings and his soles indistinguishable. Two separate causes, and the
   second is the one that reads as deliberate — `SimStage.RobotBody` defaults
   `look` to `"g1"`, so a new body is silently drawn through ANOTHER robot's
   part table, every part falls through to that robot's shell colour, and the
   result looks like a choice rather than a gap. §7d is the fix: a
   `components/<Robot>Look.tsx` beside the other three, and the look PASSED
   rather than defaulted. Split the parts by something measurable (Asimov's
   is link names backed by mesh bounding boxes — shells over 148 mm,
   housings under 136 mm) and keep the colours restrained: the model states
   one grey, so the palette is a look, not the vendor's spec, and the module
   has to say which half is which.


**Two colours on one link means cutting the SOURCE STL — not
   `model.mesh_vert`** (2026-10-04, Asimov). `mesh_vert` is MuJoCo's
   RECENTRED copy; writing it back subtracts the recentre twice and the
   halves land where the original never was (torso down 18 cm, neck column
   gone, head floating). Four corrections were measured against that wrong
   premise and all four failed. Split the file on disk — `mars.split_base_mesh`
   has always done this — and the halves land within 4e-6 mm with no
   compensation. Thresholds are then in the FILE's frame, which is not the
   compiled one: a −74 mm plane read off `mesh_vert` missed a foot spanning
   −34..+23 mm entirely and made the model stop building.
13. **Diff geometry at `mj_resetData` (qpos0), never at "keyframe 0".** A
   spec that adds its own STAND keyframe makes "keyframe 0" two different
   poses, and links nobody touched then read as moved by tens of mm — which
   is how the trap above was first misdiagnosed.
14. **A visible anomaly in your own render is evidence.** That floating
   head appeared in two renders before the user pointed at it, and both times
   it was explained away as camera framing.

## 9. Level 2: a room, through `WorldPerson` and not `WorldRobot`

The obvious-looking seam is wrong, and it cost an afternoon of staring at the
wrong class. `world/arena.WorldRobot` is for DRIVER-STEPPED WHEELED bodies —
it says "a planar base cannot topple" and keeps `falls` at 0 for the life of a
run. A legged body in a room is a **`WorldPerson`**: that is how the G1 has
always walked a scenario, on a waypoint tour with its twist as the command,
and it already handles falling, yielding and being followed.

So level 2 for a walker is:

```json
{"persons": [{"id": "p0", "kind": "asimov1", "pos": [1.6, -1.6], "speed": 0.25,
              "path": [[1.6, 1.6], [-1.6, 1.6], [-1.6, -1.6], [1.6, -1.6]]}]}
```

`kind` is a **registry id**, resolved through `Body.attach()` and
`Body.driver()`. What a driver owes `WorldPerson` is `G1Walker`'s surface,
named: `cmd`, `spawn`, `pose`, `fallen`, `control`, `base_body_id`,
`body_ids`, `scene_bodies` and `bodies_payload`. Note `control` writes `ctrl`
and does **not** step — the world owns the clock and runs
`substeps = CTRL_DT / room timestep` physics steps per tick, so a room on a
2 ms clock still ticks a 50 Hz policy at 50 Hz.

**Settles:** `uv run record-world scenarios/<name>.json --camera follow:p0`,
then READ the sheet. Asimov walks the full 12.8 m circuit in ~40 s at 0.25 m/s
with the pelvis steady at 0.62 m and no falls.

Two things worth knowing before you look at the picture:

* **A top-down camera over a 8x7 m floor hides a 1.2 m robot's motion.** I
  read the first sheet as "it never left its spawn" twice; a trace of
  `(x, y, yaw)` showed it walking the whole lap. Follow the subject, and
  trust the numbers over the thumbnail.
* **Speed is a level-design parameter, not a default.** At 0.5 m/s the Asimov
  simply outruns a 25 cm duck, and the ducks that look "stuck" in the sheet
  are the ones left behind. 0.25 m/s makes a followable tour.

## 10. Still open

Nothing blocking a walker. A legged body still enters a room as a *person*
rather than a roster entry — the day one needs sensors, brains and an
`eval-*` battery of its own is the day it declares `frames()` and becomes a
`WorldRobot`, and that class's wheeled assumptions are what will need the
look then.
