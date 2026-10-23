"use client";

// The 3D stage: fetches /scene once, streams frames over WS, renders every
// duck on a shared floor. MuJoCo is Z-up; everything sim-space lives inside
// one group rotated -90° about X so three.js Y-up lighting/controls stay sane.
// Rendering is kept deliberately light (no shadow maps, merged geometry,
// capped DPR) — the first version lost the WebGL context.

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import {
  duckRowKeys,
  LabClient,
  fetchScene,
  LAB_HTTP,
  robotsInFrame,
  SceneLoads,
  type DuckFrame,
  type RobotId,
  type Scene,
} from "@/lib/lab";
import { assignDrag, nearestDuck, type AssignTarget } from "@/lib/assign";
import {
  cameraKeyDown,
  cameraKeyUp,
  cameraKeysClear,
  DUCK_HOME,
  DUCK_SUBJECT_M,
  homeView,
  recordShot,
  requestViewReset,
  stageMaxDist,
  subjectSizeM,
  type Shot,
} from "@/lib/camera";
import { loadJSON, saveJSON } from "@/lib/persist";
import { getCapture } from "@/lib/record";
import { getSelectedDuck, setSelectedDuck } from "@/lib/select";
import { modalIsOpen } from "@/lib/ui";
import { buildBodyGeometries, Duck } from "./Duck";
import { robotLook } from "@/lib/robots";
import CameraKeys, { type ControlsLike } from "./CameraKeys";
import { useTruckSwipe } from "./useTruckSwipe";
import { Hud } from "./Hud";
import { PolicyPanel } from "./PolicyPanel";
import { TeachPanel } from "./TeachPanel";
import { pushToast, Toasts } from "./Toasts";
import { AnimPanel } from "./AnimPanel";
import { RecordPanel } from "./RecordPanel";
import { CaptureCanvas, Snapshotter } from "./Capture";
import { PoseDuck } from "./PoseDuck";
import StageFog from "./StageFog";

/** FALLBACK layout only — one duck-sized pitch for the whole roster. The
 *  server sends each slot its own offset now (DuckFrame.offset), because the
 *  pitch belongs to the robot: this 0.65 m is the duck's, and six 1.3 m G1
 *  helpers laid out on it stood inside each other. Kept for a server that
 *  predates the field. */
function gridOffsets(n: number, spacing = 0.65): [number, number][] {
  const cols = Math.ceil(Math.sqrt(n));
  return Array.from({ length: n }, (_, i) => [
    (i % cols) * spacing - ((Math.min(n, cols) - 1) * spacing) / 2,
    Math.floor(i / cols) * spacing,
  ]);
}

/** Slot positions for one frame: the server's per-robot layout where it sends
 *  one, the duck-pitched grid where it does not. */
function frameOffsets(ducks: Pick<DuckFrame, "offset">[]): [number, number][] {
  const grid = gridOffsets(ducks.length);
  return ducks.map((d, i) => d.offset ?? grid[i]);
}

function Ducks({
  scene,
  scenes,
  client,
}: {
  scene: Scene;
  /** Mesh sets by robot id. The duck's is always present; another body's
   *  arrives once a roster row says it is that robot (GET /scene?robot=). */
  scenes: Partial<Record<RobotId, Scene>>;
  client: LabClient;
}) {
  const bodies = useMemo(() => buildBodyGeometries(scene), [scene]);
  // One geometry set per robot on the stage, built once — not per duck. The
  // duck's is built here as well as above: `bodies` still backs any row whose
  // mesh set has not arrived yet (and every server that predates robot
  // selection, where no row carries one).
  const bodiesByRobot = useMemo(() => {
    const out: Partial<Record<RobotId, ReturnType<typeof buildBodyGeometries>>> = {};
    for (const [id, s] of Object.entries(scenes)) {
      if (!s) continue;
      try {
        // The body's own look: "duck" merges and vertex-colours as it always
        // has, "g1" and "mars" take their material tables, and every OTHER
        // body is "generic" — welded, smoothed, painted from the scene dump's
        // own rgba. That is the line that puts a Menagerie model on the stage
        // without a component of its own (lib/robots.robotLook).
        const look = robotLook(id);
        out[id as RobotId] =
          look === "duck" ? buildBodyGeometries(s) : buildBodyGeometries(s, null, { look });
      } catch (e) {
        // A mesh set that fails to build must not take the stage with it:
        // this runs during render, so an uncaught throw here would unmount
        // the whole Canvas — every duck AND the floor — over one robot.
        console.error(`[viewer] ${id} meshes failed to build`, e);
      }
    }
    return out;
  }, [scenes]);
  // Roster keyed by the STABLE stream id — a policy assign renames a duck,
  // which must update its label without remounting (and re-lerping) it.
  // (`key` is the id dedup-qualified by duckRowKeys: a roster with duplicate
  // ids — seen with legacy lab-state restores — must not collide React keys.)
  const [roster, setRoster] = useState<
    {
      id: string;
      name: string;
      key: string;
      robot: RobotId;
      offset: [number, number];
    }[]
  >([]);
  const rosterSig = useRef("");
  const duckRefs = useRef(new Map<string, React.MutableRefObject<DuckFrame | null>>());

  // Fan the single frame out into per-duck refs (no React re-render per frame).
  useFrame(() => {
    const f = client.frame;
    if (!f) return;
    // The robot is part of the signature: a slot that changes body must
    // re-render with the other mesh set, and the name alone need not change.
    const sig = f.ducks.map((d) => `${d.id}\t${d.name}\t${d.robot ?? ""}`).join("\n");
    if (sig !== rosterSig.current) {
      rosterSig.current = sig;
      const keys = duckRowKeys(f.ducks);
      // Offsets ride in the roster state, not a per-frame recompute: the
      // layout can only change when the roster does, and every input to it
      // (the row count and every row's robot) is in the signature above.
      const offs = frameOffsets(f.ducks);
      setRoster(
        f.ducks.map((d, i) => ({
          id: d.id,
          name: d.name,
          key: keys[i],
          robot: (d.robot as RobotId) || "microduck",
          offset: offs[i],
        })),
      );
      // A removed duck must not stay "selected" — the Delete key would then
      // fire remove_duck at a ghost id forever.
      const sel = getSelectedDuck();
      if (sel && !f.ducks.some((d) => d.id === sel)) setSelectedDuck(null);
    }
    f.ducks.forEach((d) => {
      const r = duckRefs.current.get(d.id);
      if (r) r.current = d;
    });
  });

  return (
    <>
      {roster.map((d) => {
        let ref = duckRefs.current.get(d.id);
        if (!ref) {
          ref = { current: null };
          duckRefs.current.set(d.id, ref);
        }
        // A row whose mesh set has NOT arrived yet draws nothing at all.
        // Falling back to the duck's meshes drew a G1 as a scatter of duck
        // parts at humanoid joint positions — the poses are streamed in that
        // robot's own body order, so the wrong set is not a rough
        // approximation, it is debris. The duck itself always has its scene
        // (it is fetched at mount), so only another body can wait.
        const geo = d.robot === "microduck" ? bodies : bodiesByRobot[d.robot];
        if (!geo) return null;
        return (
          <Duck
            key={`${d.key}:${d.robot}`}
            duckId={d.id}
            bodies={geo}
            frameRef={ref}
            offset={d.offset}
            label={d.name}
          />
        );
      })}
    </>
  );
}

// Inside-the-Canvas helper: every frame, project each duck's trunk (frame body
// index 1) to client px and publish into the shared assign store. The scene
// group is rotated -90° about X, so MuJoCo (x, y, z) sits at three.js world
// (x, z, -y); each duck additionally carries its grid offset in MuJoCo XY.
// --- camera persistence ------------------------------------------------------

type Vec3Tuple = [number, number, number];
interface SavedCamera {
  p: Vec3Tuple; // camera position (three.js Y-up world)
  t: Vec3Tuple; // OrbitControls target
}

function isVec3(v: unknown): v is Vec3Tuple {
  return (
    Array.isArray(v) &&
    v.length === 3 &&
    v.every((n) => typeof n === "number" && Number.isFinite(n))
  );
}

/** Last saved camera pose, or null (first visit / bad data). */
function loadSavedCamera(): SavedCamera | null {
  const raw = loadJSON<Partial<SavedCamera> | null>("camera", null);
  return raw && isVec3(raw.p) && isVec3(raw.t) ? { p: raw.p, t: raw.t } : null;
}

// The home view and the dolly ceiling are the DUCK's (lib/camera.DUCK_HOME,
// `stageMaxDist`'s 7 m) and both are scaled by the biggest body on the stage —
// `stageSize` below. A fixed 1.93 m reset put a 1.296 m Asimov's head out of
// frame, which is the same duck-sized-metres bug the fog band had.
//
// ONE dolly clamp for the stage, read by both the mouse (OrbitControls) and
// the keyboard (CameraKeys) — /sim already shares its pair this way. They were
// two separate literals here and had drifted apart (0.3/7 against 0.25/8),
// which is what made W stop dead: see `effectiveClamp` in lib/camera.
//
// The NEAR end does not scale: 0.3 m is how close you may get to inspect a
// duck's beak, and it is how close you may get to inspect an Asimov's hand.
const CAM_MIN_DIST = 0.3;

/** How big the biggest body on the stage is, metres — what the home view and
 *  the dolly ceiling are sized by. The largest, not the selected one, because
 *  a reset has to frame the whole roster and a roster can be mixed. A robot
 *  whose mesh set has not arrived answers the duck's size, which is what the
 *  stage was framed for before any of this. */
function stageSize(
  robots: readonly RobotId[],
  scenes: Partial<Record<RobotId, Scene>>,
): number {
  let big = DUCK_SUBJECT_M;
  for (const r of robots) big = Math.max(big, subjectSizeM(scenes[r]?.size));
  return big;
}

/** Inside-the-Canvas helper: persist the camera pose after every orbit/pan/zoom
 *  gesture (OrbitControls "end"). Restore happens via the Canvas/OrbitControls
 *  initial props, so there is no visible jump on load. */
function CameraPersistence() {
  const controls = useThree((s) => s.controls) as unknown as ControlsLike | null;
  const camera = useThree((s) => s.camera);
  useEffect(() => {
    if (!controls) return;
    const save = () =>
      saveJSON("camera", {
        p: camera.position.toArray(),
        t: controls.target.toArray(),
      });
    controls.addEventListener("end", save);
    return () => controls.removeEventListener("end", save);
  }, [controls, camera]);
  return null;
}

// 🎥 follow-cam shot parameters live in lib/camera (DUCK_SHOT): a ¾ front
// view (±az off the subject's heading), distance/height sized so the subject
// fills about half the frame at the stage's 40° fov, plus a very slow azimuth
// drift so long takes stay alive. `recordShot` scales all of it by how big
// the robot BEING FILMED is — the duck's 0.78 m lens is inside an Asimov.

const wrapAngle = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

/** Inside-the-Canvas helper for the 🎥 record flow (the canvas itself is
 *  registered and pumped by CaptureCanvas, Capture.tsx): while a capture is
 *  framing/recording flies the camera to a ¾ front view of the target duck
 *  and keeps it centered.
 *  OrbitControls is paused for the duration (CameraKeys takes a `paused`
 *  callback and drops held motions while it is true), and
 *  the camera simply stays where the take ended. The azimuth is chosen ONCE
 *  per take — the duck's heading rotated ±shot.az toward whichever side the
 *  camera already sits — not tracked per frame: a backflipping duck's heading
 *  whips 180° mid-roll and would slingshot the camera around it.
 *
 *  So is the DISTANCE, and for the same reason it is chosen at all: the shot
 *  is scaled by the size of the body in the slot (`Scene.size`, measured
 *  lab-side), because one set of metres tuned on a 25 cm duck put the lens
 *  inside a 1.3 m Asimov. Picking it per take rather than per frame means a
 *  mesh set that arrives mid-take cannot re-frame a rolling shot. */
function RecordCamera({
  client,
  scenes,
  maxRadius,
}: {
  client: LabClient;
  /** Mesh sets by robot id — `size` is read off whichever one the filmed
   *  slot is, and its absence frames at the duck's size (what every take did
   *  before this). */
  scenes: Partial<Record<RobotId, Scene>>;
  /** The stage's live dolly ceiling (`stageMaxDist`), which OrbitControls
   *  re-clamps to inside the `controls.update()` below whether or not the
   *  controls are enabled. */
  maxRadius: number;
}) {
  const controls = useThree((s) => s.controls) as unknown as ControlsLike | null;
  const camera = useThree((s) => s.camera);
  // Never leave the user without controls (unmount mid-take).
  useEffect(() => {
    return () => {
      if (controls) controls.enabled = true;
    };
  }, [controls]);

  const shot = useRef<{ epoch: number; az: number; t0: number; cam: Shot } | null>(
    null
  );
  const paused = useRef(false);
  const aim = useMemo(() => new THREE.Vector3(), []);
  const desired = useMemo(() => new THREE.Vector3(), []);
  const fwd = useMemo(() => new THREE.Vector3(), []);
  const q = useMemo(() => new THREE.Quaternion(), []);

  useFrame((st, dtRaw) => {
    const cap = getCapture();
    const active = cap.phase === "framing" || cap.phase === "recording";
    if (!active) {
      if (paused.current && controls) controls.enabled = true;
      paused.current = false;
      shot.current = null;
      return;
    }
    if (controls && !paused.current) {
      controls.enabled = false;
      paused.current = true;
    }
    const f = client.frame;
    const idx = f ? f.ducks.findIndex((d) => d.id === cap.duckId) : -1;
    const row = idx >= 0 ? f!.ducks[idx] : undefined;
    const trunk = row?.bodies[1];
    if (!f || !trunk) return; // duck vanished mid-take — hold the last shot
    const off = frameOffsets(f.ducks)[idx];
    // MuJoCo (x, y, z) → three world (x, z, -y), plus the duck's grid offset.
    aim.set(trunk[0] + off[0], trunk[2] + 0.02, -(trunk[1] + off[1]));

    const now = st.clock.elapsedTime;
    if (!shot.current || shot.current.epoch !== cap.epoch) {
      // Duck heading on the floor, as a three-world azimuth (atan2(x, z)).
      q.set(trunk[4], trunk[5], trunk[6], trunk[3]); // wxyz → xyzw
      fwd.set(1, 0, 0).applyQuaternion(q); // trunk +x = forward, MuJoCo frame
      const headingAz = Math.atan2(fwd.x, -fwd.y);
      const camAz = Math.atan2(
        camera.position.x - aim.x,
        camera.position.z - aim.z
      );
      // How big this slot's robot is — the duck's own size when the roster
      // row names no body, or its mesh set has not arrived.
      const size = scenes[row!.robot ?? "microduck"]?.size;
      // The ceiling is passed because OrbitControls re-clamps to it inside
      // the `controls.update()` below, enabled or not (lib/camera.recordShot).
      const cam = recordShot(subjectSizeM(size), maxRadius);
      // ¾ view on whichever side needs the shorter glide.
      const side =
        Math.abs(wrapAngle(headingAz + cam.az - camAz)) <=
        Math.abs(wrapAngle(headingAz - cam.az - camAz))
          ? 1
          : -1;
      shot.current = {
        epoch: cap.epoch,
        az: wrapAngle(headingAz + side * cam.az),
        t0: now,
        cam,
      };
    }
    const { cam } = shot.current;
    const az = shot.current.az + cam.drift * (now - shot.current.t0);
    desired.set(
      aim.x + Math.sin(az) * cam.dist,
      aim.y + cam.height,
      aim.z + Math.cos(az) * cam.dist
    );

    const dt = Math.min(dtRaw, 0.1); // tab-stall guard, like CameraKeys
    camera.position.lerp(desired, 1 - Math.exp(-3.5 * dt));
    if (controls) {
      controls.target.lerp(aim, 1 - Math.exp(-6 * dt));
      camera.lookAt(controls.target);
      controls.update?.();
    } else {
      camera.lookAt(aim);
    }
  });
  return null;
}

function AssignTargets({ client }: { client: LabClient }) {
  const { camera, gl } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const f = client.frame;
    if (!f) {
      assignDrag.targets = [];
      assignDrag.hoverDuck = null;
      return;
    }
    const offsets = frameOffsets(f.ducks);
    const rect = gl.domElement.getBoundingClientRect();
    const targets: AssignTarget[] = f.ducks.map((d, i) => {
      const t = d.bodies[1] ?? [0, 0, 0];
      v.set(t[0] + offsets[i][0], t[2], -(t[1] + offsets[i][1]));
      v.project(camera);
      return {
        id: d.id,
        x: rect.left + ((v.x + 1) / 2) * rect.width,
        y: rect.top + ((1 - v.y) / 2) * rect.height,
        visible: v.z < 1, // in front of the camera
      };
    });
    assignDrag.targets = targets;
    // Live highlight while a chip is dragged or armed (drop handlers redo the
    // lookup from the actual event position, so this is presentation-only).
    assignDrag.hoverDuck = assignDrag.mode
      ? nearestDuck(assignDrag.px, assignDrag.py)
      : null;
  });
  return null;
}

export default function Viewer() {
  const [scene, setScene] = useState<Scene | null>(null);
  // Mesh sets for every robot the roster is currently showing.
  const [scenes, setScenes] = useState<Partial<Record<RobotId, Scene>>>({});
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Read once on mount (this component is ssr:false, so storage is available).
  const [savedCam] = useState(loadSavedCamera);
  const clientRef = useRef<LabClient | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  // Two-finger horizontal swipe → the same lateral truck as A/D.
  useTruckSwipe(rootRef);

  // Load a robot's mesh set the first time a roster row says it is that
  // robot. Polled off the client's own frame (1 Hz) rather than subscribed:
  // the frame is a mutable ref read per-frame inside the Canvas, and a
  // React subscription to it would re-render the whole stage at 50 Hz.
  //
  // Mounted ONCE, with its bookkeeping in refs. Keyed on `scenes` instead,
  // every arriving scene tore the effect down and re-ran it — and a G1 fetch
  // in flight at that moment (21 MB, ~2.5 s) resolved into a dead closure and
  // was thrown away. The stage then drew a G1 with the duck's meshes.
  // One object, so the "already loaded" gate and the stored scene can never
  // disagree again — see SceneLoads in lib/lab.
  const sceneLoads = useRef(new SceneLoads());
  // When a robot first appeared in a frame, so a stage that never draws it
  // can SAY SO. See the warning in the tick below.
  const robotSeenAt = useRef(new Map<string, number>());
  const warnedMissing = useRef(new Set<string>());
  // Which bodies are on the stage right now, as STATE — the home view and the
  // dolly ceiling are sized by the biggest of them, and both are props. Set
  // from the same 1 Hz tick that loads their meshes, and only when the set
  // actually changes, so the roster cannot re-render the stage at frame rate.
  const [stageRobots, setStageRobots] = useState<RobotId[]>([]);
  useEffect(() => {
    let stopped = false;
    const tick = () => {
      const f = clientRef.current?.frame;
      if (!f) return;
      const onStage = robotsInFrame(f.ducks);
      setStageRobots((prev) =>
        prev.length === onStage.length && prev.every((r, i) => r === onStage[i])
          ? prev
          : onStage
      );
      for (const robot of robotsInFrame(f.ducks)) {
        // The retry budget for a lab without that robot's assets lives in
        // SceneLoads; so does the rule that only a STORED scene closes the
        // gate. Marking it loaded here and dropping the scene below left the
        // robot with no mesh set for the life of the page.
        if (!sceneLoads.current.wants(robot)) continue;
        sceneLoads.current.begin(robot);
        fetchScene(robot)
          .then((s) => {
            if (stopped) {
              // Torn down mid-flight (Strict Mode's double-mount, a hot
              // reload). Drop the scene, but leave the robot FETCHABLE so
              // the next mount's tick picks it straight back up.
              sceneLoads.current.dropped(robot);
              return;
            }
            sceneLoads.current.stored(robot);
            setScenes((prev) => ({ ...prev, [robot]: s }));
          })
          .catch(() => sceneLoads.current.failed(robot));
      }
      // A stage with bodies on the roster and none on screen is SILENT: no
      // error, no failed request, a green "live" badge and a HUD listing
      // every policy with moving counters. Say which robot has no mesh set
      // and which state it is stuck in, once, after it has had long enough.
      const now = Date.now();
      for (const robot of robotsInFrame(f.ducks)) {
        if (!robotSeenAt.current.has(robot)) robotSeenAt.current.set(robot, now);
        const waited = now - (robotSeenAt.current.get(robot) ?? now);
        const { state, failures } = sceneLoads.current.describe(robot);
        if (state === "ready" || waited < 8000 || warnedMissing.current.has(robot)) {
          continue;
        }
        warnedMissing.current.add(robot);
        console.warn(
          `[viewer] "${robot}" has been on the stage for ${Math.round(waited / 1000)}s ` +
            `with NO mesh set (scene ${state}${failures ? `, ${failures} failed fetch(es)` : ""}). ` +
            `Its bodies cannot draw. ` +
            // Failures first: a retry can be in flight at the moment of the
            // check, which reports "fetching" over the top of four 404s and
            // sends the reader looking at download size instead of the lab.
            (failures > 0
              ? `GET ${LAB_HTTP}/scene?robot=${robot} is failing — does this lab have that robot's assets?`
              : state === "fetching"
                ? `The fetch has not come back; the scene is several MB, but 8s is long.`
                : `Nothing has requested it — that is a bug in this loader, not the lab.`),
        );
      }
    };
    const id = setInterval(tick, 1000);
    tick();
    return () => {
      stopped = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    const client = new LabClient(setConnected);
    clientRef.current = client;
    const load = () =>
      fetchScene()
        .then((s) => {
          setScene(s);
          setScenes((prev) => ({ ...prev, microduck: s }));
          setError(null);
        })
        .catch(() => {
          setError("duck-lab server not reachable on :8788 — start it with `uv run duck-lab …`");
          setTimeout(load, 2000);
        });
    load();

    // Drive keys. Registered on window in the *capture* phase so they work no
    // matter what has focus (canvas, OrbitControls, dev overlay) and nothing
    // downstream can swallow them first — but never while the user is TYPING,
    // and never with a browser shortcut modifier held (⌘R must reload, not
    // reset ducks).
    //
    // Only TEXT-ENTRY targets count as typing. A focused slider (the teach
    // panel's reward-weight <input type="range">) or a just-clicked pad
    // <button> must NOT eat W/A/S/D/Q/E/X/R — that stranded the keyboard
    // until the user clicked the bare canvas.
    const NON_TEXT_INPUTS = new Set([
      "range", "checkbox", "radio", "button", "submit", "reset",
      "color", "file", "image",
    ]);
    const isTyping = (t: EventTarget | null) => {
      if (!(t instanceof HTMLElement)) return false;
      if (t.isContentEditable) return true;
      if (t.tagName === "TEXTAREA" || t.tagName === "SELECT") return true;
      if (t instanceof HTMLInputElement) return !NON_TEXT_INPUTS.has(t.type);
      return false;
    };
    // Arrow keys are drive aliases — but many form controls consume arrows
    // natively (range sliders, selects, radio groups, text carets), so arrows
    // only drive from "neutral" focus (wrapper, canvas, body, plain buttons).
    const arrowsBelongToTarget = (t: EventTarget | null) =>
      t instanceof HTMLElement &&
      (t.tagName === "INPUT" ||
        t.tagName === "TEXTAREA" ||
        t.tagName === "SELECT" ||
        t.isContentEditable);
    // The keyboard flies the CAMERA (game-editor style) — the user teaches
    // ducks through RL, never by teleop, so no drive commands here. keydown
    // begins a motion, keyup ends it; CameraKeys (inside the Canvas, where
    // OrbitControls lives) integrates held motions × dt for smooth flow.
    // The exception is R: restarting the sim is the one action worth a bare
    // key, because a side-by-side comparison is only legible when every duck
    // starts its episode at the same moment. The view reset yields to Shift+R.
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      // A dialog owns the keyboard while it is open. isTyping() is not enough:
      // a dialog that focuses a button (both of ours do) would still let
      // Backspace remove the selected duck and `r` reset every episode behind
      // the overlay. The dialog cannot stop us from its own listener — two
      // window/capture listeners cannot suppress each other — so the scene
      // stands down voluntarily.
      if (modalIsOpen()) return;
      if (isTyping(e.target)) return;
      const arrow = e.key.startsWith("Arrow");
      if (arrow && arrowsBelongToTarget(e.target)) return; // slider keeps its arrows
      // Delete/Backspace removes the SELECTED duck (click a duck to select).
      // Backspace must be claimed too — un-prevented it can navigate back.
      if (e.key === "Delete" || e.key === "Backspace") {
        const sel = getSelectedDuck();
        if (!sel) return;
        e.preventDefault();
        if (e.repeat) return;
        const training = clientRef.current?.frame?.training;
        // Mirror the HUD-row rules: the server refuses these anyway, but a
        // keypress that silently does nothing reads as broken.
        if (sel === "trainee" && (training?.status === "training" || training?.restarting)) {
          pushToast("🎓 the trainee can't be removed while training");
          return;
        }
        if (sel.startsWith("helper") && training?.restarting) {
          pushToast("⏳ trainer restarting — try removing the helper again in a moment");
          return;
        }
        clientRef.current?.sendRemoveDuck(sel);
        setSelectedDuck(null);
        return;
      }
      if (e.key === "Escape") {
        setSelectedDuck(null);
        return;
      }
      if (!e.shiftKey && e.key.toLowerCase() === "r") {
        e.preventDefault();
        if (e.repeat) return; // a leaned-on key must not machine-gun the lab
        clientRef.current?.sendReset();
        // The server resets silently (no `events` line back), so the only
        // confirmation the user gets is this local toast.
        pushToast("↺ sim restarted — every duck from zero");
        return;
      }
      if (cameraKeyDown(e.key, e.shiftKey)) e.preventDefault(); // no scroll/find-as-you-type
    };
    const onKeyUp = (e: KeyboardEvent) => cameraKeyUp(e.key);
    const onBlur = () => cameraKeysClear(); // no motion stuck across focus loss
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("blur", onBlur);

    // Pull keyboard focus into the page up front: embedded panes and some
    // browsers won't route key events to the document until something in it
    // has been focused.
    rootRef.current?.focus({ preventScroll: true });

    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      window.removeEventListener("blur", onBlur);
      client.close();
    };
  }, []);

  // Clicking the stage re-grabs focus for the wrapper; clicks on real
  // interactive elements (pad buttons, future chat input) keep their focus.
  // The pointer-down is also remembered for the click-to-select handler below:
  // position (to tell a click from an orbit drag) and whether an assign
  // gesture was live (armed chips assign on pointerdown, so by click time
  // assignDrag.mode is already cleared — snapshot it here instead).
  const downAt = useRef<{ x: number; y: number; assigning: boolean }>({
    x: 0,
    y: 0,
    assigning: false,
  });
  const refocus = (e: React.PointerEvent<HTMLDivElement>) => {
    downAt.current = { x: e.clientX, y: e.clientY, assigning: assignDrag.mode !== null };
    const t = e.target as HTMLElement | null;
    if (t?.closest("button, input, textarea, select, a, [contenteditable]")) return;
    rootRef.current?.focus({ preventScroll: true });
  };

  // The stage's own scale, and the two things sized by it. Both are passed to
  // OrbitControls AND to CameraKeys, which is why they are one value here: the
  // controls re-clamp to their own `maxDistance` inside `update()`, so a
  // ceiling the keys believe in and the controls do not is a dolly that stops
  // dead with nothing on screen to explain it (lib/camera.effectiveClamp).
  const subjectM = stageSize(stageRobots, scenes);
  const camMaxDist = stageMaxDist(subjectM);
  const home = useMemo(
    () => homeView(DUCK_HOME, subjectM, DUCK_SUBJECT_M, camMaxDist),
    [subjectM, camMaxDist]
  );

  // First visit with nothing saved: the Canvas mounted at the DUCK's home
  // because no frame had arrived yet to say what is on the stage. Fly to the
  // right one as soon as that is known — once, and never over a saved camera
  // or a view the person has since moved.
  const homedOnce = useRef(false);
  useEffect(() => {
    if (savedCam || homedOnce.current || !stageRobots.length) return;
    homedOnce.current = true;
    requestViewReset();
  }, [savedCam, stageRobots]);

  // Click a duck to select it (nearest projected duck within the same screen
  // radius the assign flow uses); click empty floor to deselect. Orbit drags
  // and armed-assign clicks don't count.
  const selectAt = (e: React.MouseEvent<HTMLDivElement>) => {
    const t = e.target as HTMLElement | null;
    if (t?.tagName !== "CANVAS") return; // panel/button clicks aren't stage clicks
    if (downAt.current.assigning) return;
    if (Math.hypot(e.clientX - downAt.current.x, e.clientY - downAt.current.y) > 5) return;
    setSelectedDuck(nearestDuck(e.clientX, e.clientY));
  };

  return (
    <div
      ref={rootRef}
      tabIndex={0}
      onPointerDown={refocus}
      onClick={selectAt}
      style={{ position: "fixed", inset: 0, background: "#101216", outline: "none" }}
    >
      <Canvas
        dpr={[1, 1.5]}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        camera={{
          position: savedCam?.p ?? DUCK_HOME.p,
          fov: 40,
          near: 0.01,
          far: 50,
        }}
      >
        <color attach="background" args={["#101216"]} />
        {/* Fog is the floor-edge fade, and it travels with the view: a 1.2 m
            body framed from 5 m used to sit inside a band tuned for a 25 cm
            duck at 1.9 m (components/StageFog.tsx, lib/camera.fogRange). */}
        <StageFog color="#101216" near={4} far={10} farMax={18} />
        <hemisphereLight intensity={0.65} groundColor="#2a2c33" color="#dfe6f0" />
        <directionalLight position={[2.5, 4, 2]} intensity={1.9} />
        <directionalLight position={[-2, 2.5, -1.5]} intensity={0.4} color="#8fa3c7" />

        {/* floor + grid live in three's Y-up world */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.002, 0]}>
          <circleGeometry args={[9, 64]} />
          <meshStandardMaterial color="#181b21" roughness={0.95} />
        </mesh>
        <gridHelper args={[18, 72, "#3a4150", "#262a33"]} position={[0, -0.001, 0]} />

        {/* MuJoCo Z-up world */}
        <group rotation={[-Math.PI / 2, 0, 0]}>
          {scene && clientRef.current && (
            <Ducks scene={scene} scenes={scenes} client={clientRef.current} />
          )}
          {/* 🎬 editor's ghost duck — server-side FK only, no env, no stream */}
          {scene && <PoseDuck scene={scene} />}
        </group>
        {clientRef.current && <AssignTargets client={clientRef.current} />}
        {clientRef.current && (
          <RecordCamera
            client={clientRef.current}
            scenes={scenes}
            maxRadius={camMaxDist}
          />
        )}

        <OrbitControls
          makeDefault
          target={savedCam?.t ?? DUCK_HOME.t}
          maxPolarAngle={Math.PI / 2 - 0.02}
          minDistance={CAM_MIN_DIST}
          maxDistance={camMaxDist}
          // Trackpad two-finger vertical (and wheel) zoom at the default 1.0
          // crossed half the zoom range in one small swipe — tame it.
          zoomSpeed={0.4}
        />
        <CameraPersistence />
        <CameraKeys
          home={home}
          minDist={CAM_MIN_DIST}
          maxDist={camMaxDist}
          paused={() => {
            const ph = getCapture().phase;
            return ph === "framing" || ph === "recording";
          }}
        />
        <CaptureCanvas />
        <Snapshotter />
      </Canvas>
      <Hud clientRef={clientRef} connected={connected} error={error} />
      <RecordPanel clientRef={clientRef} />
      <PolicyPanel clientRef={clientRef} />
      <TeachPanel clientRef={clientRef} />
      <AnimPanel />
      <Toasts clientRef={clientRef} />
    </div>
  );
}
