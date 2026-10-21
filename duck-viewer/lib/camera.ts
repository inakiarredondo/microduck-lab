// Keyboard → camera motion (game-editor style). Keys set/clear entries in a
// held-set on keydown/keyup; the CameraKeys helper inside the Canvas applies
// velocity × dt every frame while a key is held — smooth continuous motion,
// not per-keypress steps (the first version stepped like a scroll wheel;
// Maya/UE-trained hands expect flow). The view reset (Shift+R) is the one
// discrete action.

export type CameraMotion =
  | "truckLeft"   // A — slide view left (mouse already owns orbit)
  | "truckRight"  // D
  | "dollyIn"     // W / ↑
  | "dollyOut"    // S / ↓
  | "orbitLeft"   // ←
  | "orbitRight"  // →
  | "up"          // E — vertical truck (E rises, Q sinks)
  | "down";       // Q

const KEY_MOTIONS: Record<string, CameraMotion> = {
  a: "truckLeft",
  d: "truckRight",
  w: "dollyIn",
  s: "dollyOut",
  arrowup: "dollyIn",
  arrowdown: "dollyOut",
  arrowleft: "orbitLeft",
  arrowright: "orbitRight",
  e: "up",
  q: "down",
};

const held = new Set<CameraMotion>();
let resetPending = false;

/** keydown: begin the motion (or queue the view reset). True if it was a
 *  camera key. Plain R now restarts the SIM (Viewer owns that), so the view
 *  reset moved to Shift+R — passed as a flag rather than sniffed from the key
 *  case, which Caps Lock would lie about. */
export function cameraKeyDown(key: string, shift = false): boolean {
  const k = key.toLowerCase();
  if (k === "r") {
    if (!shift) return false; // plain R isn't ours — leave it to the caller
    requestViewReset();
    return true;
  }
  const m = KEY_MOTIONS[k];
  if (!m) return false;
  held.add(m);
  return true;
}

/** keyup: end the motion. Safe to call for any key. */
export function cameraKeyUp(key: string): void {
  const m = KEY_MOTIONS[key.toLowerCase()];
  if (m) held.delete(m);
}

/** Stuck-key guard: clear everything (window blur, page hide). */
export function cameraKeysClear(): void {
  held.clear();
}

export function heldMotions(): ReadonlySet<CameraMotion> {
  return held;
}

/** Queue the view reset without a keystroke — what a page calls when it has
 *  nothing saved to restore and has just learned how big the stage is. The
 *  same one-shot flag Shift+R sets, so `CameraKeys` performs it in the frame
 *  loop where the camera and controls live. */
export function requestViewReset(): void {
  resetPending = true;
}

export function takeReset(): boolean {
  const r = resetPending;
  resetPending = false;
  return r;
}

// --- trackpad swipe impulses -------------------------------------------------
// Two-finger horizontal swipes arrive as a smooth high-rate wheel-event
// stream, not key holds — so they accumulate PIXEL IMPULSES here rather than
// entries in the held-set (held state would stick between events). CameraKeys
// drains the total once per frame and applies it as a lateral truck.

let truckImpulsePx = 0;

/** Accumulate a horizontal swipe impulse (wheel deltaX, in pixels). */
export function truckImpulse(dxPixels: number): void {
  if (Number.isFinite(dxPixels)) truckImpulsePx += dxPixels;
}

/** Per-frame drain: return the accumulated impulse and clear it. */
export function takeTruckImpulse(): number {
  const px = truckImpulsePx;
  truckImpulsePx = 0;
  return px;
}

/** What a dolly tick should actually do, once the near/far clamps are applied.
 *
 *  W used to mean "shrink the orbit radius", and the orbit target is a FIXED
 *  point — the middle of the room on /sim. So flying at something sitting a
 *  metre off-centre was impossible: the radius hit `minDist` while the thing
 *  you were aiming at was still off to one side, and the camera simply stopped
 *  with nothing on screen to explain why. It reads as collision. It is a clamp.
 *
 *  So a dolly that the near clamp REFUSES becomes a fly: the distance the
 *  clamp ate is handed back as `fly`, and the caller moves the camera AND the
 *  orbit target forward by it. The rig keeps its shape (orbit still orbits
 *  around whatever you flew to), and you can reach anything in the room.
 *
 *  Only the near end does this. Pinning at `maxDist` and pulling further back
 *  would drag the orbit target away across the floor, which is a different
 *  gesture (that is what the trucks are for) and not one anybody asked for.
 */
export interface DollyStep {
  /** the orbit radius to apply, always within [minDist, maxDist] */
  radius: number;
  /** metres to move camera AND target along the view direction, forward
   *  positive. 0 whenever the dolly alone could do the job. */
  fly: number;
}

export function dollyStep(wanted: number, minDist: number, maxDist: number): DollyStep {
  if (!(wanted > 0) || !(minDist >= 0) || !(maxDist >= minDist)) {
    return { radius: Math.max(minDist, 0), fly: 0 };
  }
  if (wanted < minDist) return { radius: minDist, fly: minDist - wanted };
  if (wanted > maxDist) return { radius: maxDist, fly: 0 };
  return { radius: wanted, fly: 0 };
}

/** The near/far clamp the dolly must actually respect.
 *
 *  Two clamps govern the same radius and only one of them gets the last word.
 *  CameraKeys is handed `minDist`/`maxDist` as props, but it finishes every
 *  frame with `controls.update()`, and OrbitControls re-clamps the camera to
 *  ITS OWN `minDistance`/`maxDistance` right there. So a dolly the props allow
 *  and the controls refuse is simply undone, silently, every frame.
 *
 *  That is how W died on the lab page: props said 0.25, OrbitControls said
 *  0.3. W wound the radius down to 0.2929 (never below 0.25), `dollyStep` saw
 *  nothing to refuse and returned fly 0, then `controls.update()` pushed the
 *  camera straight back out to 0.3. Measured on the running lab: radius pinned
 *  at exactly 0.3000 and the orbit target frozen across 100 presses of W. The
 *  fly that exists precisely to stop this was unreachable — the one clamp that
 *  could have triggered it was the one nobody was checking.
 *
 *  So reconcile them before deciding: take the TIGHTER bound at each end, the
 *  one that will actually be enforced. The near bound then bites inside the
 *  range the controls permit, the fly fires, and the two can no longer
 *  disagree about where the wall is — whatever a future call site passes.
 */
export function effectiveClamp(
  minDist: number,
  maxDist: number,
  controlsMin?: number,
  controlsMax?: number,
): { min: number; max: number } {
  const min = Math.max(minDist, Number.isFinite(controlsMin) ? (controlsMin as number) : 0);
  const max = Math.min(maxDist, Number.isFinite(controlsMax) ? (controlsMax as number) : Infinity);
  // A call site whose bounds cross would otherwise hand `dollyStep` a max
  // below its min; collapse to the near bound rather than emit a dead range.
  return { min, max: Math.max(min, max) };
}

// --- fog that tracks the view -------------------------------------------------

/** How far out the fog starts, as a multiple of the orbit radius.
 *
 *  Fog here is a floor-edge fade, not an effect on the robots: the stage is a
 *  9 m disc of near-black floor on a near-black background, and without a
 *  fade its rim and the grid's square end are hard lines across the picture.
 *  Fixed metres do that job for exactly one subject size. The lab's 4 → 10 m
 *  band was tuned on a 25 cm duck, which the camera frames from ~1.9 m; a
 *  1.2 m Asimov on its own 1.6 m pitch needs the camera ~5 m back, and from
 *  there the BODIES sit inside the band — measured on a 4-slot roster, the
 *  far Asimov came out 35 % blended into the background colour and the grid
 *  had vanished entirely.
 *
 *  So the band travels with the view instead. 1.7 is the multiple that leaves
 *  both pages' home cameras where they were tuned: the lab's home radius is
 *  1.93 m (1.7 x = 3.3 m, inside its 4 m floor, so nothing moves) and /sim's
 *  is 3.58 m (1.7 x = 6.09 m against a tuned 6 m — a 1.5 % nudge). Pull back
 *  past home and the fog walks out ahead of the camera, so whatever is being
 *  framed stays out of it.
 */
export const FOG_NEAR_MULT = 1.7;

export interface FogRange {
  near: number;
  far: number;
}

/** Where the fog band belongs for an orbit radius of `dist`.
 *
 *  `near0`/`far0` are the page's tuned band — its near is a FLOOR (a close-up
 *  must not fog the floor under the robot's feet) and the width `far0 - near0`
 *  is preserved, so the fade keeps its depth however far out the view goes.
 *  `farMax` caps the far plane: push the band past the floor's own rim and
 *  nothing fades at all, which is the hard edge this is here to hide.
 */
export function fogRange(dist: number, near0: number, far0: number, farMax = Infinity): FogRange {
  const band = Math.max(0, far0 - near0);
  const ceiling = Math.max(near0, farMax - band);
  const wanted = Number.isFinite(dist) && dist > 0 ? FOG_NEAR_MULT * dist : near0;
  const near = Math.min(Math.max(near0, wanted), ceiling);
  return { near, far: near + band };
}

// --- the 🎥 record shot, sized to the subject ---------------------------------

/** The ¾ front shot the 🎥 take glides to, as the duck's own numbers:
 *  `az` radians off the subject's heading, `dist` metres out on the floor,
 *  `height` metres above the trunk it aims at, and a very slow azimuth
 *  `drift` so long takes stay alive.
 *
 *  These were tuned so that a 25 cm duck fills about half the frame at the
 *  stage's 40° fov, and for three years that was every subject the lab had.
 *  `recordShot` is what makes them a shot of ANY body. */
export const DUCK_SHOT = { az: 0.61, dist: 0.78, height: 0.34, drift: 0.05 };

/** How big the duck they were tuned on is: 0.277 m tall, 0.185 m across
 *  (`lab/robots.measured_size("microduck")` — geom AABBs at STAND, the same
 *  measurement `lab_spacing_m`'s 0.65 m is derived from).
 *
 *  Spelled to the digit the lab sends rather than rounded, so a duck stage
 *  scales by EXACTLY 1 and every framing tuned on the duck comes out
 *  bit-identical. Rounded to 0.277 it was out by 0.065 %, which no eye can
 *  see and every pinned test can. */
export const DUCK_SUBJECT_M = 0.27718;

export interface Shot {
  az: number;
  dist: number;
  height: number;
  drift: number;
}

/** How big the thing being filmed is, metres: the larger of its standing
 *  height and its widest horizontal extent (`Scene.size`).
 *
 *  Height alone is the tempting answer and it crops the wide, low bodies —
 *  MOSS is 0.406 m tall and 0.569 m across, the Go2 0.394 by 0.754 — because
 *  what has to fit in the frame is apparent size, not stature. A scene from
 *  a lab that predates the field (or a body whose model would not compile)
 *  answers nothing, and the duck's size is then the fallback: the shot comes
 *  out exactly as it always did. */
export function subjectSizeM(
  size?: { height?: number; width?: number } | null,
): number {
  const big = Math.max(size?.height ?? 0, size?.width ?? 0);
  return Number.isFinite(big) && big > 0 ? big : DUCK_SUBJECT_M;
}

/** The ¾ shot for a subject `subjectM` metres big, within `maxRadius`.
 *
 *  Everything but the ANGLES scales with the subject, so the picture is the
 *  same picture — the duck's shot sits 2.82 subject-lengths out and 1.23 up,
 *  a 23.6° lens, and so does the Asimov's at 3.65 m instead of 0.78 m. Filmed
 *  at the duck's distance, a 1.3 m robot has the camera inside its chest.
 *
 *  `maxRadius` is OrbitControls' own `maxDistance`, and passing it is not
 *  optional politeness: `controls.update()` re-clamps the camera to it every
 *  frame whether or not the controls are enabled, so a shot further out than
 *  that is silently dragged in and the take is framed at a distance nobody
 *  chose (`effectiveClamp` above is the same bug from the dolly's end). A
 *  shot that does not fit is scaled down whole, keeping its elevation. */
export function recordShot(subjectM?: number | null, maxRadius = Infinity): Shot {
  const size = Number.isFinite(subjectM ?? NaN) && (subjectM as number) > 0
    ? (subjectM as number)
    : DUCK_SUBJECT_M;
  let k = size / DUCK_SUBJECT_M;
  const radius = Math.hypot(DUCK_SHOT.dist, DUCK_SHOT.height) * k;
  if (Number.isFinite(maxRadius) && maxRadius > 0 && radius > maxRadius) {
    k *= maxRadius / radius;
  }
  return {
    az: DUCK_SHOT.az,
    dist: DUCK_SHOT.dist * k,
    height: DUCK_SHOT.height * k,
    drift: DUCK_SHOT.drift,
  };
}

// --- the home view (Shift+R), sized to what is on the stage ------------------

export interface HomePose {
  /** camera position, three world */
  p: [number, number, number];
  /** orbit target */
  t: [number, number, number];
}

/** The lab stage's home view as the duck's own numbers: 1.93 m of orbit
 *  radius, aimed 0.12 m off the floor at the middle of the grid. That is
 *  about 7 duck-lengths out — a view of the STAGE and its roster, not a
 *  portrait, which is why it is a wider multiple than `DUCK_SHOT`'s 2.8. */
export const DUCK_HOME: HomePose = { p: [1.2, 0.7, 1.4], t: [0, 0.12, 0] };

/** `/sim`'s home view, and the room it was tuned on: 3.58 m of orbit radius
 *  over the 3.5 x 3.0 m living room / playroom. A room is the subject there,
 *  so this one scales on the floor, not on a robot — the rooms this lab ships
 *  run from MOSS's 2.6 m yard to the Asimov's 8 m hall. */
export const SIM_HOME: HomePose = { p: [2.2, 1.6, 2.4], t: [0, 0.12, 0] };
export const SIM_HOME_ROOM_M = 3.5;

/** `home`, scaled for a subject `subjectM` big where it was tuned on one
 *  `referenceM` big, within `maxRadius` of its target.
 *
 *  The whole rig scales — the offset from target to camera, and the target's
 *  own height — so the view keeps its angle and its proportions and only its
 *  SIZE changes. A stage holding a 1.296 m Asimov is 4.7x the duck it was
 *  drawn for, and at the duck's 1.93 m the reset lands with the robot filling
 *  the frame rather than standing in a stage.
 *
 *  `maxRadius` is OrbitControls' `maxDistance` for the same reason
 *  `recordShot` takes it: `controls.update()` re-clamps to it every frame, so
 *  a home further out than that is not a home, it is a shot that gets dragged
 *  in the instant it arrives. */
export function homeView(
  home: HomePose,
  subjectM?: number | null,
  referenceM = DUCK_SUBJECT_M,
  maxRadius = Infinity,
): HomePose {
  const size = Number.isFinite(subjectM ?? NaN) && (subjectM as number) > 0
    ? (subjectM as number)
    : referenceM;
  let k = size / referenceM;
  const off = [
    home.p[0] - home.t[0],
    home.p[1] - home.t[1],
    home.p[2] - home.t[2],
  ] as const;
  const radius = Math.hypot(off[0], off[1], off[2]) * k;
  if (Number.isFinite(maxRadius) && maxRadius > 0 && radius > maxRadius) {
    k *= maxRadius / radius;
  }
  const t: [number, number, number] = [
    home.t[0] * k,
    home.t[1] * k,
    home.t[2] * k,
  ];
  return {
    p: [t[0] + off[0] * k, t[1] + off[1] * k, t[2] + off[2] * k],
    t,
  };
}

/** How far the lab stage's dolly may pull back with a subject this big.
 *
 *  7 m was the duck's ceiling and it is 25 duck-lengths — on a 1.296 m Asimov
 *  the same number is 5.4 body-lengths, which is INSIDE the home view that
 *  body wants (9.04 m). A ceiling that cuts the home view is the clamp
 *  choosing the framing, which is the bug `effectiveClamp` documents from the
 *  dolly's end.
 *
 *  So it is the same multiple of whatever is on the stage, floored at the
 *  duck's 7 m (a small body must not shrink the room you can move in) and
 *  capped at `ceiling`: the stage floor is a 9 m disc, so a camera further
 *  out than 18 m is framing background. */
export function stageMaxDist(
  subjectM?: number | null,
  base = 7,
  referenceM = DUCK_SUBJECT_M,
  ceiling = 18,
): number {
  const size = Number.isFinite(subjectM ?? NaN) && (subjectM as number) > 0
    ? (subjectM as number)
    : referenceM;
  return Math.min(ceiling, Math.max(base, (base * size) / referenceM));
}
