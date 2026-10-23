// THE SHOVE, as an arrow. The lab streams the external force on a robot's
// trunk (`duck.push`) and this turns it into the three numbers the mesh
// needs, so the arithmetic is testable without a canvas.
//
// Why it exists: a body in the lab can be shoved at 506 N to show it
// recovering, and without a drawn force a viewer sees a robot that falls
// over for no reason. The "spotting" chip already covers the demo ASSIST
// (the vertical channel); this is the horizontal shove only.

/** `{at, vec, n}` from the lab: trunk position, force, and its magnitude. */
export type Push = { at: number[]; vec: number[]; n: number } | null | undefined;

/** Metres of arrow per newton of force — a STATED scale, not a fitted one.
 *
 *  1 metre per kilonewton. The shove is drawn at its real relative size: a
 *  251 N nudge is a third the length of a 753 N hit, because that is what
 *  they are.
 *
 *  The first version pinned full length at 506 N, the NOMINAL shove — and
 *  the env draws `uniform(0.5, 1.5) x push_bw x mass x g`, so at 1.6
 *  bodyweights on this 32 kg robot the real range is 251-753 N and
 *  everything above 506 clamped to the same length. Half the distribution,
 *  and the harder half, all drew identically: the one thing the arrow exists
 *  to show was the thing it hid.
 */
export const PUSH_M_PER_N = 1 / 1000;
/** Short enough to read as a nudge, long enough to see at all. */
export const PUSH_MIN_M = 0.12;
/** Longer than the robot is tall stops being a vector and becomes scenery. */
export const PUSH_MAX_M = 0.95;

export type PushArrow = {
  /** Where the force is applied, in the scene's MuJoCo-style Z-up frame. */
  at: [number, number, number];
  /** Rotation about Z for a mesh whose local axis is +Y (cylinder, cone).
   *
   *  A rotation about Z by φ maps local +Y to (-sin φ, cos φ, 0), so pointing
   *  it along a heading θ needs φ = θ - π/2. Getting this wrong draws an
   *  arrow at right angles to the push, which looks plausible and is wrong.
   */
  yaw: number;
  /** Shaft+head length: `PUSH_M_PER_N` per newton, clamped at both ends. */
  length: number;
  /** For the label, so the scale is stated rather than guessed from length. */
  newtons: number;
};

/** How long the arrow stays up after the impulse ends, in ms.
 *
 *  The shove itself is 0.15 s — about SEVEN frames at 50 Hz — which is as
 *  hard for a person to see as it was to screenshot. Holding it briefly is a
 *  viewer decision and changes no physics: the force really was applied, and
 *  the fade says the arrow is a recent event rather than a current one.
 */
export const PUSH_HOLD_MS = 650;

/** The arrow to draw NOW, given the live push and the last one seen.
 *
 *  Returns `{arrow, opacity}`: full opacity while the force is applied, then
 *  fading to nothing over `PUSH_HOLD_MS`. `last` is the caller's remembered
 *  `{arrow, at}` (a ref, never React state — this runs every frame).
 */
export function pushArrowHeld(
  push: Push,
  last: { arrow: PushArrow | null; at: number },
  now: number,
): { arrow: PushArrow | null; opacity: number } {
  const live = pushArrow(push);
  if (live) {
    last.arrow = live;
    last.at = now;
    return { arrow: live, opacity: 1 };
  }
  if (!last.arrow) return { arrow: null, opacity: 0 };
  const age = now - last.at;
  if (age >= PUSH_HOLD_MS) {
    last.arrow = null;
    return { arrow: null, opacity: 0 };
  }
  return { arrow: last.arrow, opacity: 1 - age / PUSH_HOLD_MS };
}

export function pushArrow(push: Push): PushArrow | null {
  if (!push || !Array.isArray(push.vec) || !Array.isArray(push.at)) return null;
  const [fx, fy] = [Number(push.vec[0]) || 0, Number(push.vec[1]) || 0];
  const n = Math.hypot(fx, fy);
  // Below a newton is numerical dust, not a shove — the server applies the
  // same floor, and both are cheap next to drawing a flickering arrow.
  if (!(n >= 1)) return null;
  return {
    at: [Number(push.at[0]) || 0, Number(push.at[1]) || 0, Number(push.at[2]) || 0],
    yaw: Math.atan2(fy, fx) - Math.PI / 2,
    length: Math.min(PUSH_MAX_M, Math.max(PUSH_MIN_M, n * PUSH_M_PER_N)),
    newtons: Number(push.n) || n,
  };
}
