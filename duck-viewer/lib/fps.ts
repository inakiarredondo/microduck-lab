/** How fast the BROWSER is drawing, as opposed to how fast the lab is
 *  simulating.
 *
 *  These are different numbers and the difference is the point. The /sim HUD
 *  has always shown RTF — the lab's own sim-time-over-wall-time — and a page
 *  whose render loop is struggling still reports whatever RTF the lab
 *  achieved, so a stage that is visibly stuttering can sit next to a healthy
 *  RTF and give no clue which half is at fault. MEASURED on the four-figurine
 *  showcase room (2026-10-02): the page read RTF 0.34 while the lab's world
 *  loop, timed over its own socket, was running 0.92 and the physics had a
 *  30x ceiling. The whole gap was draw calls, and nothing on screen said so.
 *
 *  A module-level store rather than React state, read once per HUD tick
 *  (4 Hz) instead of driving a re-render per frame — the viewer's rule for
 *  anything sampled inside r3f's loop.
 */

/** Frames farther apart than this are not slow frames, they are a GAP — a
 *  backgrounded tab, a tab-switch, a long GC pause, the first frame after
 *  the canvas mounts. Folding one into the average pins the readout near
 *  zero long after drawing recovered, so they end the current interval
 *  instead of extending it. 1 s is well past even a very bad frame (3 fps). */
export const GAP_MS = 1000;

/** Smoothing on the frame INTERVAL. 0.1 settles in roughly 20 frames — fast
 *  enough to show a change you just made, slow enough that the number is
 *  readable rather than flickering. */
export const SMOOTHING = 0.1;

let prev = 0;
let interval = 0;

/** Call once per rendered frame, with a monotonic timestamp in ms. */
export function tickFrame(nowMs: number): void {
  const dt = prev ? nowMs - prev : 0;
  prev = nowMs;
  // dt <= 0 is a duplicated or non-monotonic timestamp: carries no rate.
  if (dt <= 0 || dt > GAP_MS) return;
  interval = interval ? interval + (dt - interval) * SMOOTHING : dt;
}

/** Frames per second, or 0 before a rate is known. */
export function readFps(): number {
  return interval > 0 ? 1000 / interval : 0;
}

/** Forget the measurement — a new canvas is not a continuation of the old. */
export function resetFps(): void {
  prev = 0;
  interval = 0;
}
