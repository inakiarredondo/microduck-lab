import { describe, expect, it } from "vitest";
import { dollyStep, effectiveClamp } from "./camera";

const MIN = 0.3, MAX = 12;

describe("dollyStep", () => {
  it("passes an unclamped dolly straight through, with no fly", () => {
    expect(dollyStep(2, MIN, MAX)).toEqual({ radius: 2, fly: 0 });
    expect(dollyStep(MIN, MIN, MAX)).toEqual({ radius: MIN, fly: 0 });
    expect(dollyStep(MAX, MIN, MAX)).toEqual({ radius: MAX, fly: 0 });
  });

  it("turns a REFUSED near dolly into a forward fly", () => {
    // this is the whole bug: without it the camera just stops dead
    const s = dollyStep(MIN - 0.05, MIN, MAX);
    expect(s.radius).toBe(MIN);
    expect(s.fly).toBeCloseTo(0.05, 9);
  });

  it("hands back exactly the distance the clamp ate, so nothing is lost", () => {
    for (const wanted of [0.29, 0.2, 0.05, 0.001]) {
      const s = dollyStep(wanted, MIN, MAX);
      expect(s.radius - s.fly).toBeCloseTo(wanted, 9);
    }
  });

  it("does NOT fly backwards at the far clamp", () => {
    const s = dollyStep(MAX + 5, MIN, MAX);
    expect(s.radius).toBe(MAX);
    expect(s.fly).toBe(0);
  });

  it("never returns a radius outside the clamps", () => {
    for (const wanted of [-1, 0, 0.001, 0.3, 5, 12, 50, 1e6]) {
      const { radius } = dollyStep(wanted, MIN, MAX);
      expect(radius).toBeGreaterThanOrEqual(MIN);
      expect(radius).toBeLessThanOrEqual(MAX);
    }
  });

  it("is safe on degenerate input rather than emitting NaN", () => {
    for (const bad of [NaN, Infinity, -0]) {
      const s = dollyStep(bad as number, MIN, MAX);
      expect(Number.isFinite(s.radius)).toBe(true);
      expect(Number.isFinite(s.fly)).toBe(true);
    }
  });

  it("repeated held ticks keep flying, so holding W keeps going", () => {
    // once pinned, every further tick must still yield forward motion
    let flown = 0;
    for (let i = 0; i < 10; i++) {
      const s = dollyStep(MIN * Math.exp(-1.5 * 0.016), MIN, MAX);
      flown += s.fly;
    }
    expect(flown).toBeGreaterThan(0.05);
  });
});

// The clamp CameraKeys is told about and the clamp OrbitControls enforces are
// two different numbers, and `controls.update()` — which CameraKeys calls at
// the end of every frame — gives the second one the last word. On the lab page
// they had drifted apart (props 0.25, controls 0.3) and W stopped dead: the
// radius never fell below 0.25, so `dollyStep` never had a dolly to refuse,
// never returned a fly, and the re-clamp undid the rest. Measured on the
// running lab: radius exactly 0.3000, orbit target frozen across 100 presses.
describe("effectiveClamp", () => {
  it("takes the tighter bound at each end", () => {
    expect(effectiveClamp(0.25, 8, 0.3, 7)).toEqual({ min: 0.3, max: 7 });
    expect(effectiveClamp(0.5, 6, 0.3, 7)).toEqual({ min: 0.5, max: 6 });
  });

  it("falls back to the props when the controls declare no clamp", () => {
    expect(effectiveClamp(0.3, 12, undefined, undefined)).toEqual({ min: 0.3, max: 12 });
    // OrbitControls' own defaults are 0 and Infinity — neither should widen us
    expect(effectiveClamp(0.3, 12, 0, Infinity)).toEqual({ min: 0.3, max: 12 });
  });

  it("never returns a max below its min", () => {
    const c = effectiveClamp(5, 6, 0.3, 1); // crossed bounds
    expect(c.max).toBeGreaterThanOrEqual(c.min);
  });
});

/** One frame of the W loop, as CameraKeys runs it: dolly the radius, ask
 *  `dollyStep` what that means, then let OrbitControls re-clamp. Returns the
 *  total forward distance flown — the thing the user sees as moving. */
function holdW(
  propMin: number,
  controlsMin: number,
  frames: number,
  reconcile: boolean,
): number {
  const dt = 1 / 60;
  let radius = controlsMin; // already parked against the controls' wall
  let flown = 0;
  for (let i = 0; i < frames; i++) {
    const wanted = radius * Math.exp(-1.5 * dt);
    const c = reconcile
      ? effectiveClamp(propMin, 8, controlsMin, 7)
      : { min: propMin, max: 8 }; // the old code: props only
    const step = dollyStep(wanted, c.min, c.max);
    flown += step.fly;
    radius = Math.max(controlsMin, step.radius); // controls.update() has the last word
  }
  return flown;
}

describe("holding W against the near wall", () => {
  it("goes nowhere when the prop clamp is looser than the controls' (the bug)", () => {
    expect(holdW(0.25, 0.3, 120, false)).toBe(0);
  });

  it("flies once the two clamps are reconciled", () => {
    expect(holdW(0.25, 0.3, 120, true)).toBeGreaterThan(0.5);
  });

  it("flies on a page whose clamps already agreed, either way (/sim)", () => {
    expect(holdW(0.3, 0.3, 120, false)).toBeGreaterThan(0.5);
    expect(holdW(0.3, 0.3, 120, true)).toBeGreaterThan(0.5);
  });
});
