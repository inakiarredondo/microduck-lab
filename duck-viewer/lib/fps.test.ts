import { beforeEach, describe, expect, it } from "vitest";
import { GAP_MS, readFps, resetFps, SMOOTHING, tickFrame } from "./fps";

const settle = (ms: number, n = 300, t0 = 1000) => {
  let t = t0;
  for (let i = 0; i < n; i++) { t += ms; tickFrame(t); }
  return readFps();
};

describe("fps meter", () => {
  beforeEach(resetFps);

  it("reads 0 before any frame, and after one frame (no interval yet)", () => {
    expect(readFps()).toBe(0);
    tickFrame(1000);
    expect(readFps()).toBe(0);
  });

  it("converges on the true rate", () => {
    expect(settle(1000 / 60)).toBeCloseTo(60, 1);
    resetFps();
    expect(settle(1000 / 24)).toBeCloseTo(24, 1);
  });

  it("ignores GAPS instead of averaging them in", () => {
    // 60 fps, then the tab is hidden for 10 s, then 60 fps again. A gap that
    // folded into the average would drag the readout far below 60.
    settle(1000 / 60);
    const before = readFps();
    let t = 1000 + 300 * (1000 / 60);
    t += 10_000;
    tickFrame(t);                       // the gap frame
    expect(readFps()).toBeCloseTo(before, 5);
    for (let i = 0; i < 5; i++) { t += 1000 / 60; tickFrame(t); }
    expect(readFps()).toBeCloseTo(60, 0);
  });

  it("treats a frame just inside the gap bound as a slow frame, not a gap", () => {
    settle(1000 / 60);
    const fast = readFps();
    let t = 1000 + 300 * (1000 / 60);
    t += GAP_MS - 1;
    tickFrame(t);
    expect(readFps()).toBeLessThan(fast); // it counted, so the rate fell
  });

  it("ignores a repeated or non-monotonic timestamp", () => {
    settle(1000 / 60);
    const before = readFps();
    const t = 1000 + 300 * (1000 / 60);
    tickFrame(t);      // same instant as the last frame
    tickFrame(t - 50); // and one going backwards
    expect(readFps()).toBeCloseTo(before, 5);
  });

  it("smooths rather than snapping to the latest frame", () => {
    settle(1000 / 60);
    let t = 1000 + 300 * (1000 / 60);
    t += 1000 / 10;                       // one 10 fps frame
    tickFrame(t);
    // one bad frame must not drag the readout all the way to 10
    expect(readFps()).toBeGreaterThan(40);
    expect(readFps()).toBeLessThan(60);
    expect(SMOOTHING).toBeGreaterThan(0);
  });

  it("resets, so a remounted canvas does not inherit the old rate", () => {
    settle(1000 / 60);
    expect(readFps()).toBeGreaterThan(0);
    resetFps();
    expect(readFps()).toBe(0);
  });
});
