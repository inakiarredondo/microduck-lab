// The arrow's arithmetic. The orientation is the part worth pinning: a yaw
// off by 90 degrees draws an arrow across the push instead of along it,
// which looks entirely plausible on screen.

import { describe, expect, it } from "vitest";
import { PUSH_HOLD_MS, PUSH_M_PER_N, PUSH_MAX_M, PUSH_MIN_M, pushArrow, pushArrowHeld } from "./push";

/** Where a mesh whose local axis is +Y points after a Z rotation of `yaw`. */
function heads(yaw: number): [number, number] {
  return [-Math.sin(yaw), Math.cos(yaw)];
}

describe("pushArrow", () => {
  it("draws nothing without a push", () => {
    expect(pushArrow(null)).toBeNull();
    expect(pushArrow(undefined)).toBeNull();
    // sub-newton dust is not a shove
    expect(pushArrow({ at: [0, 0, 0], vec: [0.4, 0, 0], n: 0.4 })).toBeNull();
  });

  it("points ALONG the force, not across it", () => {
    for (const [vec, want] of [
      [[1, 0], [1, 0]],
      [[0, 1], [0, 1]],
      [[-1, 0], [-1, 0]],
      [[0, -1], [0, -1]],
    ] as const) {
      const a = pushArrow({ at: [0, 0, 0], vec: [vec[0] * 506, vec[1] * 506, 0], n: 506 });
      const [hx, hy] = heads(a!.yaw);
      expect(hx).toBeCloseTo(want[0], 6);
      expect(hy).toBeCloseTo(want[1], 6);
    }
  });

  it("anchors at the trunk and reports the newtons for the label", () => {
    const a = pushArrow({ at: [1.5, -2, 0.62], vec: [0, 506, 0], n: 506 });
    expect(a!.at).toEqual([1.5, -2, 0.62]);
    expect(a!.newtons).toBe(506);
  });

  const len = (n: number) => pushArrow({ at: [0, 0, 0], vec: [n, 0, 0], n })!.length;

  it("draws length PROPORTIONAL to force, at a stated scale", () => {
    expect(len(400)).toBeCloseTo(400 * PUSH_M_PER_N, 6);
    expect(len(700)).toBeCloseTo(700 * PUSH_M_PER_N, 6);
    // twice the force is twice the arrow, which is the whole point
    expect(len(700) / len(350)).toBeCloseTo(2, 6);
  });

  it("separates the ENTIRE range the env actually produces", () => {
    // `_apply_push` draws uniform(0.5, 1.5) x push_bw x mass x g, so at 1.6
    // bodyweights on the 32 kg Asimov that is 251-753 N. The first version
    // pinned full length at the NOMINAL 506 N, so everything above it — half
    // the distribution — drew identically.
    const lo = len(251);
    const hi = len(753);
    expect(hi).toBeGreaterThan(lo * 2.5);
    expect(hi).toBeLessThanOrEqual(PUSH_MAX_M);
    for (const n of [251, 400, 506, 620, 753]) {
      expect(len(n)).toBeGreaterThan(PUSH_MIN_M);
      expect(len(n)).toBeLessThan(PUSH_MAX_M); // none of them clamped
    }
  });

  it("clamps at both ends so it can never vanish or become scenery", () => {
    expect(len(5)).toBe(PUSH_MIN_M);
    expect(len(50_000)).toBe(PUSH_MAX_M);
  });

  it("ignores the vertical channel, which is the demo ASSIST", () => {
    // the server already drops it; this is the second line of defence, since
    // an assist drawn as a shove would say the robot was pushed when it was
    // being HELD UP.
    expect(pushArrow({ at: [0, 0, 0], vec: [0, 0, 900], n: 900 })).toBeNull();
  });
});

describe("pushArrowHeld", () => {
  const shove = { at: [0, 0, 0.6], vec: [506, 0, 0], n: 506 };

  it("holds the arrow after a 7-frame impulse, then fades it out", () => {
    const last = { arrow: null as ReturnType<typeof pushArrow>, at: 0 };
    // while the force is applied: full strength
    const a = pushArrowHeld(shove, last, 1000);
    expect(a.arrow).not.toBeNull();
    expect(a.opacity).toBe(1);
    // the impulse ends — still drawn, fading
    const b = pushArrowHeld(null, last, 1000 + PUSH_HOLD_MS / 2);
    expect(b.arrow).not.toBeNull();
    expect(b.opacity).toBeCloseTo(0.5, 2);
    // ...and gone once it has aged out
    const c = pushArrowHeld(null, last, 1000 + PUSH_HOLD_MS + 1);
    expect(c.arrow).toBeNull();
    expect(c.opacity).toBe(0);
    expect(last.arrow).toBeNull(); // the memory is released, not leaked
  });

  it("a fresh shove replaces a fading one rather than queueing", () => {
    const last = { arrow: null as ReturnType<typeof pushArrow>, at: 0 };
    pushArrowHeld(shove, last, 0);
    pushArrowHeld(null, last, 300);
    const again = pushArrowHeld({ at: [0, 0, 0.6], vec: [0, -506, 0], n: 506 }, last, 400);
    expect(again.opacity).toBe(1);
    // and it points the NEW way, not the old one
    const [hx, hy] = [-Math.sin(again.arrow!.yaw), Math.cos(again.arrow!.yaw)];
    expect(hx).toBeCloseTo(0, 6);
    expect(hy).toBeCloseTo(-1, 6);
  });

  it("draws nothing when nothing has ever pushed", () => {
    const last = { arrow: null as ReturnType<typeof pushArrow>, at: 0 };
    expect(pushArrowHeld(null, last, 5000).arrow).toBeNull();
  });
});
