// The stage fog's near/far, pinned. The case that put this here: a 1.2 m
// Asimov framed from ~5 m sat INSIDE a fog band tuned for a 25 cm duck at
// ~1.9 m, so the far body of a 4-slot roster rendered 35 % blended into the
// background and the grid was gone. The band now travels with the view.

import { describe, expect, it } from "vitest";
import { fogRange } from "@/lib/camera";

// Each page's tuned band, and the radius its home camera sits at.
const LAB = { near: 4, far: 10, farMax: 18, home: 1.93 };
const SIM = { near: 6, far: 14, farMax: 26, home: 3.58 };

describe("fogRange", () => {
  it("leaves the lab's home view exactly as it was tuned", () => {
    expect(fogRange(LAB.home, LAB.near, LAB.far, LAB.farMax)).toEqual({ near: 4, far: 10 });
  });

  it("leaves /sim's home view within a nudge of its tuning", () => {
    const r = fogRange(SIM.home, SIM.near, SIM.far, SIM.farMax);
    expect(r.near).toBeCloseTo(6.086, 3);
    expect(r.far - r.near).toBe(8); // the band's width never changes
  });

  it("walks the band out ahead of a pulled-back camera", () => {
    // The Asimov view: camera 5.1 m out, the far body 6.09 m from it. Under
    // the fixed 4 → 10 band that body was 35 % fogged; it must now be clear.
    const r = fogRange(5.1, LAB.near, LAB.far, LAB.farMax);
    expect(r.near).toBeGreaterThan(6.09);
    expect(r.far).toBe(r.near + 6);
  });

  it("still fades the floor's rim at the far end", () => {
    // Lab floor is a 9 m disc; from 5.1 m out its far rim is ~13.8 m away and
    // has to be most of the way gone, or the disc shows a hard edge.
    const r = fogRange(5.1, LAB.near, LAB.far, LAB.farMax);
    expect((13.8 - r.near) / (r.far - r.near)).toBeGreaterThan(0.5);
  });

  it("never lets the band leave the floor behind (farMax)", () => {
    // Orbit radius is clamped per page, but the cap is what guarantees the
    // fade exists at full zoom-out rather than relying on that clamp.
    const r = fogRange(100, LAB.near, LAB.far, LAB.farMax);
    expect(r).toEqual({ near: 12, far: 18 });
  });

  it("keeps the near floor on a close-up, so feet never fog", () => {
    expect(fogRange(0.3, LAB.near, LAB.far, LAB.farMax)).toEqual({ near: 4, far: 10 });
  });

  it("falls back to the tuned band for a distance that is not a number", () => {
    expect(fogRange(NaN, LAB.near, LAB.far, LAB.farMax)).toEqual({ near: 4, far: 10 });
    expect(fogRange(0, LAB.near, LAB.far, LAB.farMax)).toEqual({ near: 4, far: 10 });
  });

  it("is monotone in distance", () => {
    let prev = 0;
    for (let d = 0.5; d <= 12; d += 0.5) {
      const { near } = fogRange(d, LAB.near, LAB.far, LAB.farMax);
      expect(near).toBeGreaterThanOrEqual(prev);
      prev = near;
    }
  });
});
