// The 🎥 record shot's framing, pinned. The case that put this here: the take
// camera's metres were tuned on a 25 cm duck (0.78 m out, 0.34 m up) and every
// other body got the same ones, so filming a 1.296 m Asimov parked the lens
// inside its chest. The shot now scales with the subject.

import { describe, expect, it } from "vitest";
import {
  DUCK_SHOT,
  DUCK_SUBJECT_M,
  recordShot,
  subjectSizeM,
} from "@/lib/camera";

// `lab/robots.measured_size` for each body: geom AABBs at its stand keyframe.
const SIZE = {
  microduck: { height: 0.27718, width: 0.18453 },
  asimov1: { height: 1.29608, width: 0.456 },
  g1: { height: 1.30711, width: 0.53382 },
  moss: { height: 0.40579, width: 0.56881 },
  go2: { height: 0.39352, width: 0.75359 },
};

const CAM_MAX_DIST = 7; // the lab page's OrbitControls maxDistance

describe("subjectSizeM", () => {
  it("takes the larger of height and width", () => {
    expect(subjectSizeM(SIZE.asimov1)).toBeCloseTo(1.29608, 5); // tall
    expect(subjectSizeM(SIZE.go2)).toBeCloseTo(0.75359, 5); // wide and low
    expect(subjectSizeM(SIZE.moss)).toBeCloseTo(0.56881, 5);
  });

  it("falls back to the duck when the lab says nothing", () => {
    for (const absent of [undefined, null, {}, { height: 0, width: 0 }])
      expect(subjectSizeM(absent)).toBe(DUCK_SUBJECT_M);
  });
});

describe("recordShot", () => {
  it("leaves the duck's shot exactly as it was tuned", () => {
    const s = recordShot(subjectSizeM(SIZE.microduck), CAM_MAX_DIST);
    expect(s.dist).toBeCloseTo(DUCK_SHOT.dist, 2);
    expect(s.height).toBeCloseTo(DUCK_SHOT.height, 2);
    expect(s.az).toBe(DUCK_SHOT.az);
    expect(s.drift).toBe(DUCK_SHOT.drift);
  });

  it("frames an unknown body at the duck's shot", () => {
    for (const bad of [undefined, null, 0, -1, NaN])
      expect(recordShot(bad, CAM_MAX_DIST).dist).toBeCloseTo(DUCK_SHOT.dist, 6);
  });

  it("gets the lens out of the Asimov", () => {
    const s = recordShot(subjectSizeM(SIZE.asimov1), CAM_MAX_DIST);
    // 1.296 m of robot at 0.78 m was inside it; 2.82 subject-lengths out is
    // 3.65 m, the same picture the duck gets.
    expect(s.dist).toBeCloseTo(3.65, 1);
    expect(s.height).toBeCloseTo(1.59, 1);
    expect(s.dist).toBeGreaterThan(SIZE.asimov1.height);
  });

  it("keeps the shot's elevation whatever the subject", () => {
    const lens = (m: number) =>
      Math.atan2(recordShot(m, CAM_MAX_DIST).height, recordShot(m, CAM_MAX_DIST).dist);
    const duck = lens(DUCK_SUBJECT_M);
    for (const s of Object.values(SIZE))
      expect(lens(subjectSizeM(s))).toBeCloseTo(duck, 6);
  });

  it("never asks for a radius OrbitControls would claw back", () => {
    // `controls.update()` re-clamps to maxDistance every frame, enabled or
    // not — a shot further out than that is silently dragged in, and the take
    // is framed at a distance nobody chose. So it is scaled down WHOLE.
    const huge = recordShot(40, CAM_MAX_DIST);
    expect(Math.hypot(huge.dist, huge.height)).toBeLessThanOrEqual(CAM_MAX_DIST + 1e-9);
    expect(Math.atan2(huge.height, huge.dist)).toBeCloseTo(
      Math.atan2(DUCK_SHOT.height, DUCK_SHOT.dist), 6);
    // Every body the lab actually has fits without being clamped: the
    // clamp must be the guard rail, not the thing choosing the framing.
    for (const s of Object.values(SIZE)) {
      const free = recordShot(subjectSizeM(s));
      const held = recordShot(subjectSizeM(s), CAM_MAX_DIST);
      expect(held.dist).toBeCloseTo(free.dist, 9);
      expect(Math.hypot(held.dist, held.height)).toBeLessThan(CAM_MAX_DIST);
    }
  });
});
