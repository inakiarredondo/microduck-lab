// The view home (Shift+R) and the dolly ceiling, pinned. Same disease as the
// fog band and the 🎥 shot: 1.93 m of orbit radius is 7 duck-lengths and half
// an Asimov, so the reset that shows the duck on its stage put a 1.296 m robot
// through the top of the frame. Both now scale with what is actually there.

import { describe, expect, it } from "vitest";
import {
  DUCK_HOME,
  DUCK_SUBJECT_M,
  homeView,
  SIM_HOME,
  SIM_HOME_ROOM_M,
  stageMaxDist,
} from "@/lib/camera";

const LAB_MAX = 7; // the duck's dolly ceiling, and stageMaxDist's base
const SIM_MAX = 12; // /sim's — a room needs more than the lab floor

const radius = (h: { p: number[]; t: number[] }) =>
  Math.hypot(h.p[0] - h.t[0], h.p[1] - h.t[1], h.p[2] - h.t[2]);

/** The view DIRECTION, which scaling must never touch. */
const dir = (h: { p: number[]; t: number[] }) => {
  const r = radius(h);
  return [(h.p[0] - h.t[0]) / r, (h.p[1] - h.t[1]) / r, (h.p[2] - h.t[2]) / r];
};

// `lab/robots.measured_size`, the same bodies lib/shot.test.ts uses.
const SIZE = { duck: 0.27718, asimov1: 1.29608, g1: 1.30711, go2: 0.75359 };

describe("homeView on the lab stage", () => {
  it("leaves a duck stage exactly where it has always been", () => {
    const h = homeView(DUCK_HOME, SIZE.duck, DUCK_SUBJECT_M, LAB_MAX);
    expect(h.p[0]).toBeCloseTo(1.2, 3);
    expect(h.p[1]).toBeCloseTo(0.7, 3);
    expect(h.p[2]).toBeCloseTo(1.4, 3);
    expect(h.t).toEqual([0, expect.closeTo(0.12, 3), 0]);
    expect(radius(h)).toBeCloseTo(1.933, 3);
  });

  it("knows nothing is on the stage yet", () => {
    for (const absent of [undefined, null, 0, -1, NaN])
      expect(radius(homeView(DUCK_HOME, absent, DUCK_SUBJECT_M, LAB_MAX)))
        .toBeCloseTo(1.933, 3);
  });

  it("pulls back for an Asimov by exactly its size", () => {
    const k = SIZE.asimov1 / DUCK_SUBJECT_M; // 4.68
    const h = homeView(DUCK_HOME, SIZE.asimov1, DUCK_SUBJECT_M, stageMaxDist(SIZE.asimov1));
    expect(radius(h)).toBeCloseTo(1.933 * k, 2); // 9.04 m, not 1.93
    expect(h.t[1]).toBeCloseTo(0.12 * k, 3); // aims at its chest, not its shins
  });

  it("only ever changes the view's SIZE, never its angle", () => {
    const d0 = dir(homeView(DUCK_HOME, SIZE.duck, DUCK_SUBJECT_M, LAB_MAX));
    for (const s of Object.values(SIZE)) {
      const d = dir(homeView(DUCK_HOME, s, DUCK_SUBJECT_M, stageMaxDist(s)));
      for (let i = 0; i < 3; i++) expect(d[i]).toBeCloseTo(d0[i], 9);
    }
  });
});

describe("stageMaxDist", () => {
  it("is the duck's 7 m on a duck stage, and never less", () => {
    expect(stageMaxDist(SIZE.duck)).toBeCloseTo(LAB_MAX, 6);
    expect(stageMaxDist(0.05)).toBe(LAB_MAX); // a small body gets no smaller room
    expect(stageMaxDist(undefined)).toBe(LAB_MAX);
  });

  it("is a guard rail, never the thing that frames the shot", () => {
    // The whole point: a ceiling INSIDE the home view is the clamp choosing
    // the framing (lib/camera.effectiveClamp, from the dolly's end). At the
    // fixed 7 m the Asimov's 9.04 m home was exactly that.
    for (const s of Object.values(SIZE)) {
      const max = stageMaxDist(s);
      expect(radius(homeView(DUCK_HOME, s, DUCK_SUBJECT_M, max))).toBeLessThanOrEqual(max);
      expect(radius(homeView(DUCK_HOME, s, DUCK_SUBJECT_M))).toBeLessThanOrEqual(max + 1e-9);
    }
  });

  it("stops at the floor's own rim", () => {
    // The stage floor is a 9 m disc: past 18 m there is nothing to frame.
    expect(stageMaxDist(40)).toBe(18);
    // Uncapped it would be 7 x 4.68 = 32.7 m, from which a 1.3 m robot is
    // four pixels of background.
    expect((7 * SIZE.asimov1) / DUCK_SUBJECT_M).toBeCloseTo(32.7, 1);
  });
});

describe("homeView on /sim, where the subject is the room", () => {
  it("leaves the playroom it was tuned on exactly as it was", () => {
    const h = homeView(SIM_HOME, SIM_HOME_ROOM_M, SIM_HOME_ROOM_M, SIM_MAX);
    expect(h.p).toEqual(SIM_HOME.p);
    expect(radius(h)).toBeCloseTo(3.58, 2);
  });

  it("stands back from the Asimov's 8 m hall and leans into MOSS's 2.6 m yard", () => {
    const hall = homeView(SIM_HOME, 8.0, SIM_HOME_ROOM_M, SIM_MAX);
    const yard = homeView(SIM_HOME, 2.6, SIM_HOME_ROOM_M, SIM_MAX);
    expect(radius(hall)).toBeCloseTo(8.17, 2);
    expect(radius(yard)).toBeCloseTo(2.66, 2);
    // Every room this lab ships still fits inside /sim's own ceiling.
    for (const room of [2.6, 3.5, 3.9, 4.3, 6.0, 6.5, 7.0, 8.0])
      expect(radius(homeView(SIM_HOME, room, SIM_HOME_ROOM_M, SIM_MAX)))
        .toBeLessThanOrEqual(SIM_MAX);
  });
});
