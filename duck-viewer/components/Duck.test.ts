// What `buildBodyGeometries` does with a body's OWN normals.
//
// A dump that ships `SceneMesh.n` has to be drawn with it VERBATIM: no weld,
// no `computeVertexNormals`, no crease guess. Inventing them instead is what
// made every Menagerie model arrive melted — off by more than 15 deg on 14 %
// of the Go2's face corners and 55 % of the SO-ARM100's — and the failure is
// silent, because a geometry with wrong normals is a geometry, renders fine,
// and only LOOKS like a dented robot.
//
// These cases were each checked against a planted regression (drop the
// `shipped` branch, flip the all-or-nothing to per-mesh, forget `normalScale`)
// before being written down; a test nobody has seen fail proves nothing.

import { describe, expect, it } from "vitest";
import type * as THREE from "three";
import type { Scene } from "@/lib/lab";
import { buildBodyGeometries } from "./Duck";

/** A unit cube's corner, as a dump: eight positions, twelve triangles, and
 *  normals that are nothing a weld could reconstruct — each vertex normal
 *  points at +z, which is FLAT where the geometry is a box. That is the
 *  point: if the result's normals are all +z the shipped array was used, and
 *  if they are the cube's own face normals it was recomputed. */
function cubeScene(opts: { normals?: boolean; vertScale?: number } = {}): Scene {
  const s = 1 / (opts.vertScale ?? 1);
  const v = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
  ].flatMap(([x, y, z]) => [x * s, y * s, z * s]);
  const f = [
    0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
    1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0,
  ];
  const mesh: Scene["meshes"][number] = { v, f };
  if (opts.normals) mesh.n = Array.from({ length: 8 }, () => [0, 0, 1000]).flat();
  const scene: Scene = {
    bodies: ["world", "chassis"],
    meshes: [mesh],
    geoms: [{
      mesh: 0, body: 1, pos: [0, 0, 0], quat: [1, 0, 0, 0],
      mat: "shell", rgba: [0.5, 0.5, 0.5, 1],
    }],
  };
  if (opts.vertScale) scene.vertScale = opts.vertScale;
  if (opts.normals) scene.normalScale = 0.001;
  return scene;
}

function normalsOf(geometry: THREE.BufferGeometry | null): number[][] {
  const a = geometry!.getAttribute("normal");
  return Array.from({ length: a.count }, (_, i) => [a.getX(i), a.getY(i), a.getZ(i)]);
}

describe("buildBodyGeometries, with a body that ships its own normals", () => {
  it("draws with them instead of recomputing, so an authored flat face stays flat", () => {
    const [, chassis] = buildBodyGeometries(cubeScene({ normals: true }), null, {
      look: "generic",
    });
    // Every normal is the authored +z, including on the four side faces a
    // recompute would have pointed along ±x / ±y.
    for (const n of normalsOf(chassis.geometry)) {
      expect(n[0]).toBeCloseTo(0, 5);
      expect(n[1]).toBeCloseTo(0, 5);
      expect(n[2]).toBeCloseTo(1, 5);
    }
  });

  // NOT tested here: that `normalScale` is applied. It cannot be, from
  // outside — `BufferGeometry.applyMatrix4` runs on every geom and NORMALISES
  // the normal attribute, so a dump read as raw 1000s comes out identical to
  // one read as unit vectors. A plant that dropped the rescale passed all six
  // of these, which is why this paragraph is here instead of a green test:
  // the scale is applied because the attribute should be right, not because
  // anything downstream would notice.

  it("keeps the vertex count, because a shipped-normal mesh is never welded", () => {
    const shipped = buildBodyGeometries(cubeScene({ normals: true }), null, { look: "generic" });
    expect(shipped[1].geometry!.getAttribute("position").count).toBe(8);
  });

  it("falls back to inventing them when ANY mesh lacks them", () => {
    // All-or-nothing: a welded part can come back non-indexed and a shipped
    // one is indexed, and `mergeGeometries` refuses a body whose parts
    // disagree — so one mesh the lab could not ship must not take out a body.
    const scene = cubeScene({ normals: true });
    scene.meshes.push({ v: scene.meshes[0].v.slice(), f: scene.meshes[0].f.slice() });
    scene.geoms.push({ ...scene.geoms[0], mesh: 1, pos: [2, 0, 0] });
    const [, chassis] = buildBodyGeometries(scene, null, { look: "generic" });
    expect(chassis.geometry).not.toBeNull();
    // Recomputed: a cube's corners cannot all point at +z.
    expect(normalsOf(chassis.geometry).some((n) => Math.abs(n[2]) < 0.99)).toBe(true);
  });

  it("still scales POSITIONS by vertScale, which normals do not share", () => {
    const scene = cubeScene({ normals: true, vertScale: 1e-5 });
    const [, chassis] = buildBodyGeometries(scene, null, { look: "generic" });
    const p = chassis.geometry!.getAttribute("position");
    const xs = Array.from({ length: p.count }, (_, i) => p.getX(i));
    expect(Math.max(...xs)).toBeCloseTo(1, 5);   // a 1 m cube, not a 100 km one
    // and the lattice did not reach the normals: still the authored +z.
    for (const n of normalsOf(chassis.geometry)) expect(n[2]).toBeCloseTo(1, 5);
  });

  it("leaves a body with no normals exactly as it was", () => {
    const [, chassis] = buildBodyGeometries(cubeScene(), null, { look: "generic" });
    expect(chassis.geometry).not.toBeNull();
    expect(normalsOf(chassis.geometry).some((n) => Math.abs(n[2]) < 0.99)).toBe(true);
  });
});
