"use client";

// How a Menlo Asimov 1 is drawn — the /sim page's Asimov person
// (SimStage.tsx, `look: "asimov1"`), and whatever draws him next.
//
// WHY THIS FILE EXISTS. Asimov's MJCF paints the whole robot with ONE
// material: `visual_grey`, rgba 0.7 0.7 0.7, set on a defaults class that
// every one of the 26 visual geoms inherits (`.cache/asimov_1/xmls/
// asimov_1.xml` line 24), and the URDF beside it carries no `<color>` at
// all. So a body with no look of its own arrives as 26 identically flat
// grey parts — which is what he did, and it reads as a mannequin rather
// than a machine. Both other CAD bodies here have a look for the same
// reason (G1Look.tsx, MarsLook.tsx); this is Asimov's.
//
// WHAT IS MEASURED AND WHAT IS NOT. The SPLIT is measured: the link names
// are the model's, and the extents below are its meshes' bounding boxes, so
// "this link is a compact actuator housing between two long shells" is a
// fact about the robot. The COLOURS are measured too, but off menloresearch
// /asimov-mjlab's own README images rather than off the model — the MJCF
// states one grey and nothing in it can say what the machine is painted.
// This table replaced an invented white-and-dark-grey one on 2026-10-03,
// the day those renders were looked at; it is still a reading of a picture,
// and per-link materials from Menlo would replace it again.
//
// Clustering the CAD render gives #141311 on 49 % of the robot (the long
// shells), #7c3f08 and #562b06 on 19 % (the thigh plates, knee discs, hip
// covers and top plate, lit and shadowed), and #7b7c82 / #d1d1d4 on the
// clear shoe covers. The hardware photo beside it gives #373939 / #333b40
// shells and black shoes. The PHOTO's shell wins: the render sits on a white
// page where near-black is the strongest thing on it, the lab stage is
// #0c0d11, and a #141311 robot there is a silhouette with no shape in it —
// the same trade MarsLook.tsx measured from the other end.
//
// The Python side paints the SPEC with the same three colours and the same
// rule (`robots/asimov_walk.paint_shell`), which is what puts them in
// MuJoCo renders and in the lab stage's generic paint. Change one, change
// both: they are two readings of one table, not two tables.

import { useThree } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import { roomEnv } from "./G1Look";

export const ASIMOV_KINDS = ["shell", "joint", "silver", "red", "hand", "sole", "foot"] as const;
export type AsimovPartKind = (typeof ASIMOV_KINDS)[number];

/** The long covers: graphite, the hardware photo's shell colour. */
export const ASIMOV_SHELL = "#33363b";
/** The actuator housings: the CAD render's burnt orange. The render's own
 *  midtones are #7c3f08 lit and #562b06 shadowed, which are a dim key
 *  light's — this is the albedo behind them, checked by rendering it back
 *  under MuJoCo's lights beside the crop. */
export const ASIMOV_JOINT = "#c26a14";
/** The chest plate and the head. 22,484 strongly-red torso pixels in
 *  menlo.ai's standing shot have a median of #fb4d2b; taken down here
 *  because that median is a lit surface and the albedo behind it is darker.
 *  A shell at the literal sample clips to flat vermilion under the stage's
 *  own key light. */
export const ASIMOV_RED = "#d63b24";
/** The hands. The real grippers are white (#e1e3e3 / #d2d4d2). NOTE this
 *  model has no gripper — `wrist_yaw_link` is a 19x38x38 mm stub where the
 *  claw would mount — so this paints a part the robot does not have. */
export const ASIMOV_HAND = "#dcdede";
/** The sole: magenta on the real robot. LIVE — `ASIMOV_RULES` below matches
 *  it on `asimov_sole_l` / `_r`, the halves `robots/asimov_walk.CUT_PARTS`
 *  splits off each foot's source STL by which way the face POINTS. (A note
 *  here once said cutting was forbidden and that nothing read this. Both
 *  were wrong, written the afternoon a cut of the COMPILED mesh broke the
 *  model; cutting the file on disk has always worked — `mars.split_base_mesh`
 *  predates all of it.) */
export const ASIMOV_SOLE = "#d62658";
/** The hip housings: bare machined metal. menlo.ai's standing shot samples
 *  the outer hip plate at #d4cbc0 — warm, because the workshop light is —
 *  neutralised here so it reads as metal rather than a second, duller
 *  accent beside the bronze. */
export const ASIMOV_SILVER = "#bfc3c8";
/** The sole: black in the photograph. The CAD's clear covers read #7b7c82,
 *  and a sole lighter than the leg above it reads as snow. */
export const ASIMOV_FOOT = "#1b1d21";

/** Which material an Asimov geom takes, from its body and mesh name.
 *
 *  The rules are the model's own link names, and the reason each one is on
 *  the side it is on is the mesh's measured extent (mm, longest last):
 *
 *    SHELL   waist_yaw 199x251x392 · knee 109x131x370 (shin) ·
 *            hip_yaw 117x128x317 (thigh) · neck_pitch 132x173x193 (head) ·
 *            pelvis 132x135x185 · shoulder_roll 75x100x183 (upper arm) ·
 *            elbow 73x76x148 (forearm)
 *    JOINT   shoulder_yaw 69x66x136 · shoulder_pitch 83x120x91 ·
 *            hip_pitch 104x112x111 · hip_roll 53x50x105 ·
 *            neck_yaw 62x71x76 · ankle_pitch 51x52x53 ·
 *            wrist_yaw 19x38x38
 *    FOOT    ankle_roll 60x76x196
 *
 *  Every shell is longer than 148 mm and every housing shorter than 136 mm,
 *  so the split is a real gap in the model and not a judgement call — but it
 *  is written by NAME rather than by a threshold, because a 12 mm gap is a
 *  thin thing to hang a renderer on and a name can be read by the next
 *  person. `ankle_roll` is the one link named for a joint that is a shell:
 *  it is the foot.
 *
 *  Matched on the BODY name as well as the mesh name, like `marsPartKind` —
 *  the dump sends both, the meshes here happen to be `<link>.STL`, and a
 *  re-export that renames the files should not silently repaint the robot. */
const ASIMOV_RULES: readonly (readonly [RegExp, AsimovPartKind])[] = [
  // CUT parts first, always. The lab splits a source STL where the robot is
  // two colours and the model is one mesh (`robots/asimov_walk.CUT_PARTS`),
  // and both halves still carry the BODY name they came from — so a
  // body-name rule below would swallow them. `asimov_sole_l` under the
  // `ankle_roll` rule is a magenta foot, which is what happened first time.
  [/asimov_sole_[lr]$/, "sole"],
  [/asimov_sole_[lr]_rest/, "foot"],
  // The helmet: a red outer rim with a dark visor across the face and crown.
  // The visor is its own SHELL in the mesh, so the lab cuts it by connected
  // component rather than by a plane (`robots/asimov_walk._triangle_shells`).
  [/asimov_visor$/, "shell"],
  [/asimov_visor_rest/, "red"],
  [/asimov_torso_vest$/, "red"],
  [/asimov_torso_vest_rest/, "shell"],
  [/ankle_roll/, "foot"],
  // The hip plate is bare metal on the real robot and the bronze starts at
  // the knee. `robots/asimov_walk.SILVER_LINKS` is the same rule.
  [/hip_pitch|hip_roll/, "silver"],
  // The helmet, taking the chest's red: it samples #cf2a18 to the chest's
  // #f7472a in menlo.ai's walking shots — the same hue, shaded, because it
  // faces away from the key light. Whole-link, so its black visor reds with
  // it; the visor is a candidate for a CUT_PARTS entry, not a repaint.
  // `waist_yaw` reaches here only as the UNCUT torso, which takes red; the
  // cut halves are matched above. (This said the torso could not be cut and
  // that the repo forbids it — neither is true, see the note on ASIMOV_SOLE.)
  [/neck_pitch|waist_yaw/, "red"],
  [/wrist_yaw/, "hand"],
  // NOT the shoulder rotators: the only gold on the shoulder is a small pad
  // on top, and the barrel under it is black in both menlo.ai shots. NOT the
  // helmet either — it carries a big black visor and reads black.
  [/ankle_pitch|neck_yaw/, "joint"],
];
export function asimovPartKind(bodyName?: string | null, meshName?: string | null): AsimovPartKind {
  const names = [bodyName, meshName].filter(Boolean).map((s) => String(s).toLowerCase());
  for (const [re, kind] of ASIMOV_RULES) {
    if (names.some((n) => re.test(n))) return kind;
  }
  return "shell";
}

/** The material for one part kind.
 *
 *  The shell takes a clearcoat and almost no metalness: it is the gloss on a
 *  painted or moulded panel, and on a DARK shell that gloss is most of what
 *  separates it from a dark stage — the colour alone cannot (the trade
 *  MarsLook.tsx measured from the pale end). The housings take metalness and
 *  a lower roughness because they are machined, anodised cases, and the foot
 *  takes neither — a sole is rubber, and a specular highlight on it reads as
 *  wet floor. */
export function makeAsimovMaterial(
  kind: AsimovPartKind,
  opts: { envMap?: THREE.Texture | null; envScale?: number } = {}
): THREE.Material {
  const envMap = opts.envMap ?? null;
  const env = opts.envScale ?? 1;
  switch (kind) {
    case "joint":
      return new THREE.MeshPhysicalMaterial({
        envMap, color: ASIMOV_JOINT, roughness: 0.34, metalness: 0.55,
        clearcoat: 0.25, clearcoatRoughness: 0.3, envMapIntensity: 1.0 * env,
      });
    case "silver":
      // Brushed aluminium: more metalness than the anodised housings and a
      // touch rougher, so it catches the room without turning into a mirror.
      return new THREE.MeshPhysicalMaterial({
        envMap, color: ASIMOV_SILVER, roughness: 0.30, metalness: 0.78,
        clearcoat: 0.15, clearcoatRoughness: 0.35, envMapIntensity: 1.15 * env,
      });
    case "red":
      // A painted, lacquered panel: the chest is the glossiest thing on the
      // robot in every photograph of it.
      return new THREE.MeshPhysicalMaterial({
        envMap, color: ASIMOV_RED, roughness: 0.22, metalness: 0.04,
        clearcoat: 0.85, clearcoatRoughness: 0.12, envMapIntensity: 1.1 * env,
      });
    case "hand":
      return new THREE.MeshPhysicalMaterial({
        envMap, color: ASIMOV_HAND, roughness: 0.42, metalness: 0.03,
        clearcoat: 0.3, clearcoatRoughness: 0.3, envMapIntensity: 0.8 * env,
      });
    case "sole":
      return new THREE.MeshStandardMaterial({
        envMap, color: ASIMOV_SOLE, roughness: 0.72, metalness: 0.03,
        envMapIntensity: 0.5 * env,
      });
    case "foot":
      return new THREE.MeshStandardMaterial({
        envMap, color: ASIMOV_FOOT, roughness: 0.78, metalness: 0.04,
        envMapIntensity: 0.5 * env,
      });
    default:
      return new THREE.MeshPhysicalMaterial({
        envMap, color: ASIMOV_SHELL, roughness: 0.33, metalness: 0.08,
        clearcoat: 0.6, clearcoatRoughness: 0.22, envMapIntensity: 1.0 * env,
      });
  }
}

/** The Asimov material array, indexed like ASIMOV_KINDS. Null when `enabled`
 *  is false, so a stage with no Asimov on it pays nothing — the same contract
 *  as `useG1Materials` and `useMarsMaterials`, including the `envScale` split
 *  (the lab stage has its own lights and passes 0.35; /sim has none and
 *  passes 1). */
export function useAsimovMaterials(
  enabled: boolean,
  opts: { envScale?: number } = {}
): THREE.Material[] | null {
  const gl = useThree((s) => s.gl);
  const { envScale = 0.35 } = opts;
  return useMemo(() => {
    if (!enabled) return null;
    const envMap = roomEnv(gl);
    return ASIMOV_KINDS.map((k) => makeAsimovMaterial(k, { envMap, envScale }));
  }, [enabled, gl, envScale]);
}
