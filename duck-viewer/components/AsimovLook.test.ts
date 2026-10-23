// The one thing in AsimovLook.tsx a renderer can get wrong without anybody
// noticing: which part kind each of the 26 links lands on. The colours are a
// look and are judged by looking (.claude/skills/sim-smoke); the SPLIT is a
// claim about the model, and it is checked here against the model's own link
// names — all 26 of them, because a rule that silently stops matching leaves
// the robot flat grey again and nothing fails.

import { describe, expect, it } from "vitest";
import { ASIMOV_KINDS, asimovPartKind } from "./AsimovLook";

/** Every visual link in `.cache/asimov_1/xmls/asimov_1.xml`, with the mesh
 *  name the dump sends for it. 26 meshes on 27 bodies — the 27th is `world`. */
const LINKS: readonly (readonly [string, string])[] = [
  ["pelvis_link", "pelvis.STL"],
  ["waist_yaw_link", "waist_yaw_link.STL"],
  ["neck_yaw_link", "neck_yaw_link.STL"],
  ["neck_pitch_link", "neck_pitch_link.STL"],
  ...(["left", "right"] as const).flatMap((s) => [
    [`${s}_hip_pitch_link`, `${s}_hip_pitch_link.STL`],
    [`${s}_hip_roll_link`, `${s}_hip_roll_link.STL`],
    [`${s}_hip_yaw_link`, `${s}_hip_yaw_link.STL`],
    [`${s}_knee_link`, `${s}_knee_link.STL`],
    [`${s}_ankle_pitch_link`, `${s}_ankle_pitch_link.STL`],
    [`${s}_ankle_roll_link`, `${s}_ankle_roll_link.STL`],
    [`${s}_shoulder_pitch_link`, `${s}_shoulder_pitch_link.STL`],
    [`${s}_shoulder_roll_link`, `${s}_shoulder_roll_link.STL`],
    [`${s}_shoulder_yaw_link`, `${s}_shoulder_yaw_link.STL`],
    [`${s}_elbow_link`, `${s}_elbow_link.STL`],
    [`${s}_wrist_yaw_link`, `${s}_wrist_yaw_link.STL`],
  ] as const),
];

describe("which material each Asimov link takes", () => {
  it("covers all 26 visual links", () => {
    expect(LINKS).toHaveLength(26);
  });

  it("gives every link a kind the material array actually has", () => {
    // `sharedKinds.indexOf(kind)` in SimStage falls back to 0 on a miss, so a
    // kind outside the table would silently paint a part as a SHELL.
    for (const [body, mesh] of LINKS) {
      expect(ASIMOV_KINDS, `${body}`).toContain(asimovPartKind(body, mesh));
    }
  });

  it("calls the long structural links shells", () => {
    // Every one of these is over 148 mm on its longest axis — the torso, the
    // thigh, the shin, the upper arm, the forearm, the head, the pelvis.
    for (const n of ["left_knee_link", "right_hip_yaw_link", "pelvis_link",
                     "left_shoulder_roll_link", "right_elbow_link"]) {
      expect(asimovPartKind(n, `${n}.STL`), n).toBe("shell");
    }
  });

  it("calls the compact actuator housings joints", () => {
    // Every one of these is under 136 mm — a block between two long shells.
    for (const n of ["left_ankle_pitch_link", "right_ankle_pitch_link",
                     "neck_yaw_link"]) {
      expect(asimovPartKind(n, `${n}.STL`), n).toBe("joint");
    }
  });

  it("calls the HIP housings silver, not bronze", () => {
    // The hip plate is bare metal on the real robot and the bronze starts at
    // the knee (menlo.ai's hero shots). These two are the same SIZE as the
    // other housings — the split is by what they are made of, not by extent,
    // which is why they are named rather than measured into place.
    for (const n of ["left_hip_pitch_link", "right_hip_pitch_link",
                     "left_hip_roll_link", "right_hip_roll_link"]) {
      expect(asimovPartKind(n, `${n}.STL`), n).toBe("silver");
    }
  });

  it("reds the torso, both faces, because it is one mesh", () => {
    // The chest is red on the real robot and the back is not, but the model
    // gives the whole torso one mesh and it cannot be cut — see the note in
    // robots/asimov_walk.py. All-red is the closer compromise: the robot
    // carries a red plate on its upper back too.
    // The torso is CUT: the vest half is red, the rest black. Both halves
    // still carry the body name, so the mesh rule has to win — that ordering
    // is the whole case.
    expect(asimovPartKind("waist_yaw_link", "asimov_torso_vest")).toBe("red");
    expect(asimovPartKind("waist_yaw_link", "asimov_torso_vest_rest")).toBe("shell");
    // the soles, cut off the bottom of each foot the same way
    expect(asimovPartKind("left_ankle_roll_link", "asimov_sole_l")).toBe("sole");
    expect(asimovPartKind("left_ankle_roll_link", "asimov_sole_l_rest")).toBe("foot");
    // and the helmet, CUT by connected component: a red rim and a dark visor
    expect(asimovPartKind("neck_pitch_link", "asimov_visor_rest")).toBe("red");
    expect(asimovPartKind("neck_pitch_link", "asimov_visor")).toBe("shell");
  });

  it("leaves the shoulder rotators BLACK", () => {
    // These were briefly bronze. In menlo.ai's shots the only gold on the
    // shoulder is a pad on the very top; the rotator barrel under it is
    // black, and the pad is not separable from the barrel.
    for (const n of ["left_shoulder_pitch_link",
                     "right_shoulder_pitch_link", "left_shoulder_yaw_link",
                     "right_shoulder_yaw_link"]) {
      expect(asimovPartKind(n, `${n}.STL`), n).toBe("shell");
    }
  });

  it("calls the wrists hands", () => {
    // The one entry that paints a part the robot does not have: there is no
    // gripper mesh, only the stub it would mount on.
    for (const n of ["left_wrist_yaw_link", "right_wrist_yaw_link"]) {
      expect(asimovPartKind(n, `${n}.STL`), n).toBe("hand");
    }
  });

  it("calls ankle_roll the FOOT and not a joint", () => {
    // The one link named for a joint that is a shell: it is the sole, and it
    // is the only one the `ankle_` rule must not swallow. Ordering in
    // ASIMOV_RULES is what decides this.
    expect(asimovPartKind("left_ankle_roll_link", "left_ankle_roll_link.STL")).toBe("foot");
    expect(asimovPartKind("right_ankle_roll_link", "right_ankle_roll_link.STL")).toBe("foot");
  });

  it("matches on the BODY name when a re-export renames the mesh files", () => {
    // The dump sends both; a vendor who renames `left_ankle_roll_link.STL` to
    // `foot_L.stl` must not repaint the robot.
    expect(asimovPartKind("left_ankle_roll_link", "foot_L.stl")).toBe("foot");
    expect(asimovPartKind("right_hip_roll_link", "part_17.stl")).toBe("silver");
  });

  it("falls back to the shell for a link it has never seen", () => {
    expect(asimovPartKind("some_new_link", "some_new_link.STL")).toBe("shell");
  });
});
