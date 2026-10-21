// The two pure helpers the policy palette reads a run's RECORD through.
//
// The palette used to print the run directory verbatim
// ("teach-g1_imitate-g1_front_kick-25ac56-s2") and hand out a chain's last
// stage as "the whole trick". Both were wrong on their own terms: a user
// can't tell which run is the good one from a hash, and the stage that
// measured best is usually NOT the tail — the later stages are kept for
// comparison. These pin the fallbacks, because most runs have no record and
// must keep reading exactly as they always did.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";
import { chainPick, DEFAULT_ROBOTS, policyTitle, robotsInFrame, SceneLoads, type Policy } from "./lab";

function stage(label: string, extra: Partial<Policy> = {}): Policy {
  return {
    id: `runs:${label}`,
    label,
    group: "runs",
    path: `runs/${label}/policy.onnx`,
    ...extra,
  };
}

describe("policyTitle", () => {
  it("shows the record's title when the run has one", () => {
    const p = stage("teach-g1_imitate-g1_front_kick-25ac56-s2", {
      title: "Front kick (G1) · perform “g1-front-kick”",
    });
    expect(policyTitle(p)).toBe("Front kick (G1) · perform “g1-front-kick”");
  });

  it("falls back to the directory name when there is no record", () => {
    expect(policyTitle(stage("p-n256-s31"))).toBe("p-n256-s31");
  });

  it("strips the trainer's teach- prefix, like the chain header does", () => {
    expect(policyTitle(stage("teach-g1_front_kick-25ac56-s2"))).toBe(
      "g1_front_kick-25ac56-s2"
    );
  });

  it("leaves the raw label alone — it is what --init-from and the wire use", () => {
    const p = stage("teach-back-b18a5c-s5", { title: "Backflip" });
    policyTitle(p);
    expect(p.label).toBe("teach-back-b18a5c-s5");
  });
});

describe("chainPick", () => {
  it("returns the stage the record measured best, not the tail", () => {
    const stages = [stage("c-s1"), stage("c-s2", { pick: true }), stage("c-s3")];
    expect(chainPick(stages).label).toBe("c-s2");
  });

  it("returns the last stage when no stage was picked", () => {
    const stages = [stage("c-s1"), stage("c-s2"), stage("c-s3")];
    expect(chainPick(stages).label).toBe("c-s3");
  });

  it("returns a pick that sits first, where the tail default is furthest off", () => {
    const stages = [stage("c-s1", { pick: true }), stage("c-s2"), stage("c-s3")];
    expect(chainPick(stages).label).toBe("c-s1");
  });

  it("returns the only stage of a one-stage chain either way", () => {
    expect(chainPick([stage("c-s1")]).label).toBe("c-s1");
    expect(chainPick([stage("c-s1", { pick: true })]).label).toBe("c-s1");
  });
});

describe("DEFAULT_ROBOTS", () => {
  it("describes the duck fully, so a lab too old to send `robots` is not crippled", () => {
    // The palette, the 🎓 panel and the 🎬 panel all read these fields now.
    // A missing `animate` used to be harmless (every listed body was posable);
    // with the flag in play, an undescribed duck would drop out of the editor.
    expect(DEFAULT_ROBOTS).toHaveLength(1);
    const [duck] = DEFAULT_ROBOTS;
    expect(duck).toMatchObject({ id: "microduck", kind: "legged", ready: true, animate: true });
  });
});

describe("robotsInFrame", () => {
  it("names one mesh set per body on the stage, not one per row", () => {
    // The viewer fetches GET /scene?robot= once per distinct body; a duck
    // roster must still ask for exactly one.
    expect(robotsInFrame([{ robot: "mars" }, { robot: "mars" }, {}])).toEqual(["mars", "microduck"]);
    expect(robotsInFrame([{}, {}])).toEqual(["microduck"]);
  });

  it("passes a discovered id through untouched", () => {
    // "menagerie:unitree_go2" has to survive as a whole: it is what
    // GET /scene?robot= is asked for, and half of it names nothing.
    expect(robotsInFrame([{ robot: "menagerie:unitree_go2" }])).toEqual([
      "menagerie:unitree_go2",
    ]);
  });
});

describe("SceneLoads — the mesh-set gate", () => {
  // The bug this class exists to make impossible, replayed as the sequence
  // that produced it on 2026-10-06: two Asimovs on the stage, the lab live,
  // the HUD listing both policies with moving counters, and not one body
  // drawn. Removing and re-spawning the ducks did nothing, because the page
  // had already decided it had asimov1's meshes.
  it("a scene DROPPED after teardown stays fetchable", () => {
    const l = new SceneLoads();
    expect(l.wants("asimov1")).toBe(true);
    l.begin("asimov1");
    expect(l.wants("asimov1")).toBe(false);   // in flight

    // Strict Mode unmounts; the fetch resolves into the dead closure.
    l.dropped("asimov1");

    expect(l.loadedRobots()).toEqual([]);     // NOT loaded — it was binned
    expect(l.wants("asimov1")).toBe(true);    // ...so the next mount refetches
  });

  it("only a STORED scene closes the gate", () => {
    const l = new SceneLoads();
    l.begin("g1");
    l.stored("g1");
    expect(l.loadedRobots()).toEqual(["g1"]);
    expect(l.wants("g1")).toBe(false);        // never fetched twice
  });

  it("gives up on a lab that does not have the robot, but not before 5 tries", () => {
    const l = new SceneLoads();
    for (let i = 0; i < 5; i++) {
      expect(l.wants("go2")).toBe(true);
      l.begin("go2");
      l.failed("go2");
    }
    expect(l.wants("go2")).toBe(false);       // 404 forever — stop polling
  });

  it("a failing robot that finally answers is not held back by its failures", () => {
    const l = new SceneLoads();
    l.begin("mars"); l.failed("mars");        // lab restarting mid-session
    expect(l.wants("mars")).toBe(true);
    l.begin("mars"); l.stored("mars");
    expect(l.loadedRobots()).toEqual(["mars"]);
  });
  // The class above can only hold the invariant if its ONE caller uses it the
  // right way round, and the gap between the rule and the call site is
  // precisely where this bug lived: the bookkeeping was three loose refs
  // inline in the effect, and nothing could have tested it. Checked against
  // the source, the way inset.test checks three's renderer.
  it("Viewer's torn-down path calls dropped(), never stored()", () => {
    const src = readFileSync(
      fileURLToPath(new URL("../components/Viewer.tsx", import.meta.url)),
      "utf8",
    );
    expect(src, "Viewer must use the shared gate").toContain("new SceneLoads()");
    expect(src, "the loose refs must not come back").not.toMatch(/loadedScenes|pendingScenes|sceneFailures/);

    // The `stopped` branch — everything between `if (stopped) {` and its
    // close — must bin the scene and leave the robot fetchable.
    const at = src.indexOf("if (stopped) {");
    expect(at, "the fetch must still guard on teardown").toBeGreaterThan(-1);
    const branch = src.slice(at, src.indexOf("}", src.indexOf("return;", at)));
    expect(branch, "a dropped scene must stay fetchable").toContain("dropped(robot)");
    expect(branch, "a dropped scene must NOT close the gate").not.toContain("stored(robot)");
  });
});

describe("SceneLoads.describe — naming a silent failure", () => {
  // The blank stage of 2026-10-06 produced no error, no failed request and a
  // green "live" badge. Three wrong diagnoses came out of that (a stale
  // trainee, a dead render loop, a camera parked off-stage), each argued from
  // one sequence. These four states are what a caller should be able to say
  // instead of guess.
  it("walks waiting -> fetching -> ready", () => {
    const l = new SceneLoads();
    expect(l.describe("asimov1").state).toBe("waiting");
    l.begin("asimov1");
    expect(l.describe("asimov1").state).toBe("fetching");
    l.stored("asimov1");
    expect(l.describe("asimov1").state).toBe("ready");
  });

  it("reports failures with their count, and a dropped scene as NOT ready", () => {
    const l = new SceneLoads();
    l.begin("go2"); l.failed("go2");
    expect(l.describe("go2")).toEqual({ state: "failed", failures: 1 });

    const m = new SceneLoads();
    m.begin("asimov1"); m.dropped("asimov1");
    // The whole point: a binned scene must never look ready.
    expect(m.describe("asimov1").state).not.toBe("ready");
  });
});
