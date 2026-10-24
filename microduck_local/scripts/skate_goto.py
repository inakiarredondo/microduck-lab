"""Send the duck to rooms of a real floor plan — on skates and on foot — and score every trip.

    uv run python scripts/skate_goto.py casa                    # private/casa/plan.json
    uv run python scripts/skate_goto.py casa --trials 4 --trips 6 --seconds 90
    uv run python scripts/skate_goto.py casa --mode walk            # on foot only (default: both)

A floor plan (`world/floorplan.py`, PRIVATE — it never leaves your machine)
becomes a world; one duck — on roller skates (`pollen:roller`) or on its
feet (`pollen:alpha_walking`) — is driven by the `skate_route` brain's
`goto(room)` (PRIVILEGED: true pose, true map). Both modes see the same
seeds, so their trips are the same trips. Each
trial spawns it in a random room at a random heading and asks for `--trips`
random rooms in a row, each different from the last. Per trip:

    ok        the brain says it arrived AND the trunk is in that room
    time      seconds the trip took (cap `--seconds`)
    hits      contact episodes between the duck and anything static
    stairs    did the trunk enter the stairwell (no wall there: a fall)
    edge_m    the trunk's closest approach to the stairwell, m
    falls     the world's own fall count during the trip
"""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import numpy as np

from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.runtime import Senses, attach_world
from microduck_local.brain.skate import MODES, SkateRoute
from microduck_local.world import World
from microduck_local.world.floorplan import FloorPlan, plan_path


def _static_geoms(model: mujoco.MjModel) -> set[int]:
    return {g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("wall", "box"))}


def _duck_geoms(model: mujoco.MjModel, root: int) -> set[int]:
    out = set()
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        while b > 0 and b != root:
            b = model.body_parentid[b]
        if b == root:
            out.add(g)
    return out


def trial(fp: FloorPlan, seed: int, trips: int, seconds: float, verbose: bool,
          mode: str = "roller") -> list[dict]:
    rng = np.random.default_rng(seed)
    names = sorted(fp.rooms)
    start = str(rng.choice(names))
    sc = fp.scenario(spawn_room=start, yaw=float(rng.uniform(-math.pi, math.pi)), mode=mode)
    onnx = POLICIES_DIR / (MODES[mode]["policy"].split(":", 1)[1] + ".onnx")
    w = World(sc, infer_for={"d0": onnx_infer(onnx)}, seed=seed)
    w.reset()
    d = w.ducks["d0"]
    brain = attach_world(SkateRoute(plan=fp, mode=mode), w, "d0")
    scen, mine = _static_geoms(w.model), _duck_geoms(w.model, d.root_body)
    keep = np.asarray(fp.keepout()) if fp.keepout() else np.zeros((0, 4))

    def to_stairs(x: float, y: float) -> float:
        if not len(keep):
            return float("inf")
        dx = np.maximum(np.maximum(keep[:, 0] - x, x - keep[:, 2]), 0.0)
        dy = np.maximum(np.maximum(keep[:, 1] - y, y - keep[:, 3]), 0.0)
        return float(np.hypot(dx, dy).min())

    here, rows = start, []
    for _ in range(trips):
        goal = str(rng.choice([n for n in names if n != here]))
        brain.goto(goal)
        t0, f0, hits, touching, stairs, edge = w.t, d.falls, 0, False, False, float("inf")
        while w.t - t0 < seconds:
            d.set_cmd(w.data, brain.step(Senses(t=w.t, speed=d.heading_speed(w.data))).twist)
            w.step()
            now = any((c.geom1 in mine and c.geom2 in scen) or (c.geom2 in mine and c.geom1 in scen)
                      for c in w.data.contact[:w.data.ncon])
            hits += int(now and not touching)
            touching = now
            x, y = map(float, d.trunk_pos(w.data)[:2])
            stairs |= fp.on_stairs(x, y)
            edge = min(edge, to_stairs(x, y))
            if brain.state in ("done", "no_path"):
                break
        x, y = map(float, d.trunk_pos(w.data)[:2])
        at = fp.area_at(x, y)
        row = dict(mode=mode, seed=seed, frm=here, to=goal, ok=brain.state == "done" and at == goal,
                   time=round(w.t - t0, 1), hits=hits, stairs=stairs, falls=d.falls - f0,
                   edge_m=round(edge, 2), state=brain.state, at=at, replans=brain.replans)
        rows.append(row)
        if verbose:
            print("  ", row, flush=True)
        here = at or here
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan", help="name under private/ or a path to plan.json")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--trips", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=150.0, help="cap per trip")
    ap.add_argument("--mode", choices=("roller", "walk", "both"), default="both")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    fp = FloorPlan.load(plan_path(a.plan))
    modes = ("roller", "walk") if a.mode == "both" else (a.mode,)
    rows = []
    for mode in modes:
        for s in range(a.trials):
            t0 = time.time()
            r = trial(fp, s, a.trips, a.seconds, a.verbose, mode)
            rows += r
            print(f"{mode:6s} trial {s}: {sum(x['ok'] for x in r)}/{len(r)} trips ok, hits {sum(x['hits'] for x in r)}, "
                  f"stairs {sum(x['stairs'] for x in r)}, falls {sum(x['falls'] for x in r)} "
                  f"[{time.time() - t0:.0f}s wall]", flush=True)
    print()
    for mode in modes:
        m = [x for x in rows if x["mode"] == mode]
        ok = [x for x in m if x["ok"]]
        print(f"{a.plan} {mode:6s}: {len(ok)}/{len(m)} trips arrived; hits {sum(x['hits'] for x in m)}, "
              f"stairs entries {sum(x['stairs'] for x in m)}, falls {sum(x['falls'] for x in m)}; "
              f"median trip {float(np.median([x['time'] for x in ok])) if ok else float('nan'):.0f} s, "
              f"closest to the stairs {min(x['edge_m'] for x in m):.2f} m")
    for x in rows:
        if not x["ok"] or x["hits"] or x["stairs"] or x["falls"]:
            print("  FAIL/flag:", x)


if __name__ == "__main__":
    main()
