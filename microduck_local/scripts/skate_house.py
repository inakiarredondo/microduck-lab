"""Run the skating house tour headless and score it.

    uv run python scripts/skate_house.py                       # casa-demo, 3 seeds
    uv run python scripts/skate_house.py --house casa-demo --seeds 5 --seconds 240

One duck on roller skates (`pollen:roller`), driven by the `skate_route` brain
(`brain/skate.py`, PRIVILEGED: true pose, true map), through the house's tour
(`world/house.HOUSES`). Per seed it prints:

    done      the tour reached its last goal
    time      seconds to get there (or the cap)
    route_ok  the rooms the trunk was IN, in order, are the tour's (an
              independent check on `goals`, read off the true pose)
    hits      separate contact EPISODES between any duck geom and a wall or a
              piece of furniture (a scrape that lasts 1 s is one hit)
    contact   total seconds in such contact
    falls     the world's own fall count
    goals     goals reached / total

A hit is read off MuJoCo's contact list, not a clearance threshold.
"""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import numpy as np

from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.runtime import Senses, attach_world
from microduck_local.brain.skate import SkateRoute
from microduck_local.world import World
from microduck_local.world.house import HOUSES


def scenery_geoms(model: mujoco.MjModel) -> set[int]:
    out = set()
    for g in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n.startswith("wall") or (n.startswith("box") and n.endswith("_geom")):
            out.add(g)
    return out


def duck_geoms(model: mujoco.MjModel, root_body: int) -> set[int]:
    bodies = {b for b in range(model.nbody) if _under(model, b, root_body)}
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}


def _under(model, b, root) -> bool:
    while b > 0:
        if b == root:
            return True
        b = model.body_parentid[b]
    return False


def run(house: str, seed: int, seconds: float, verbose: bool = False) -> dict:
    plan, route = HOUSES[house]()
    # The world is deterministic, so a seed must change something: the spawn
    # heading (any) and a small offset from the room's most open point.
    rng = np.random.default_rng(seed)
    sc = plan.scenario(duck_spawn=(route[0], float(rng.uniform(-math.pi, math.pi))))
    x, y, yaw = sc.ducks[0].spawn
    sc.ducks[0].spawn = (x + float(rng.uniform(-0.15, 0.15)), y + float(rng.uniform(-0.15, 0.15)), yaw)
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "roller.onnx")}, seed=seed)
    w.reset()
    d = w.ducks["d0"]
    brain = attach_world(SkateRoute(goals=plan.route_waypoints(route)), w, "d0")
    scen, mine = scenery_geoms(w.model), duck_geoms(w.model, d.root_body)
    n_goals = len(plan.route_waypoints(route))
    hits, contact_steps, touching, t_done = 0, 0, False, None
    path_len, last = 0.0, d.trunk_pos(w.data)[:2].copy()
    visits: list[str] = []           # areas the trunk was actually in, in order
    for k in range(int(seconds * 50)):
        intent = brain.step(Senses(t=w.t, speed=d.heading_speed(w.data)))
        d.set_cmd(w.data, intent.twist)
        w.step()
        now = False
        for c in w.data.contact[:w.data.ncon]:
            if (c.geom1 in mine and c.geom2 in scen) or (c.geom2 in mine and c.geom1 in scen):
                now = True
                break
        if now:
            contact_steps += 1
            if not touching:
                hits += 1
                if verbose:
                    print(f"  hit at t={w.t:.1f}s near {brain.inputs().get('next')}")
        touching = now
        p = d.trunk_pos(w.data)[:2]
        area = plan.area_at(float(p[0]), float(p[1]))
        if area is not None and (not visits or visits[-1] != area.name):
            visits.append(area.name)
        path_len += float(np.linalg.norm(p - last))
        last = p.copy()
        if brain.state == "done":
            t_done = w.t
            break
        if verbose and k % 500 == 0:
            print(f"  t={w.t:5.1f} state={brain.state:10s} {brain.inputs()}")
    return dict(seed=seed, done=t_done is not None, time=round(t_done or w.t, 1),
                hits=hits, contact_s=round(contact_steps / 50, 2), falls=d.falls,
                goals=f"{len(brain.reached)}/{n_goals}", replans=brain.replans,
                path_m=round(path_len, 1), state=brain.state,
                route_ok=_follows(visits, route), visited=">".join(visits))


def _follows(visits: list[str], route: list[str]) -> bool:
    """The trunk's own room sequence IS the tour — strict: a doorway straddle
    that flickers between two rooms counts against it."""
    return visits == route


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--house", default="casa-demo", choices=sorted(HOUSES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    rows = []
    for s in range(a.seeds):
        t0 = time.time()
        r = run(a.house, s, a.seconds, a.verbose)
        rows.append(r)
        print(r, f"[{time.time() - t0:.0f}s wall]", flush=True)
    done = sum(r["done"] for r in rows)
    ok = sum(r["route_ok"] for r in rows)
    print(f"\n{a.house}: {done}/{len(rows)} tours complete ({ok} along the planned room sequence), hits {sum(r['hits'] for r in rows)}, "
          f"falls {sum(r['falls'] for r in rows)}, median time "
          f"{float(np.median([r['time'] for r in rows])):.0f} s")


if __name__ == "__main__":
    main()
