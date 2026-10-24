"""Locks for `world/floorplan.py` (a colour-map floor plan as a world) and the
skate brain's `goto` on it. Every plan here is SYNTHETIC, drawn in the test:
a real home's plan is private and never enters the repository."""

import math

import numpy as np
import pytest

from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.runtime import Senses, attach_world
from microduck_local.brain.skate import OccupancyGrid, SkateParams, SkateRoute
from microduck_local.world import World
from microduck_local.world.compose import ROBOT_XML
from microduck_local.world.floorplan import FLOOR, OBSTACLE, PRIVATE_DIR, STAIRS, FloorPlan

BLUE, YELLOW, GREY, DOOR = (80, 120, 176), (176, 128, 40), (64, 64, 64), (56, 122, 134)


def _plan(stairs: bool = True, railed: bool = False, inner: bool = False) -> FloorPlan:
    """Two rooms side by side, 1 px = 1 cm: a blue room 300 x 200, a yellow
    room 200 x 200, a grey wall between them with an 80 cm door marked in
    the door colour, and (optionally) a blacked-out stairwell in the blue
    room's top-left corner behind a thin grey edge line. `inner` adds a second
    stairwell INSIDE the blue room, astride the straight line from the
    room's open point to the door — what a planner must swing round."""
    im = np.full((240, 560, 3), 16, np.uint8)
    im[20:220, 20:320] = BLUE
    im[20:220, 330:530] = YELLOW
    im[20:220, 320:330] = GREY
    im[80:160, 320:330] = DOOR                       # door: rows 80..160 = 80 cm
    if stairs:
        im[0:20, 20:120] = 0                         # the stairwell, beyond the room's top edge
        im[20:23, 20:120] = GREY                     # the vacuum's edge line
    if inner:
        im[70:130, 180:240] = 0
    spec = {"scale": {"from_px": [0, 0], "to_px": [100, 0], "cm": 100},
            "rooms": [{"name": "azul", "rgb": list(BLUE)}, {"name": "amarillo", "rgb": list(YELLOW)}],
            "floor_rgb": [list(DOOR)],
            "stairs": {"dark_max": 12, "railed": railed} if stairs else {}}
    return FloorPlan.from_pixels("synthetic", im, spec)


def test_rooms_doors_and_scale_read_off_the_colours():
    fp = _plan()
    assert fp.size == pytest.approx((5.6, 2.4))
    assert set(fp.rooms) == {"azul", "amarillo"}
    assert fp.rooms["azul"].sum() * 0.0025 == pytest.approx(6.0, rel=0.05)
    # the door is floor and the wall around it is not
    door_i, door_j = fp.ij(*fp.xy(24, 65))           # row 120 px / 5, col 325 px / 5
    assert fp.labels[door_i, door_j] == FLOOR
    assert fp.labels[fp.ij(*fp.xy(6, 65))] == OBSTACLE
    assert fp.area_at(*fp.open_point("amarillo")) == "amarillo"


def test_an_unrailed_stair_edge_is_open_and_a_railed_one_is_a_wall():
    open_ = _plan(railed=False)
    railed = _plan(railed=True)
    edge = open_.ij(*open_.xy(4, 10))                # the grey line, row 20-23 px
    assert open_.labels[edge] == STAIRS              # no wall: rolling over it is a fall
    assert railed.labels[edge] == OBSTACLE
    assert open_.keepout()

    def walled(fp, x, y):
        return any(abs(x - b.pos[0]) <= b.size[0] / 2 and abs(y - b.pos[1]) <= b.size[1] / 2 for b in fp.boxes())
    x, y = open_.xy(*edge)
    assert not walled(open_, x, y) and walled(railed, x, y)


def test_boxes_cover_only_the_obstacles_a_duck_can_reach():
    fp = _plan(stairs=False)
    boxes = fp.boxes()
    assert 0 < len(boxes) < 60
    # no box overlaps a floor cell
    for b in boxes:
        x0, x1 = b.pos[0] - b.size[0] / 2 + 1e-6, b.pos[0] + b.size[0] / 2 - 1e-6
        y0, y1 = b.pos[1] - b.size[1] / 2 + 1e-6, b.pos[1] + b.size[1] / 2 - 1e-6
        for x in np.linspace(x0, x1, 5):
            for y in np.linspace(y0, y1, 5):
                assert fp.labels[fp.ij(x, y)] != FLOOR


def test_the_planner_keeps_off_the_stairs_and_goes_through_the_door():
    fp = _plan(inner=True)
    sc = fp.scenario(spawn_room="azul")
    p = SkateParams()
    g = OccupancyGrid.from_scenario(sc, p, keepout=fp.keepout())
    path = g.astar(np.array(fp.open_point("azul")), np.array(fp.open_point("amarillo")))
    assert path is not None
    keep = np.asarray(fp.keepout())
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0, 1, 10):
            x, y = a + (b - a) * t
            dx = np.maximum(np.maximum(keep[:, 0] - x, x - keep[:, 2]), 0)
            dy = np.maximum(np.maximum(keep[:, 1] - y, y - keep[:, 3]), 0)
            assert np.hypot(dx, dy).min() >= p.clearance + p.keepout_margin - 0.05
    # it crosses x = the wall only inside the door's span
    xw = fp.xy(0, 65)[0]
    cross = [a + (b - a) * ((xw - a[0]) / (b[0] - a[0])) for a, b in zip(path, path[1:])
             if (a[0] - xw) * (b[0] - xw) < 0]
    door_lo, door_hi = sorted((fp.xy(16, 0)[1], fp.xy(32, 0)[1]))
    assert cross and all(door_lo <= c[1] <= door_hi for c in cross)


def test_the_private_directory_is_ignored_by_git():
    import subprocess
    root = PRIVATE_DIR.parents[1]
    r = subprocess.run(["git", "check-ignore", "-q", str(PRIVATE_DIR / "x" / "plan.json")], cwd=root)
    assert r.returncode == 0, "private/ must be git-ignored: a home's plan never enters the repo"


@pytest.mark.skipif(not ROBOT_XML["rollers"].exists() or not (POLICIES_DIR / "roller.onnx").exists(),
                    reason="microduck_rl checkout or upstream policies not found")
def test_goto_reaches_the_other_room_without_touching_anything():
    import mujoco
    fp = _plan()
    w = World(fp.scenario(spawn_room="azul", yaw=math.pi),
              infer_for={"d0": onnx_infer(POLICIES_DIR / "roller.onnx")}, seed=0)
    w.reset()
    d = w.ducks["d0"]
    brain = attach_world(SkateRoute(plan=fp), w, "d0")
    static = {g for g in range(w.model.ngeom)
              if (mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("box")}
    for room in ("amarillo", "azul"):
        brain.goto(room)
        touched = stairs = False
        for _ in range(60 * 50):
            d.set_cmd(w.data, brain.step(Senses(t=w.t, speed=d.heading_speed(w.data))).twist)
            w.step()
            touched |= any((w.model.geom_bodyid[c.geom1] > 0 and c.geom2 in static) or
                           (w.model.geom_bodyid[c.geom2] > 0 and c.geom1 in static)
                           for c in w.data.contact[:w.data.ncon]
                           if not (c.geom1 in static and c.geom2 in static))
            stairs |= fp.on_stairs(*map(float, d.trunk_pos(w.data)[:2]))
            if brain.state == "done":
                break
        assert brain.state == "done", room
        assert fp.area_at(*map(float, d.trunk_pos(w.data)[:2])) == room
        assert not touched and not stairs and d.falls == 0
