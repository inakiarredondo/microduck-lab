"""Locks for the skating house tour: the plan compiler (world/house.py), the
roller-skate duck in a world (`collision="rollers"`) and the privileged
route brain (brain/skate.py)."""

import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.runtime import REGISTRY, Senses, attach_world
from microduck_local.brain.skate import OccupancyGrid, SkateParams, SkateRoute
from microduck_local.world import Duck, Scenario, World, compose, validate_scenario
from microduck_local.world.compose import ROBOT_XML, ROLLER_WHEEL_FRICTIONLOSS, DuckAddress
from microduck_local.world.house import (
    DEMO_ROUTE,
    Door,
    Furniture,
    Garden,
    HousePlan,
    Room,
    demo_plan,
)
from microduck_local.world.scenario import ScenarioError

needs_rl = pytest.mark.skipif(not ROBOT_XML["rollers"].exists(), reason="microduck_rl checkout not found")
needs_roller = pytest.mark.skipif(not (POLICIES_DIR / "roller.onnx").exists(),
                                  reason="upstream policies not checked out")


def _two_rooms() -> HousePlan:
    return HousePlan("two", rooms=[Room("a", (0, 0), (3, 2)), Room("b", (3, 0), (5, 2))],
                     doors=[Door((3, 1), 0.8)],
                     gardens=[Garden("g", (0, -1), (5, 0))], exits=[Door((1, 0), 0.9)])


# -- the plan compiler ---------------------------------------------------------

def test_a_shared_wall_is_drawn_once_and_the_door_is_cut_out_of_it():
    walls = _two_rooms().walls()
    on_x3 = [w for w in walls if w.start[0] == 3 and w.end[0] == 3]
    # one wall at x=3 from y=0 to 2, minus a 0.8 m door centred at y=1
    spans = sorted((min(w.start[1], w.end[1]), max(w.start[1], w.end[1])) for w in on_x3)
    assert spans == [pytest.approx((0.0, 0.6)), pytest.approx((1.4, 2.0))]


def test_the_exit_is_cut_and_the_garden_fence_skips_the_house_wall():
    walls = _two_rooms().walls()
    y0 = sorted((min(w.start[0], w.end[0]), max(w.start[0], w.end[0]), w.height)
                for w in walls if w.start[1] == 0 and w.end[1] == 0)
    # the house's south wall, full height, with the 0.9 m exit at x=1; no fence on top of it
    assert [(a, b) for a, b, _ in y0] == [pytest.approx((0.0, 0.55)), pytest.approx((1.45, 3.0)),
                                          pytest.approx((3.0, 5.0))]
    assert all(h > 0.5 for *_, h in y0)
    fence = [w for w in walls if w.start[1] == -1 and w.end[1] == -1]
    assert fence and all(w.height < 0.5 for w in fence)


def test_route_waypoints_cross_each_door_square_on():
    plan = _two_rooms()
    wps = dict(plan.route_waypoints(["a", "b", "a", "g"]))
    assert wps["1:a>b:door"] == (3, 1)
    assert wps["1:a>b:approach"][0] < 3 < wps["1:a>b:through"][0]
    assert wps["2:b>a:approach"][0] > 3 > wps["2:b>a:through"][0]
    assert wps["3:a>g:through"][1] < 0          # out into the garden
    with pytest.raises(KeyError):
        plan.door_between("b", "g")              # no door from b to the garden


def test_open_point_avoids_the_furniture_at_the_centre():
    plan = HousePlan(rooms=[Room("r", (0, 0), (4, 4))],
                     furniture=[Furniture("table", (2, 2), (1.0, 1.0))])
    x, y = plan.open_point("r")
    assert max(abs(x - 2), abs(y - 2)) > 0.5


def test_a_plan_must_be_centred_before_it_becomes_a_world():
    with pytest.raises(ValueError):
        _two_rooms().scenario()
    sc = _two_rooms().centred().scenario(duck_spawn=("a", 0.0))
    assert sc.collision == "rollers" and sc.ducks[0].policy == "pollen:roller"


def test_rollers_is_a_valid_scenario_collision_and_nothing_else_new_is():
    raw = {"version": 1, "name": "r", "collision": "rollers"}
    assert validate_scenario(raw).collision == "rollers"
    with pytest.raises(ScenarioError):
        validate_scenario({**raw, "collision": "skates"})


# -- the planner -----------------------------------------------------------------

def test_the_planner_never_crosses_an_inflated_obstacle():
    plan = demo_plan()
    sc = plan.scenario(duck_spawn=(DEMO_ROUTE[0], 0.0))
    p = SkateParams()
    grid = OccupancyGrid.from_scenario(sc, p)
    goals = plan.route_waypoints(DEMO_ROUTE)
    cur = np.asarray(goals[0][1], float)
    for _, g in goals[1:]:
        seg = grid.astar(cur, np.asarray(g, float))
        assert seg is not None
        for a, b in zip(seg, seg[1:]):
            for t in np.linspace(0, 1, 20):
                q = a + (b - a) * t
                assert not grid.blocked[grid.cell(q)]
        cur = np.asarray(g, float)


def test_the_brain_is_registered():
    assert "skate_route" in REGISTRY.available()
    assert REGISTRY.make("skate_route").kind == "skate_route"


# -- the duck on skates ----------------------------------------------------------

@needs_rl
def test_a_roller_duck_has_the_contract_joints_and_free_wheels():
    m = compose(Scenario(name="r", collision="rollers", ducks=[Duck("d0", (0, 0, 0))]))
    adr = DuckAddress.resolve(m, "d0")
    assert len(adr.joint_qpos) == C.NUM_JOINTS
    wheels = [j for j in range(m.njnt)
              if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("d0/passive_")]
    assert len(wheels) == 4
    assert all(m.dof_frictionloss[m.jnt_dofadr[j]] == ROLLER_WHEEL_FRICTIONLOSS for j in wheels)
    # the wheels are not in the contract and have no actuator
    assert not set(m.jnt_qposadr[wheels]) & set(adr.joint_qpos)
    assert m.nu == C.NUM_JOINTS + 1               # 14 servos + the mouth


@needs_rl
@needs_roller
def test_the_shipped_roller_policy_skates_and_steers_in_a_world():
    sc = Scenario(name="r", floor=(8.0, 8.0), collision="rollers", ducks=[Duck("d0", (-3.0, -3.0, 0.0))])
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "roller.onnx")}, seed=0)
    w.reset()
    d = w.ducks["d0"]
    start = d.trunk_pos(w.data)[:2].copy()
    for _ in range(4 * 50):                        # 4 s: push, and hold +90 deg
        err = (math.pi / 2 - d.yaw(w.data) + math.pi) % (2 * math.pi) - math.pi
        d.set_cmd(w.data, (0.6, 0.0, float(np.clip(err, -1, 1)) or 1e-4))
        w.step()
    assert d.falls == 0
    assert np.linalg.norm(d.trunk_pos(w.data)[:2] - start) > 0.5
    assert abs(math.degrees(d.yaw(w.data)) - 90) < 15


@needs_rl
@needs_roller
def test_the_route_brain_tours_the_demo_house_without_touching_anything():
    plan = demo_plan()
    sc = plan.scenario(duck_spawn=(DEMO_ROUTE[0], 0.0))
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "roller.onnx")}, seed=0)
    w.reset()
    d = w.ducks["d0"]
    brain = attach_world(SkateRoute(goals=plan.route_waypoints(DEMO_ROUTE)), w, "d0")
    scen = {g for g in range(w.model.ngeom)
            if (mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("wall", "box"))}
    root = d.root_body
    mine = set()
    for g in range(w.model.ngeom):
        b = w.model.geom_bodyid[g]
        while b > 0 and b != root:
            b = w.model.body_parentid[b]
        if b == root:
            mine.add(g)
    rooms, touched = [], False
    for _ in range(120 * 50):
        d.set_cmd(w.data, brain.step(Senses(t=w.t, speed=d.heading_speed(w.data))).twist)
        w.step()
        touched |= any((c.geom1 in mine and c.geom2 in scen) or (c.geom2 in mine and c.geom1 in scen)
                       for c in w.data.contact[:w.data.ncon])
        a = plan.area_at(*map(float, d.trunk_pos(w.data)[:2]))
        if a is not None and (not rooms or rooms[-1] != a.name):
            rooms.append(a.name)
        if brain.state == "done":
            break
    assert brain.state == "done"
    assert rooms == DEMO_ROUTE
    assert d.falls == 0 and not touched
