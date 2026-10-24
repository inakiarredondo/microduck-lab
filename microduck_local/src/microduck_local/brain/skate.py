"""`skate_route`: tour a house on roller skates, room by room, without hitting anything.

The reflex tier is the shipped `roller.onnx` (`Scenario.collision ==
"rollers"`), whose command is NOT a walker's twist:

    vx   0 = coast, > 0 = push (trained to 0.6), < 0 = brake
    wz   HEADING ERROR in radians, clipped to +-1 (not a yaw rate)
    vy   unused (trained at 0)

Measured before this was written (scratchpad probes, the lab's XML-PD world
and upstream's BAM rehearsal): the policy holds any commanded heading within
~5 deg on flat floor, 0 falls in 25 runs; coasting stops it within ~1 cm;
-0.5 does not brake, it skates BACKWARDS, so this brain never asks for less
than `BRAKE`. A sill as low as 5 mm stops the blades dead, so the plan has
no thresholds between rooms.

THIS IS A PRIVILEGED BRAIN (the /sim "control arm", `truth=True` in
`brain/runtime.attach_world`'s terms): it reads the duck's TRUE pose and the
room's TRUE geometry off the world. It is the simulation demo, and the
baseline a sensed brain has to match; it cannot run on a robot as written,
because the robot has neither a map nor a drift-free pose.

How it drives:
  1. Rasterise the scenario's walls and static boxes into an occupancy grid
     and inflate it by the duck's footprint (`clearance`), plus a soft cost
     that keeps the path in the middle of corridors and doorways.
  2. A* through the tour's waypoints (`world/house.HousePlan.route_waypoints`:
     each door is approached square-on, crossed, and left square-on).
  3. Pure pursuit on that path: heading error to a point `lookahead` ahead,
     PUSH while that error is small and the next corner is far, COAST
     otherwise. The skates shed speed fast enough that coasting is the brake.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .runtime import Intent, Senses

PUSH = 0.6          # the trained top of the push range
BRAKE = -0.2        # the most negative command used; -0.5 skates backwards
# ON FEET (`mode="walk"`, the shipped walker): the command is a real twist —
# vx in m/s, wz a yaw RATE in rad/s — so the same planner drives it with a
# proportional turn and a forward speed inside the walker's trained range.
WALK_VX = 0.30      # forward, m/s (alpha_walking's trained |vx| <= 0.3)
WALK_SLOW = 0.15    # forward when the goal is close, m/s
WALK_TURN_GAIN = 2.0  # wz = gain * heading error, clipped to +-WALK_WZ_MAX
# The yaw-rate bound on feet. NOT the walker's trained 1.0: measured in the
# lab's world (6 s, flat floor), alpha_walking turning IN PLACE answers
# wz -1.0 with -0.04 rad/s (a dead zone, right turns only) and +1.0 with
# +0.58, while +-2.0 gives -1.52 / ~+1.5 at every vx from 0 to 0.2, 0 falls
# — the same widening brain_env's BRAIN_WZ notes. At +-1 a duck facing away
# from its goal span in place for minutes.
WALK_WZ_MAX = 2.0
#: The two drives this brain knows, and the duck each one needs.
MODES = {"roller": {"collision": "rollers", "policy": "pollen:roller"},
         "walk": {"collision": "all", "policy": "pollen:alpha_walking"}}


@dataclass
class SkateParams:
    resolution: float = 0.05     # occupancy grid cell, m
    clearance: float = 0.20      # duck footprint radius + margin, m (blade tip to trunk ~0.1 m)
    soft: float = 0.35           # within this of an obstacle the path pays extra
    lookahead: float = 0.30      # pure-pursuit carrot distance, m
    push_err: float = 0.30       # push only while |heading error| < this, rad
    corner_turn: float = 0.6     # a path bend sharper than this (rad) ahead …
    corner_dist: float = 0.35    # … within this distance makes it coast early
    corner_speed: float = 0.12   # … down to this speed, m/s
    walk_turn_err: float = 0.35  # on feet: turn in place while |heading error| exceeds this, rad
    goal_tol: float = 0.20       # final goal reached, m
    replan_dev: float = 0.35     # off the path by more than this: replan from here
    keepout_margin: float = 0.15 # extra berth for a keep-out zone (a stairwell has no wall to bump);
                                 # measured on a real plan: 0.20 closed a narrow passage


class SkateRoute:
    """Brain kind `skate_route` (see module doc)."""

    kind = "skate_route"

    def __init__(self, goals: list[tuple[str, tuple[float, float]]] | None = None,
                 params: SkateParams | None = None, house: str | None = None,
                 plan=None, mode: str = "roller") -> None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}; have {sorted(MODES)}")
        self.mode = mode             # "roller" (skates) or "walk" (feet): what the command MEANS
        self.p = params or SkateParams()
        self.goals_given = goals
        self.house = house
        # A `world/floorplan.FloorPlan`: rooms by name for `goto`, and its
        # stairs as keep-out zones the planner gives a wide berth.
        self.plan = plan
        self.world = None            # attach_world fills these
        self.robot_id: str | None = None
        self.reset()

    # -- Brain protocol -------------------------------------------------------
    def reset(self) -> None:
        self.state = "plan"
        self.path: np.ndarray | None = None      # (N, 2) world xy
        self.labels: list[tuple[int, str]] = []  # (path index, waypoint name)
        self.i = 0                               # progress index on the path
        self.goal_i = 0
        self.reached: list[str] = []
        self.replans = 0
        self._grid = None
        self._last = {}

    def inputs(self) -> dict:
        return dict(self._last)

    def goto(self, room: str) -> None:
        """Go to `room` from wherever the duck is now ("is anyone in the
        kitchen?"). Aims at the room's most open point; replans from scratch."""
        if self.plan is None:
            raise ValueError("goto needs a plan (SkateRoute(plan=FloorPlan...))")
        self.goals_given = [(room, self.plan.open_point(room))]
        grid = self._grid
        self.reset()
        self._grid = grid            # the map did not change

    def step(self, senses: Senses) -> Intent:
        if self.world is None or self.robot_id is None:
            self.state = "blind"
            return Intent(note="skate_route needs the world (privileged brain)")
        d = self.world.ducks[self.robot_id]
        xy = np.asarray(d.trunk_pos(self.world.data)[:2], float)
        yaw = d.yaw(self.world.data)
        if self._grid is None:
            self._grid = OccupancyGrid.from_scenario(
                self.world.scenario, self.p, keepout=self.plan.keepout() if self.plan is not None else ())
        if self.path is None:
            self._plan(xy)
            if self.path is None:
                self.state = "no_path"
                return Intent(note="no collision-free path to the next room")
        if self.state == "done":
            return Intent(twist=(0.0, 0.0, 0.0), note="arrived")

        # progress: nearest path point in a window ahead of where we were
        hi = min(len(self.path), self.i + 40)
        k = self.i + int(np.argmin(np.linalg.norm(self.path[self.i:hi] - xy, axis=1)))
        dev = float(np.linalg.norm(self.path[k] - xy))
        if dev > self.p.replan_dev:
            self.replans += 1
            self._plan(xy)
            k = 0
        self.i = k
        while self.labels and self.labels[0][0] <= self.i + 2:
            self.reached.append(self.labels.pop(0)[1])
        end = self.path[-1]
        if float(np.linalg.norm(end - xy)) < self.p.goal_tol:
            self.state = "done"
            self.reached.extend(name for _, name in self.labels)
            self.labels = []
            return Intent(twist=(0.0, 0.0, 0.0), note="arrived")

        carrot = self._ahead(self.i, self.p.lookahead)
        err = _wrap(math.atan2(carrot[1] - xy[1], carrot[0] - xy[0]) - yaw)
        bend = self._bend_ahead(self.i, self.p.corner_dist)
        to_end = float(np.linalg.norm(end - xy))
        speed = abs(senses.speed) if senses.speed is not None else 0.0
        if self.mode == "walk":
            if abs(err) > self.p.walk_turn_err:
                vx, self.state = 0.0, "turn"
            elif to_end < 0.5:
                vx, self.state = WALK_SLOW, "approach"
            else:
                vx, self.state = WALK_VX, "walk"
            wz = float(np.clip(WALK_TURN_GAIN * err, -WALK_WZ_MAX, WALK_WZ_MAX))
        elif abs(err) > self.p.push_err:
            vx, self.state = 0.0, "coast_turn"        # turn first: the blades steer at zero push
            wz = float(np.clip(err, -1.0, 1.0))
        elif (bend > self.p.corner_turn or to_end < 0.5) and speed > self.p.corner_speed:
            # a corner or the end is close and we are quick: glide down to the
            # corner speed. Only while QUICK — coasting toward a corner from a
            # standstill never reaches it.
            vx, self.state = 0.0, "glide"
            wz = float(np.clip(err, -1.0, 1.0))
        else:
            vx, self.state = PUSH, "push"
            wz = float(np.clip(err, -1.0, 1.0))
        # `WorldDuck.set_cmd` treats an EXACT zero turn with forward push as
        # "hold the current heading" and substitutes its own walker-style
        # yaw-rate term; on skates wz is a heading error, so never send 0.
        if wz == 0.0:
            wz = 1e-4
        nxt = self.labels[0][1] if self.labels else "end"
        self._last = {"next": nxt, "err_deg": round(math.degrees(err), 1), "bend_deg": round(math.degrees(bend), 1),
                      "dev_m": round(dev, 3), "replans": self.replans}
        return Intent(twist=(vx, 0.0, wz), note=f"{self.state} -> {nxt}")

    # -- planning -------------------------------------------------------------
    def _goals(self) -> list[tuple[str, tuple[float, float]]]:
        if self.goals_given is not None:
            return list(self.goals_given)
        from ..world.house import HOUSES
        name = self.house or self.world.scenario.name
        plan, route = HOUSES[name]()
        return plan.route_waypoints(route)

    def _plan(self, xy: np.ndarray) -> None:
        goals = self._goals()
        # drop the goals already reached (a replan starts from where we are)
        goals = [g for g in goals if g[0] not in self.reached]
        pts, labels, cur = [xy.copy()], [], xy
        for name, g in goals:
            seg = self._grid.astar(cur, np.asarray(g, float))
            if seg is None:
                self.path = None
                return
            pts.extend(seg[1:])
            labels.append((len(pts) - 1, name))
            cur = np.asarray(g, float)
        self.path, where = _densify(np.asarray(pts), self.p.resolution)
        # carry each label to where ITS point landed on the densified path —
        # not to the nearest path point, which for a doorway crossed twice is
        # the first crossing
        self.labels = [(where[i], n) for i, n in labels]
        self.i = 0

    def _ahead(self, i: int, dist: float) -> np.ndarray:
        acc = 0.0
        for j in range(i, len(self.path) - 1):
            acc += float(np.linalg.norm(self.path[j + 1] - self.path[j]))
            if acc >= dist:
                return self.path[j + 1]
        return self.path[-1]

    def _bend_ahead(self, i: int, dist: float) -> float:
        """Largest heading change of the path between `i` and `dist` + lookahead ahead."""
        a = self._ahead(i, 0.0 + 1e-6)
        b = self._ahead(i, self.p.lookahead)
        c = self._ahead(i, self.p.lookahead + dist)
        h1 = math.atan2(b[1] - a[1], b[0] - a[0])
        h2 = math.atan2(c[1] - b[1], c[0] - b[0])
        return abs(_wrap(h2 - h1)) if np.linalg.norm(c - b) > 1e-6 and np.linalg.norm(b - a) > 1e-6 else 0.0


@dataclass
class OccupancyGrid:
    origin: np.ndarray           # world xy of cell (0, 0)
    res: float
    blocked: np.ndarray          # (H, W) bool, inflated by clearance
    cost: np.ndarray             # (H, W) float, >= 1

    @classmethod
    def from_scenario(cls, sc, p: SkateParams, keepout=()) -> "OccupancyGrid":
        """`keepout`: (x0, y0, x1, y1) world rectangles with no geometry that
        must still never be entered (stairs), kept `clearance + keepout_margin`
        away."""
        res = p.resolution
        hx, hy = sc.floor[0] / 2, sc.floor[1] / 2
        W, H = int(math.ceil(2 * hx / res)), int(math.ceil(2 * hy / res))
        origin = np.array([-hx, -hy])
        xs = origin[0] + (np.arange(W) + 0.5) * res
        ys = origin[1] + (np.arange(H) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        # signed-ish distance to the nearest obstacle (0 inside), from every
        # wall and static box as an oriented rectangle
        dist = np.full((H, W), np.inf)
        rects = []
        for w in sc.walls:
            (x0, y0), (x1, y1) = w.start, w.end
            rects.append(((x0 + x1) / 2, (y0 + y1) / 2, math.dist(w.start, w.end) / 2, w.thickness / 2,
                          math.atan2(y1 - y0, x1 - x0)))
        for b in sc.boxes:
            if b.mass == 0.0:
                rects.append((b.pos[0], b.pos[1], b.size[0] / 2, b.size[1] / 2, b.yaw))
        for cx, cy, ax, ay, th in rects:
            c, s = math.cos(th), math.sin(th)
            u = (X - cx) * c + (Y - cy) * s
            v = -(X - cx) * s + (Y - cy) * c
            dx = np.maximum(np.abs(u) - ax, 0.0)
            dy = np.maximum(np.abs(v) - ay, 0.0)
            dist = np.minimum(dist, np.hypot(dx, dy))
        # the floor's own edge is an obstacle too
        edge = np.minimum.reduce([X - origin[0], -origin[0] - X, Y - origin[1], -origin[1] - Y])
        dist = np.minimum(dist, np.maximum(edge, 0.0))
        blocked = dist < p.clearance
        for x0, y0, x1, y1 in keepout:
            dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
            kd = np.hypot(dx, dy)
            blocked |= kd < p.clearance + p.keepout_margin
            dist = np.minimum(dist, np.maximum(kd - p.keepout_margin, 0.0))
        cost = 1.0 + 4.0 * np.clip((p.clearance + p.soft - dist) / p.soft, 0.0, 1.0) ** 2
        return cls(origin, res, blocked, cost)

    def cell(self, xy) -> tuple[int, int]:
        j = int((xy[0] - self.origin[0]) / self.res)
        i = int((xy[1] - self.origin[1]) / self.res)
        H, W = self.blocked.shape
        return min(max(i, 0), H - 1), min(max(j, 0), W - 1)

    def centre(self, ij) -> np.ndarray:
        return self.origin + (np.array([ij[1], ij[0]]) + 0.5) * self.res

    def _free_near(self, ij) -> tuple[int, int]:
        """The cell itself, or the nearest unblocked cell (a start pressed
        against a wall must still be able to leave it)."""
        if not self.blocked[ij]:
            return ij
        free = np.argwhere(~self.blocked)
        k = int(np.argmin(np.sum((free - np.array(ij)) ** 2, axis=1)))
        return tuple(int(v) for v in free[k])

    def astar(self, a, b) -> np.ndarray | None:
        s, g = self._free_near(self.cell(a)), self._free_near(self.cell(b))
        H, W = self.blocked.shape
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        gs = {s: 0.0}
        came = {}
        openq = [(0.0, s)]
        while openq:
            _, u = heapq.heappop(openq)
            if u == g:
                out = [u]
                while out[-1] in came:
                    out.append(came[out[-1]])
                out.reverse()
                pts = np.array([self.centre(c) for c in out])
                pts[0], pts[-1] = np.asarray(a, float), np.asarray(b, float)
                return _shortcut(pts, self)
            for di, dj, w in nbrs:
                v = (u[0] + di, u[1] + dj)
                if not (0 <= v[0] < H and 0 <= v[1] < W) or self.blocked[v]:
                    continue
                ng = gs[u] + w * self.res * self.cost[v]
                if ng < gs.get(v, math.inf):
                    gs[v] = ng
                    came[v] = u
                    h = math.hypot(v[0] - g[0], v[1] - g[1]) * self.res
                    heapq.heappush(openq, (ng + h, v))
        return None

    def line_free(self, p, q) -> bool:
        """Straight p-q crosses no blocked cell and gets no CLOSER to an
        obstacle than its own ends are (cost never above the ends' max)."""
        n = max(2, int(math.dist(p, q) / (self.res / 2)))
        ceiling = max(self.cost[self.cell(p)], self.cost[self.cell(q)]) + 0.05
        for t in np.linspace(0.0, 1.0, n):
            ij = self.cell((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
            if self.blocked[ij] or self.cost[ij] > ceiling:
                return False
        return True


def _shortcut(pts: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
    """Drop grid zig-zag: keep a point only where a straight run cannot skip
    it without passing closer to an obstacle than the run's own ends, so the
    path still keeps to the middle of doorways."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not grid.line_free(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return np.asarray(out)


def _densify(pts: np.ndarray, step: float) -> tuple[np.ndarray, list[int]]:
    """Resample at `step`; also return where each input point landed."""
    out, where = [pts[0]], [0]
    for p, q in zip(pts, pts[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(q - p) / step)))
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            out.append(p + (q - p) * t)
        where.append(len(out) - 1)
    return np.asarray(out), where


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


from .runtime import REGISTRY  # noqa: E402

REGISTRY.register("skate_route", SkateRoute)
