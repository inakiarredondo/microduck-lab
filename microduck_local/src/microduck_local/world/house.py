"""A house floor plan, as rooms and doors, compiled into a `Scenario`.

The skating tour (`brain/skate.py`) needs a real home rather than a walled
rectangle: rooms that share walls, doorways cut into them, an exterior door
onto a garden, furniture. Writing that as raw wall segments by hand is where
the mistakes go — a doorway that is cut in one room's wall and not in its
neighbour's leaves a wall standing in the opening — so a plan is declared the
way a person describes a flat, and the walls are DERIVED:

    plan = HousePlan(
        rooms=[Room("salon", (0, 0), (4, 3)), Room("cocina", (4, 0), (7, 3))],
        doors=[Door((4, 1.5), 0.8)],                  # on the shared wall
        gardens=[Garden("delantero", (0, -2), (7, 0))],
        exits=[Door((2, 0), 0.9)],                    # house -> garden
        furniture=[Furniture("sofa", (1, 2.5), (1.8, 0.8), 0.4)])
    scenario = plan.scenario(duck_spawn=("salon", 0.0))

Every room is an axis-aligned rectangle (x0, y0) - (x1, y1) in metres. A wall
is drawn once per distinct edge segment, so two rooms that share an edge get
ONE wall there, and a door is a gap cut in whatever wall segment passes
through its centre. A garden is open ground: it gets the floor and, around
its outside, a low fence so a duck that leaves the plan does not skate off
the end of the world. Furniture is a static box.

Wall height defaults to 0.6 m: well above the duck (~0.25 m) and its head
ToF, low enough that the /sim camera still sees over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .scenario import Box, Duck, Scenario, Wall

WALL_HEIGHT = 0.6
WALL_THICKNESS = 0.08
FENCE_HEIGHT = 0.3
#: Floor beyond the plan's outer edge, so the plane is drawn under the fence.
FLOOR_MARGIN = 0.5


@dataclass(frozen=True)
class Room:
    name: str
    lo: tuple[float, float]
    hi: tuple[float, float]

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.lo[0] + self.hi[0]) / 2, (self.lo[1] + self.hi[1]) / 2)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.lo[0] + margin <= x <= self.hi[0] - margin
                and self.lo[1] + margin <= y <= self.hi[1] - margin)


@dataclass(frozen=True)
class Garden(Room):
    """Open ground outside the house. Walls only where it borders nothing."""


@dataclass(frozen=True)
class Door:
    """A gap of `width` metres centred on `at`, cut in the wall through it."""
    at: tuple[float, float]
    width: float = 0.8


@dataclass(frozen=True)
class Furniture:
    name: str
    centre: tuple[float, float]
    size: tuple[float, float]          # full x, y extents
    height: float = 0.4
    yaw: float = 0.0                   # open_point() treats furniture as axis-aligned


@dataclass
class HousePlan:
    name: str = "house"
    rooms: list[Room] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)       # room <-> room
    gardens: list[Garden] = field(default_factory=list)
    exits: list[Door] = field(default_factory=list)       # house <-> garden
    furniture: list[Furniture] = field(default_factory=list)
    wall_height: float = WALL_HEIGHT

    # -- lookups ----------------------------------------------------------
    def area(self, name: str) -> Room:
        for r in (*self.rooms, *self.gardens):
            if r.name == name:
                return r
        raise KeyError(f"no room or garden named {name!r}; have "
                       f"{[r.name for r in (*self.rooms, *self.gardens)]}")

    def area_at(self, x: float, y: float) -> Room | None:
        for r in (*self.rooms, *self.gardens):
            if r.contains(x, y):
                return r
        return None

    def open_point(self, name: str, step: float = 0.05) -> tuple[float, float]:
        """The point of a room or garden furthest from its walls and its
        furniture — where a duck is spawned and what "go to the kitchen"
        aims at. The geometric CENTRE is not it: a table or a bed sits there
        in most real rooms."""
        r = self.area(name)
        best, best_d = r.centre, -1.0
        x = r.lo[0] + step
        while x < r.hi[0]:
            y = r.lo[1] + step
            while y < r.hi[1]:
                d = min(x - r.lo[0], r.hi[0] - x, y - r.lo[1], r.hi[1] - y)
                for f in self.furniture:
                    dx = max(abs(x - f.centre[0]) - f.size[0] / 2, 0.0)
                    dy = max(abs(y - f.centre[1]) - f.size[1] / 2, 0.0)
                    d = min(d, (dx * dx + dy * dy) ** 0.5)
                if d > best_d + 1e-9:
                    best, best_d = (round(x, 3), round(y, 3)), d
                y += step
            x += step
        return best

    def door_between(self, a: str, b: str) -> Door:
        """The door whose centre lies on the boundary of both areas."""
        ra, rb = self.area(a), self.area(b)
        for d in (*self.doors, *self.exits):
            if _on_boundary(ra, d.at) and _on_boundary(rb, d.at):
                return d
        raise KeyError(f"no door between {a!r} and {b!r}")

    # -- geometry ---------------------------------------------------------
    def walls(self) -> list[Wall]:
        """Every distinct edge segment of every room, minus the door gaps.

        Edges are split at every room corner that lies on them, so a long
        wall shared by two short rooms becomes the pieces each one owns, and
        then de-duplicated: a shared piece is drawn once. A garden edge is a
        FENCE where no room is on the other side, and nothing where the house
        is (the house's own wall is already there)."""
        segs: dict[tuple, float] = {}
        areas = [(r, self.wall_height) for r in self.rooms] + [(g, FENCE_HEIGHT) for g in self.gardens]
        xs = sorted({v for r, _ in areas for v in (r.lo[0], r.hi[0])})
        ys = sorted({v for r, _ in areas for v in (r.lo[1], r.hi[1])})
        for r, h in areas:
            for (x0, y0), (x1, y1) in _edges(r):
                cuts = ([x for x in xs if x0 < x < x1] if y0 == y1 else
                        [y for y in ys if y0 < y < y1])
                pts = ([(x0, y0)] + [(c, y0) if y0 == y1 else (x0, c) for c in cuts] + [(x1, y1)])
                for p, q in zip(pts, pts[1:]):
                    key = (_r(p), _r(q))
                    if isinstance(r, Garden):
                        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                        if any(_on_boundary(room, mid) for room in self.rooms):
                            continue          # the house wall is the fence here
                    segs[key] = max(segs.get(key, 0.0), h)
        out: list[Wall] = []
        gaps = [*self.doors, *self.exits]
        for (p, q), h in sorted(segs.items()):
            for a, b in _cut(p, q, gaps):
                out.append(Wall(a, b, h, WALL_THICKNESS))
        return out

    def boxes(self) -> list[Box]:
        return [Box((f.centre[0], f.centre[1], f.height / 2), (f.size[0], f.size[1], f.height), f.yaw,
                    0.0, (0.55, 0.42, 0.30, 1.0)) for f in self.furniture]

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        allr = [*self.rooms, *self.gardens]
        return ((min(r.lo[0] for r in allr), min(r.lo[1] for r in allr)),
                (max(r.hi[0] for r in allr), max(r.hi[1] for r in allr)))

    def scenario(self, duck_spawn: tuple[str, float] | None = None, policy: str = "pollen:roller",
                 brain: str | None = None, tof: str = "datasheet") -> Scenario:
        """The plan as a world, centred on the origin (the floor plane is
        centred there). Rooms keep the coordinates they were declared in
        only if the plan is already centred — use `centred()` first."""
        (x0, y0), (x1, y1) = self.bounds()
        if abs(x0 + x1) > 1e-6 or abs(y0 + y1) > 1e-6:
            raise ValueError("plan is not centred on the origin; call .centred() first")
        ducks = []
        if duck_spawn is not None:
            room, yaw = duck_spawn
            cx, cy = self.open_point(room)
            ducks.append(Duck("d0", (cx, cy, yaw), policy=policy, tof=tof, brain=brain))
        return Scenario(name=self.name, floor=(x1 - x0 + 2 * FLOOR_MARGIN, y1 - y0 + 2 * FLOOR_MARGIN),
                        walls=self.walls(), boxes=self.boxes(), ducks=ducks, collision="rollers")

    def centred(self) -> "HousePlan":
        """The same plan translated so its bounding box is centred on (0, 0)."""
        (x0, y0), (x1, y1) = self.bounds()
        dx, dy = -(x0 + x1) / 2, -(y0 + y1) / 2

        def mv(p):
            return (p[0] + dx, p[1] + dy)
        return HousePlan(
            name=self.name,
            rooms=[Room(r.name, mv(r.lo), mv(r.hi)) for r in self.rooms],
            doors=[Door(mv(d.at), d.width) for d in self.doors],
            gardens=[Garden(g.name, mv(g.lo), mv(g.hi)) for g in self.gardens],
            exits=[Door(mv(d.at), d.width) for d in self.exits],
            furniture=[Furniture(f.name, mv(f.centre), f.size, f.height, f.yaw) for f in self.furniture],
            wall_height=self.wall_height)

    def route_waypoints(self, route: list[str], clearance: float = 0.45) -> list[tuple[str, tuple[float, float]]]:
        """The tour as goals: for each leg a = route[i] -> b = route[i+1], the
        point `clearance` m inside a on the door's axis, the door centre, and
        the point `clearance` m inside b — so the duck squares up to every
        doorway instead of clipping its jamb — then b's most open point (`open_point`). The planner
        (`brain/skate.py`) is what keeps the path off the furniture between
        these; they only say which DOOR to take.

        Names are numbered by leg (`3:cocina`), so a room the tour passes
        through twice is two distinct goals."""
        out: list[tuple[str, tuple[float, float]]] = [(f"0:{route[0]}", self.open_point(route[0]))]
        for i, (a, b) in enumerate(zip(route, route[1:]), start=1):
            d = self.door_between(a, b)
            nx, ny = _door_normal(self.area(a), d.at)
            out.append((f"{i}:{a}>{b}:approach", (d.at[0] - nx * clearance, d.at[1] - ny * clearance)))
            out.append((f"{i}:{a}>{b}:door", d.at))
            out.append((f"{i}:{a}>{b}:through", (d.at[0] + nx * clearance, d.at[1] + ny * clearance)))
            out.append((f"{i}:{b}", self.open_point(b)))
        return out


# -- helpers ----------------------------------------------------------------

def _r(p) -> tuple[float, float]:
    return (round(p[0], 6), round(p[1], 6))


def _edges(r: Room):
    (x0, y0), (x1, y1) = r.lo, r.hi
    return [((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))]


def _on_boundary(r: Room, p, tol: float = 1e-6) -> bool:
    (x0, y0), (x1, y1) = r.lo, r.hi
    x, y = p
    on_x = (abs(x - x0) < tol or abs(x - x1) < tol) and y0 - tol <= y <= y1 + tol
    on_y = (abs(y - y0) < tol or abs(y - y1) < tol) and x0 - tol <= x <= x1 + tol
    return on_x or on_y


def _door_normal(r: Room, at) -> tuple[float, float]:
    """Unit vector from room `r` OUT through the door at `at`."""
    (x0, y0), (x1, y1) = r.lo, r.hi
    x, y = at
    cands = [(abs(x - x0), (-1.0, 0.0)), (abs(x - x1), (1.0, 0.0)),
             (abs(y - y0), (0.0, -1.0)), (abs(y - y1), (0.0, 1.0))]
    return min(cands)[1]


def _cut(p, q, doors: list[Door]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Segment p-q minus every door gap centred on it."""
    (x0, y0), (x1, y1) = p, q
    horiz = abs(y0 - y1) < 1e-9
    lo, hi = (x0, x1) if horiz else (y0, y1)
    fixed = y0 if horiz else x0
    gaps = []
    for d in doors:
        along, across = (d.at[0], d.at[1]) if horiz else (d.at[1], d.at[0])
        if abs(across - fixed) < 1e-6 and lo - 1e-6 <= along <= hi + 1e-6:
            gaps.append((along - d.width / 2, along + d.width / 2))
    pieces, start = [], lo
    for g0, g1 in sorted(gaps):
        if g0 > start + 1e-3:
            pieces.append((start, min(g0, hi)))
        start = max(start, g1)
    if hi > start + 1e-3:
        pieces.append((start, hi))

    def pt(v):
        return (v, fixed) if horiz else (fixed, v)
    return [(pt(a), pt(b)) for a, b in pieces if b - a > 1e-3]


def demo_plan() -> HousePlan:
    """A PLACEHOLDER house — four rooms off a hall, a front garden — so the
    skating tour has something to run on until the real plan's measurements
    are in. Not anybody's home; replace it, do not tune against it."""
    rooms = [
        Room("salon", (0.0, 0.0), (4.0, 3.5)),
        Room("cocina", (4.0, 0.0), (7.0, 3.5)),
        Room("dormitorio", (0.0, 3.5), (3.5, 6.5)),
        Room("bano", (3.5, 3.5), (7.0, 6.5)),
    ]
    doors = [Door((4.0, 1.2), 0.8),        # salon <-> cocina
             Door((1.8, 3.5), 0.8),        # salon <-> dormitorio
             Door((5.5, 3.5), 0.75)]       # cocina <-> bano
    gardens = [Garden("jardin_delantero", (0.0, -2.0), (7.0, 0.0))]
    exits = [Door((2.0, 0.0), 0.9)]        # salon -> jardin delantero
    furniture = [
        Furniture("sofa", (0.6, 1.8), (0.9, 2.0), 0.45),
        Furniture("mesa_salon", (2.2, 2.0), (1.0, 0.6), 0.45),
        Furniture("encimera", (6.65, 1.75), (0.6, 3.0), 0.9),
        Furniture("mesa_cocina", (5.2, 2.2), (1.2, 0.8), 0.75),
        Furniture("cama", (1.0, 5.2), (1.6, 2.0), 0.5),
        Furniture("armario", (3.2, 6.15), (0.5, 0.6), 1.2),
        Furniture("lavabo", (6.7, 5.8), (0.5, 0.8), 0.85),
        Furniture("banera", (4.4, 6.1), (1.7, 0.75), 0.55),
    ]
    return HousePlan("casa-demo", rooms, doors, gardens, exits, furniture).centred()


#: The houses the lab knows, by scenario name: a plan and its tour. A
#: `skate_route` brain on a scenario of that name tours it.
DEMO_ROUTE = ["salon", "dormitorio", "salon", "cocina", "bano", "cocina", "salon", "jardin_delantero"]
HOUSES = {"casa-demo": lambda: (demo_plan(), DEMO_ROUTE)}
