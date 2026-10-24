"""A real home's floor plan, read from a COLOUR MAP image, as a world.

The skating duck (`brain/skate.py`) is asked to go to a ROOM ("go and see if
anyone is in the kitchen"), so it needs the home as it is, not a sketch of it.
The cheapest accurate source most homes already have is the map a robot
vacuum's LIDAR built: a top-down image, one flat colour per room, walls and
furniture legs in grey, unreachable space in black. This module reads such an
image — hand-annotated with a few marks — into:

  - an occupancy grid (`CELL` m cells): floor / obstacle / STAIRS
  - one label per room, from its colour (optionally cut by a box, for two
    rooms the vacuum painted the same colour)
  - a `Scenario`: static boxes for every obstacle cell that borders the floor
    (merged into rectangles), on the roller-skate duck
  - keep-out rectangles for the stairs, which have NO wall: the edge of a
    stairwell is open, and a duck that rolls over it falls down a flight

THE PLAN IS PRIVATE DATA. A home's layout is nobody else's business, so the
image and its spec live in a directory git ignores (`PRIVATE_DIR`, or any
path you pass) and nothing derived from them is committed. Only this reader
is public.

The spec, `plan.json` beside the image (pixel coordinates are the image's):

    {"image": "map.png",
     "scale": {"from_px": [100, 400], "to_px": [200, 400], "cm": 80},   # a door you measured
     "rooms": [{"name": "living", "rgb": [80, 120, 176]},
               {"name": "hall",   "rgb": [176, 96, 72]},
               {"name": "wc", "rgb": [176, 96, 72], "box_px": [0, 0, 150, 200]}],
     "floor_rgb": [[56, 122, 134]],          # extra colours that are floor (a door mark)
     "stairs": {"dark_max": 12, "rgb": [[110, 20, 0]],    # the blacked-out stairwell
                "railed": false, "edge_cells": 3},          # no rail: the grey edge line is a drop
     "clear_px": [[60, 180, 120, 210]],      # make floor (a half-open door leaf)
     "block_px": [],                         # make obstacle (a closed exit)
     "openings": [{"name": "garden", "at_px": [10, 300], "width_cm": 85,
                   "out": [-1, 0], "depth_m": 1.2}]}   # cut a door and lay ground beyond it

Later rooms in "rooms" win, so a box-cut room listed after its parent takes
its cells. An opening adds a room of its own (the ground outside the door).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .scenario import Box, Duck, Scenario

CELL = 0.05                 # grid cell, m
WALL_HEIGHT = 0.6           # everything the map calls an obstacle; the map has no heights
BORDER_CELLS = 6            # obstacle cells further than this from any floor never meet a duck
FLOOR, OBSTACLE, STAIRS = 1, 0, 2
#: Where private plans live by default — git-ignored (see the repo's .gitignore).
PRIVATE_DIR = Path(__file__).resolve().parents[3] / "private"


@dataclass
class FloorPlan:
    name: str
    labels: np.ndarray                    # (H, W) uint8: FLOOR / OBSTACLE / STAIRS
    rooms: dict[str, np.ndarray]          # name -> (H, W) bool, floor cells of that room
    cell: float = CELL
    _open: dict = field(default_factory=dict, repr=False)

    # -- geometry ----------------------------------------------------------
    @property
    def size(self) -> tuple[float, float]:
        h, w = self.labels.shape
        return (w * self.cell, h * self.cell)

    def xy(self, i: int, j: int) -> tuple[float, float]:
        """Centre of cell (row i, col j) in world metres; the map's centre is
        the origin, +y is UP the image."""
        w, h = self.size
        return ((j + 0.5) * self.cell - w / 2, h / 2 - (i + 0.5) * self.cell)

    def ij(self, x: float, y: float) -> tuple[int, int]:
        w, h = self.size
        H, W = self.labels.shape
        return (min(max(int((h / 2 - y) / self.cell), 0), H - 1),
                min(max(int((x + w / 2) / self.cell), 0), W - 1))

    def area_at(self, x: float, y: float) -> str | None:
        i, j = self.ij(x, y)
        for name, m in self.rooms.items():
            if m[i, j]:
                return name
        return None

    def on_stairs(self, x: float, y: float) -> bool:
        i, j = self.ij(x, y)
        return bool(self.labels[i, j] == STAIRS)

    def open_point(self, room: str) -> tuple[float, float]:
        """The cell of `room` furthest from anything that is not floor —
        where "go to the kitchen" aims."""
        if room not in self._open:
            d = _chamfer(self.labels == FLOOR)
            d = np.where(self.rooms[room], d, -1.0)
            i, j = np.unravel_index(int(np.argmax(d)), d.shape)
            self._open[room] = self.xy(int(i), int(j))
        return self._open[room]

    # -- the world ---------------------------------------------------------
    def boxes(self) -> list[Box]:
        near = _chamfer(self.labels != OBSTACLE) <= BORDER_CELLS
        solid = (self.labels == OBSTACLE) & near
        out = []
        for i0, j0, i1, j1 in _rectangles(solid):
            (x0, y0), (x1, y1) = self.xy(i0, j0), self.xy(i1 - 1, j1 - 1)
            out.append(Box(((x0 + x1) / 2, (y0 + y1) / 2, WALL_HEIGHT / 2),
                           (abs(x1 - x0) + self.cell, abs(y1 - y0) + self.cell, WALL_HEIGHT),
                           0.0, 0.0, (0.78, 0.76, 0.72, 1.0)))
        return out

    def keepout(self) -> list[tuple[float, float, float, float]]:
        """Stairs as (x0, y0, x1, y1) world rectangles, for a planner."""
        out = []
        for i0, j0, i1, j1 in _rectangles(self.labels == STAIRS):
            (xa, ya), (xb, yb) = self.xy(i0, j0), self.xy(i1 - 1, j1 - 1)
            out.append((min(xa, xb) - self.cell / 2, min(ya, yb) - self.cell / 2,
                        max(xa, xb) + self.cell / 2, max(ya, yb) + self.cell / 2))
        return out

    def scenario(self, spawn_room: str | None = None, yaw: float = 0.0,
                 policy: str = "pollen:roller", brain: str | None = None) -> Scenario:
        ducks = []
        if spawn_room is not None:
            x, y = self.open_point(spawn_room)
            ducks.append(Duck("d0", (x, y, yaw), policy=policy, brain=brain))
        w, h = self.size
        return Scenario(name=self.name, floor=(w, h), boxes=self.boxes(), ducks=ducks,
                        collision="rollers")

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, spec_path: str | Path) -> "FloorPlan":
        spec_path = Path(spec_path)
        spec = json.loads(spec_path.read_text())
        from PIL import Image
        im = np.asarray(Image.open(spec_path.parent / spec["image"]).convert("RGB")).astype(int)
        return cls.from_pixels(spec_path.parent.name, im, spec)

    @classmethod
    def from_pixels(cls, name: str, im: np.ndarray, spec: dict) -> "FloorPlan":
        sc = spec["scale"]
        cm_px = float(sc["cm"]) / math.dist(sc["from_px"], sc["to_px"])
        H, W = im.shape[:2]
        px = CELL * 100 / cm_px                       # image pixels per cell

        def near(rgb, tol=45):
            return np.abs(im - np.asarray(rgb)).sum(axis=2) < tol

        room_px: dict[str, np.ndarray] = {}
        for r in spec.get("rooms", []):
            m = near(r["rgb"], r.get("tol", 45))
            if "box_px" in r:
                x0, y0, x1, y1 = r["box_px"]
                box = np.zeros_like(m)
                box[y0:y1, x0:x1] = True
                m &= box
                for other in room_px.values():
                    other &= ~box
            room_px[r["name"]] = m
        floor_px = np.zeros((H, W), bool)
        for m in room_px.values():
            floor_px |= m
        for rgb in spec.get("floor_rgb", []):
            floor_px |= near(rgb, 40)
        st = spec.get("stairs", {})
        stairs_px = np.zeros((H, W), bool)
        if st:
            dark = int(st.get("dark_max", 12))
            stairs_px = (im < dark).all(axis=2)
            for rgb in st.get("rgb", []):
                stairs_px |= near(rgb, 90)
        for x0, y0, x1, y1 in spec.get("clear_px", []):
            floor_px[y0:y1, x0:x1] = True
            stairs_px[y0:y1, x0:x1] = False
        for x0, y0, x1, y1 in spec.get("block_px", []):
            floor_px[y0:y1, x0:x1] = False

        def to_cells(mask, frac):
            nh, nw = int(H / px), int(W / px)
            out = np.zeros((nh, nw), bool)
            for i in range(nh):
                for j in range(nw):
                    blk = mask[int(i * px):int((i + 1) * px), int(j * px):int((j + 1) * px)]
                    out[i, j] = blk.size > 0 and blk.mean() >= frac
            return out

        # A cell is floor only if nearly ALL of it is: a chair leg is a few
        # pixels, and a cell it sits in is not somewhere a blade can roll.
        floor = to_cells(floor_px, 0.9)
        labels = np.where(floor, FLOOR, OBSTACLE).astype(np.uint8)
        stairs = to_cells(stairs_px, 0.5) & ~floor
        labels[stairs] = STAIRS
        # A vacuum draws the edge of a drop as a thin grey line, exactly like a
        # wall. Unless the spec says the stairwell is railed, a thin band of
        # "obstacle" between the floor and the stairs is that line, not a
        # rail: make it stairs, so the world has NO wall there and a duck that
        # rolls over the edge is counted as falling — the pessimistic reading.
        if st and not st.get("railed", False):
            band = int(st.get("edge_cells", 3))
            near_stairs = _chamfer(~stairs) <= band
            near_floor = _chamfer(~floor) <= band
            labels[(labels == OBSTACLE) & near_stairs & near_floor] = STAIRS
        rooms = {n: to_cells(m, 0.5) & floor for n, m in room_px.items()}

        for op in spec.get("openings", []):
            (ax, ay), (ox, oy) = op["at_px"], op.get("out", (-1, 0))
            ci, cj = int(ay / px), int(ax / px)
            half = int(round(op["width_cm"] / 2 / (CELL * 100)))
            depth = int(round(op.get("depth_m", 1.0) / CELL))
            wall = 4                                  # cells either side of the mark that are wall
            outside = np.zeros_like(floor)
            for t in range(-wall, depth + 1):
                for s in range(-half, half + 1):
                    i = ci + t * oy + s * (ox != 0)
                    j = cj + t * ox + s * (oy != 0)
                    if 0 <= i < labels.shape[0] and 0 <= j < labels.shape[1]:
                        labels[i, j] = FLOOR
                        if t > wall // 2:
                            outside[i, j] = True
            # the ground beyond the door: a patch as wide as the door x 3
            for t in range(wall, depth + 1):
                for s in range(-3 * half, 3 * half + 1):
                    i = ci + t * oy + s * (ox != 0)
                    j = cj + t * ox + s * (oy != 0)
                    if 0 <= i < labels.shape[0] and 0 <= j < labels.shape[1]:
                        labels[i, j] = FLOOR
                        outside[i, j] = True
            for m in rooms.values():
                m &= ~outside
            rooms[op["name"]] = outside
        return cls(name=name, labels=labels, rooms=rooms)


def plan_path(name: str) -> Path:
    """`private/<name>/plan.json`, or a path to a plan.json given directly."""
    p = Path(name)
    if p.suffix == ".json":
        return p
    return PRIVATE_DIR / name / "plan.json"


# -- helpers ---------------------------------------------------------------

def _chamfer(free: np.ndarray) -> np.ndarray:
    """Approximate distance (in cells) from every cell to the nearest cell
    where `free` is False; 0 on those cells. Two-pass 3-4 chamfer."""
    INF = 1e9
    d = np.where(free, INF, 0.0)
    H, W = d.shape
    for i in range(H):
        for j in range(W):
            if d[i, j]:
                best = d[i, j]
                if i > 0:
                    best = min(best, d[i - 1, j] + 1)
                    if j > 0:
                        best = min(best, d[i - 1, j - 1] + 1.414)
                    if j < W - 1:
                        best = min(best, d[i - 1, j + 1] + 1.414)
                if j > 0:
                    best = min(best, d[i, j - 1] + 1)
                d[i, j] = best
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            if d[i, j]:
                best = d[i, j]
                if i < H - 1:
                    best = min(best, d[i + 1, j] + 1)
                    if j < W - 1:
                        best = min(best, d[i + 1, j + 1] + 1.414)
                    if j > 0:
                        best = min(best, d[i + 1, j - 1] + 1.414)
                if j < W - 1:
                    best = min(best, d[i, j + 1] + 1)
                d[i, j] = best
    return d


def _rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover `mask` with axis-aligned rectangles (i0, j0, i1, j1), exclusive
    ends: greedy, row runs grown downward. Not minimal; a few hundred boxes
    for a whole flat at 5 cm."""
    m = mask.copy()
    H, W = m.shape
    out = []
    for i in range(H):
        j = 0
        while j < W:
            if not m[i, j]:
                j += 1
                continue
            j1 = j
            while j1 < W and m[i, j1]:
                j1 += 1
            i1 = i + 1
            while i1 < H and m[i1, j:j1].all():
                i1 += 1
            m[i:i1, j:j1] = False
            out.append((i, j, i1, j1))
            j = j1
    return out
