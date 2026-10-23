"""Locks for `scenario.Prop.look` — a PICTURE drawn over a prop's collider.

`look: "duck"` makes a prop a MICRODUCK FIGURINE: the real robot's own CAD,
posed once at its rest keyframe, frozen and scaled down. The contract is that
it changes nothing else — the collider still collides, still carries the mass,
and is still what every detector measures. These tests exist because a
cosmetic that quietly moved the physics would be very hard to notice from a
video, which is the only place a look is ever looked at.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from microduck_local.world import compose, validate_scenario
from microduck_local.world.compose import _duck_figurine
from microduck_local.world.scenario import PROP_LOOKS, ScenarioError

H, R, M = 0.093, 0.030, 0.027


def _room(look: str = "duck") -> dict:
    return {
        "name": "look-test",
        "floor": {"size": [2.0, 2.0]},
        "ducks": [],
        "props": [{"id": "k0", "pos": [0.4, 0.0, H / 2], "shape": "cylinder",
                   "size": [R, R, H], "mass": M, "cls": "toy",
                   "rgba": [0.98, 0.78, 0.2, 1.0], "look": look}],
    }


def _figurine_bbox(m, data, prefix="k0_look"):
    """The drawn figurine's real extent, from mesh VERTICES."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i in range(m.ngeom):
        if not m.geom(i).name.startswith(prefix):
            continue
        mid = int(m.geom_dataid[i])
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        w = m.mesh_vert[va:va + vn] @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return lo, hi


def test_unknown_look_is_refused():
    """An unrecognised look would render as the bare collider — a scenario
    that silently did not take. Closed set, loud failure."""
    with pytest.raises(ScenarioError, match="look"):
        validate_scenario(_room("dragon"))


def test_look_round_trips():
    sc = validate_scenario(_room())
    assert sc.props[0].look == "duck"
    assert sc.to_dict()["props"][0]["look"] == "duck"
    assert "" in PROP_LOOKS and validate_scenario(_room("")).props[0].look == ""


def test_the_look_is_visual_only_and_the_collider_is_untouched():
    """The point of the whole feature: physics and sensing see the collider."""
    m = compose(validate_scenario(_room()))
    col = m.geom("k0_geom")
    assert m.geom_contype[col.id] != 0, "the collider must still collide"
    assert float(m.geom_rgba[col.id][3]) == 0.0, "collider is hidden under the look"
    # mass AND inertia identical to the same prop with no look: the parts are
    # massless, so they cannot shift what the physics integrates.
    bare = compose(validate_scenario(_room("")))
    bid, bid0 = m.body("k0").id, bare.body("k0").id
    assert float(m.body_mass[bid]) == pytest.approx(M)
    assert float(m.body_mass[bid]) == pytest.approx(float(bare.body_mass[bid0]))
    assert list(m.body_inertia[bid]) == pytest.approx(list(bare.body_inertia[bid0]))
    parts, _ = _duck_figurine()
    looks = [m.geom(f"k0_look{i}") for i in range(len(parts))]
    assert looks, "a duck look must add geoms"
    for g in looks:
        assert m.geom_contype[g.id] == 0 and m.geom_conaffinity[g.id] == 0, \
            f"{g.name} must not collide"
        assert m.geom_group[g.id] == 2, \
            f"{g.name} must sit in the VISUAL group, out of every ranging group"


def test_no_look_leaves_the_prop_exactly_as_it_was():
    m = compose(validate_scenario(_room("")))
    assert float(m.geom_rgba[m.geom("k0_geom").id][3]) == 1.0
    assert not [n for n in (m.geom(i).name for i in range(m.ngeom)) if "look" in n]


def test_the_figurine_is_the_duck_scaled_and_seated_on_the_collider():
    """It must be the REAL duck at a true scale, standing on the prop's base
    and on its axis — not a duck-ish shape somewhere near it.

    This is the test that would have caught the bake's one real bug: the
    compiler shifts a mesh geom's pos by that mesh's centroid offset, so
    re-declaring `geom_xpos` applied the offset twice and the figurine came
    out 123 mm instead of 93 with its parts flung apart.
    """
    m = compose(validate_scenario(_room()))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    lo, hi = _figurine_bbox(m, d)
    _, duck_h = _duck_figurine()
    k = H / duck_h
    assert (hi[2] - lo[2]) == pytest.approx(H, abs=1e-4), "not scaled to the prop"
    body = d.xpos[m.body("k0").id]
    assert lo[2] == pytest.approx(float(body[2]) - H / 2, abs=1e-3), \
        "the figurine must stand on the collider's base, not float or sink"
    assert ((lo + hi) / 2)[:2] == pytest.approx(body[:2], abs=2e-3), \
        "the figurine must stand on the collider's axis"
    # ...and it is a SCALED DUCK, not a duck-sized primitive: every part is a
    # mesh, and there are as many as the duck has visual geoms.
    parts, _ = _duck_figurine()
    assert len(parts) > 20, "a figurine is the duck's CAD, not a few blobs"
    assert all(m.geom_type[m.geom(f"k0_look{i}").id] == mujoco.mjtGeom.mjGEOM_MESH
               for i in range(len(parts)))
    assert 0.2 < k < 0.5, f"expected a figurine, got {k:.2f} of a duck"


def test_the_browser_and_the_model_seat_the_figurine_identically():
    """The /sim page draws its own figurine from `GET /scene`'s `home` and
    `bbox` while `compose` bakes one into the model. Two renderers, one
    seating — and nothing but this notices when they drift."""
    from microduck_local.lab.robots import extract_scene
    scene = extract_scene()
    _, duck_h = _duck_figurine()
    assert scene["bbox"]["height"] == pytest.approx(duck_h, abs=1e-4)
    assert len(scene["home"]) == len(scene["bodies"])
    assert all(len(h["pos"]) == 3 and len(h["quat"]) == 4 for h in scene["home"])


def test_the_shipped_showcase_room_still_loads():
    path = Path(__file__).resolve().parents[1] / "scenarios" / "moss-duckling.json"
    sc = validate_scenario(json.loads(path.read_text()))
    assert [p.look for p in sc.props] == ["duck"] * len(sc.props)
    # the figurines must be the object MOSS's jaw was measured on
    from microduck_local.robots.moss_env import GRASP_PROPS
    duckling = GRASP_PROPS["duckling"]
    for p in sc.props:
        assert p.shape == duckling.shape == "cylinder"
        assert p.size[0] == pytest.approx(duckling.radius)
        assert p.size[2] == pytest.approx(2 * duckling.half_height)
        assert p.mass == pytest.approx(duckling.mass)
