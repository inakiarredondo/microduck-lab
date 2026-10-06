"""Asimov's colours: on the SPEC, on every link, and the same in both files.

The MJCF paints all 26 visual geoms with one material (`visual_grey`, rgba
0.7 0.7 0.7, on a defaults class), so without `asimov_walk.paint_shell` he is
26 identically flat grey parts. The look is read off menloresearch's own
README renders, which is a reading of a picture and says so — what a test can
hold is that the reading is APPLIED everywhere and that the two files stating
it agree.

The second one is the point. `robots/asimov_walk.py` paints the spec and
`duck-viewer/components/AsimovLook.tsx` paints the `/sim` person, and nothing
but a test connects them: the colours would drift the first time somebody
tuned one of the two, and the symptom would be a robot that is graphite in a
room and white on a stage — which is exactly the bug MarsLook.tsx's own
docstring records ("the same robot was orange in one tab and white in the
next").

Skipped, not failed, when Asimov's model is not downloaded: `AGENTS.md`, "A
test gated on optional assets SKIPS exactly where it would fail".
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

from microduck_local.robots import asimov
from microduck_local.robots import asimov_walk as W

pytestmark = pytest.mark.skipif(
    not asimov.ready(),
    reason="Asimov's model is not fetched (uv run fetch-robot asimov1)")

#: The viewer's half of the same table.
LOOK_TSX = (Path(__file__).resolve().parents[2]
            / "duck-viewer" / "components" / "AsimovLook.tsx")


def _painted() -> mujoco.MjModel:
    """The robot as every consumer gets it: painted AND with its cuts."""
    spec = mujoco.MjSpec.from_file(str(asimov.xml_path()))
    base = asimov.xml_path().parent
    for attr in ("meshdir", "texturedir", "assetdir"):
        rel = getattr(spec, attr, "") or ""
        if rel:
            setattr(spec, attr, str((base / rel).resolve()))
    W.paint_shell(spec)
    W.add_cut_parts(spec)
    return spec.compile()


def _visual_geoms(model: mujoco.MjModel):
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_contype[g]) != 0:
            continue
        mid = int(model.geom_matid[g])
        rgba = tuple(round(float(x), 4)
                     for x in (model.mat_rgba[mid] if mid >= 0
                               else model.geom_rgba[g]))
        yield model.body(int(model.geom_bodyid[g])).name, rgba


def _hex(rgba) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgba[:3])


# ------------------------------------------------------- the paint is applied

def test_not_one_visual_geom_is_left_on_the_mjcfs_single_grey():
    """The failure this whole file exists for, stated directly.

    `visual_grey` is 0.7 0.7 0.7. A geom still wearing it is a part the paint
    missed, and the symptom is not an error — it is one grey panel on an
    otherwise painted robot, which nobody notices until it is in a clip.
    """
    stale = [n for n, rgba in _visual_geoms(_painted())
             if rgba[:3] == (0.7, 0.7, 0.7)]
    assert not stale, f"still the MJCF's flat grey: {stale}"


def test_every_visual_geom_takes_one_of_the_three_colours():
    """Three kinds, and nothing in between — a fourth colour means a rule
    matched something it was not written for."""
    want = {tuple(c[:4]) for c in (W.ASIMOV_SHELL, W.ASIMOV_ACCENT,
                                   W.ASIMOV_SILVER, W.ASIMOV_RED,
                                   W.ASIMOV_HAND,
                                   W.ASIMOV_SOLE,
                                   W.ASIMOV_FOOT)}
    got = {rgba for _, rgba in _visual_geoms(_painted())}
    assert got <= want, f"unexpected colour(s): {got - want}"
    assert got == want, f"a colour nothing uses: {want - got}"


def test_the_shoe_is_the_foot_and_the_housings_are_the_accent():
    """The split, by link, against the model's own names.

    `ankle_roll` is the one joint-named link that is a shell — it is the foot
    — and a rule that read it as a housing would put two orange shoes on him.
    """
    rows = list(_visual_geoms(_painted()))
    by_name = dict(rows)
    for side in ("left", "right"):
        # the foot is TWO geoms now: the shoe and the magenta sole under it
        shoe = {rgba for name, rgba in rows if name == f"{side}_ankle_roll_link"}
        # ONE colour. The two-tone shoe needed a cut mesh, which is forbidden
        # (`AGENTS.md`) and was broken: the sole plane missed the mesh entirely.
        assert shoe == {tuple(W.ASIMOV_FOOT),
                            tuple(W.ASIMOV_SOLE)}, shoe
        for frag in W.ACCENT_LINKS:
            link = f"{side}_{frag}_link"
            if link in by_name:
                assert by_name[link] == tuple(W.ASIMOV_ACCENT), link
        for frag in W.SILVER_LINKS:
            link = f"{side}_{frag}_link"
            assert by_name[link] == tuple(W.ASIMOV_SILVER), link
        assert by_name[f"{side}_wrist_yaw_link"] == tuple(W.ASIMOV_HAND)
        for frag in ("hip_yaw", "knee", "shoulder_roll", "elbow"):
            assert by_name[f"{side}_{frag}_link"] == tuple(W.ASIMOV_SHELL), frag
    # ONE colour, and it is red. The two-tone torso came from cutting the mesh,
    # which `AGENTS.md` forbids ("this shipped on Asimov and sank his torso
    # 18 cm"); the cut table is empty now and `waist_yaw` is painted red
    # directly, which is what this module's own docstring always said it should
    # be — "`waist_yaw` is the WHOLE torso, so the back goes red with the chest".
    torso = {rgba for name, rgba in rows if name == "waist_yaw_link"}
    assert torso == {tuple(W.ASIMOV_RED), tuple(W.ASIMOV_SHELL)}, torso
    assert by_name["pelvis_link"] == tuple(W.ASIMOV_SHELL)
    # the helmet is a red rim with a dark visor; the rotators stay black
    head = {rgba for name, rgba in rows if name == "neck_pitch_link"}
    assert head == {tuple(W.ASIMOV_RED), tuple(W.ASIMOV_SHELL)}, head
    for frag in ("shoulder_pitch", "shoulder_yaw"):
        assert by_name[f"{side}_{frag}_link"] == tuple(W.ASIMOV_SHELL), frag


def test_the_material_is_cleared_and_not_only_the_rgba_set():
    """A material wins over a geom's own rgba in MuJoCo's renderer AND in the
    mesh dump (`g1.extract_visual_scene` reads `mat_rgba` whenever
    `matid >= 0`). Setting rgba under a material changes nothing anywhere and
    looks exactly like the change not being applied."""
    model = _painted()
    for g in range(model.ngeom):
        if (model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and int(model.geom_contype[g]) == 0):
            assert int(model.geom_matid[g]) < 0, model.geom(g).name


def test_the_body_the_lab_serves_is_painted_and_not_only_the_bare_spec():
    """Every reader goes through the painted spec, not just `paint_shell`'s
    caller: `robot_model()` and the viewer dump each used to load the MJCF
    again on their own, and a paint applied in one place only is how the
    MuJoCo sheet and the browser came to disagree about MARS."""
    from microduck_local.robots import registry as R

    scene = R.registry().get("asimov1").visual_scene()
    colours = {tuple(round(c, 4) for c in g["rgba"]) for g in scene["geoms"]}
    assert colours == {tuple(W.ASIMOV_SHELL), tuple(W.ASIMOV_ACCENT),
                       tuple(W.ASIMOV_SILVER), tuple(W.ASIMOV_RED),
                       tuple(W.ASIMOV_HAND), tuple(W.ASIMOV_SOLE),
                       tuple(W.ASIMOV_FOOT)}
    assert {g.get("mat") for g in scene["geoms"]} == {""}


# --------------------------------------------------- the two files agree

@pytest.mark.skipif(not LOOK_TSX.is_file(), reason="duck-viewer is not present")
def test_the_viewer_states_the_same_three_colours():
    """`AsimovLook.tsx` and `asimov_walk.py` are two readings of ONE table.

    Nothing but this test connects them, and the drift is silent: a robot
    graphite in a room and white on a stage, which is the bug MarsLook.tsx's
    own docstring records from the other side.
    """
    src = LOOK_TSX.read_text()
    for const, rgba in (("ASIMOV_SHELL", W.ASIMOV_SHELL),
                        ("ASIMOV_JOINT", W.ASIMOV_ACCENT),
                        ("ASIMOV_SILVER", W.ASIMOV_SILVER),
                        ("ASIMOV_RED", W.ASIMOV_RED),
                        ("ASIMOV_HAND", W.ASIMOV_HAND),
                        ("ASIMOV_SOLE", W.ASIMOV_SOLE),
                        ("ASIMOV_FOOT", W.ASIMOV_FOOT)):
        m = re.search(rf'export const {const} = "(#[0-9a-fA-F]{{6}})"', src)
        assert m, f"{const} is not declared in {LOOK_TSX.name}"
        assert m.group(1).lower() == _hex(rgba), (
            f"{const} is {m.group(1)} in the viewer and {_hex(rgba)} in "
            f"asimov_walk.py — change one, change both")


@pytest.mark.skipif(not LOOK_TSX.is_file(), reason="duck-viewer is not present")
def test_the_viewer_splits_the_links_the_same_way():
    """The same links on the same side of the same split. The viewer states
    it as a regex and Python as a tuple of fragments, deliberately — two
    spellings that have to agree is a test; one shared string copied into two
    files is a hope."""
    src = LOOK_TSX.read_text()
    rules = re.search(r"ASIMOV_RULES[^=]*=\s*\[(.*?)\];", src, re.S)
    assert rules, "ASIMOV_RULES is not declared"
    body = rules.group(1)
    for frag in W.ACCENT_LINKS:
        assert frag in body, f"{frag!r} is an accent link here and not there"
    for frag in W.SILVER_LINKS:
        assert frag in body, f"{frag!r} is a silver link here and not there"
    for frag in W.HAND_LINKS:
        assert frag in body, f"{frag!r} is a hand link here and not there"
    for frag in W.RED_LINKS:
        assert frag in body, f"{frag!r} is a red link here and not there"
    for frag in W.FOOT_LINKS:
        assert frag in body, f"{frag!r} is a foot link here and not there"

def test_the_look_moves_no_geometry_at_all():
    """Painting and CUTTING must not shift a visual mesh, and this pins it.

    A cut that split `model.mesh_vert` shipped on 2026-10-04 and sank the
    torso 18 cm — it ate the neck column and left the head floating above the
    shoulders. `mesh_vert` is MuJoCo's RECENTRED copy, so writing it back
    subtracts the recentre twice. Splitting the source STL instead — what
    `mars.split_base_mesh` has always done — lands every half within 4e-6 mm. The look is link-level now, and this case is what stops a
    cut coming back unnoticed.

    Compared at **qpos0 on both sides**, never a keyframe: `walk_spec` adds
    its own STAND key, so resetting both to "keyframe 0" compares two
    different POSES and reports differences in links nobody touched. That
    confounded the first diagnosis of the bug above.
    """
    raw = mujoco.MjSpec.from_file(str(asimov.xml_path()))
    base = asimov.xml_path().parent
    for attr in ("meshdir", "texturedir", "assetdir"):
        rel = getattr(raw, attr, "") or ""
        if rel:
            setattr(raw, attr, str((base / rel).resolve()))

    def bboxes(model: mujoco.MjModel) -> dict:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        out: dict = {}
        for g in range(model.ngeom):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if int(model.geom_contype[g]):
                continue
            mid = int(model.geom_dataid[g])
            va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            w = (model.mesh_vert[va:va + vn]
                 @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
            name = model.body(int(model.geom_bodyid[g])).name
            lo, hi = out.get(name, (np.full(3, np.inf), np.full(3, -np.inf)))
            out[name] = (np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0)))
        return out

    before, after = bboxes(raw.compile()), bboxes(W.walk_spec(standalone=True).compile())
    assert set(before) == set(after), set(before) ^ set(after)
    for name in before:
        err = max(np.abs(before[name][0] - after[name][0]).max(),
                  np.abs(before[name][1] - after[name][1]).max())
        assert err < 5e-4, f"{name} moved {err * 1000:.3f} mm"


# ------------------------------------------- the look cannot reach the physics

def test_painting_and_splitting_visual_geoms_cannot_move_the_robot():
    """A look change must be invisible to the simulator, and this MEASURES it.

    Two independent reasons it holds for this model: every body carries an
    explicit `<inertial>` (so geoms contribute no mass at all), and the MJCF's
    `visual` defaults class is `density="0" contype="0" conaffinity="0"` with
    collision on 22 separate capsules in a `collision` class. But those are
    arguments. What settles it is running the thing.

    The split here is deliberately CRUDE — a sphere through the shin shell,
    far coarser than anything the look would ship — because the claim is about
    splitting visual geometry at all, not about one careful cut. If a finer
    cut is ever added for the knee boss, this case already covers it.
    """
    import struct
    import tempfile

    def write_stl(path, tris):
        blob = b"".join(struct.pack("<12fH", 0, 0, 0, *a, *b, *c, 0)
                        for a, b, c in tris)
        path.write_bytes(b"split test".ljust(80, b"\0")
                         + struct.pack("<I", len(tris)) + blob)

    tmp = Path(tempfile.mkdtemp())
    base = mujoco.MjSpec.from_file(str(asimov.xml_path())).compile()
    for side in ("left", "right"):
        mid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_MESH,
                                f"{side}_knee_link.STL")
        va, vn = int(base.mesh_vertadr[mid]), int(base.mesh_vertnum[mid])
        fa, fn = int(base.mesh_faceadr[mid]), int(base.mesh_facenum[mid])
        v = np.asarray(base.mesh_vert[va:va + vn], float)
        f = np.asarray(base.mesh_face[fa:fa + fn])
        cut = np.linalg.norm(v[f].mean(1), axis=1) < 0.075
        write_stl(tmp / f"{side}_a.stl", v[f[cut]])
        write_stl(tmp / f"{side}_b.stl", v[f[~cut]])

    def build(split: bool) -> mujoco.MjModel:
        spec = W.walk_spec(standalone=True)
        if split:
            for side in ("left", "right"):
                for tag in ("a", "b"):
                    mesh = spec.add_mesh()
                    mesh.name = f"{side}_{tag}"
                    mesh.file = str(tmp / f"{side}_{tag}.stl")
                body = [b for b in spec.bodies
                        if b.name == f"{side}_knee_link"][0]
                for g in list(body.geoms):
                    if g.contype == 0 and g.type == mujoco.mjtGeom.mjGEOM_MESH:
                        spec.delete(g)
                for tag in ("a", "b"):
                    g = body.add_geom()
                    g.type = mujoco.mjtGeom.mjGEOM_MESH
                    g.meshname = f"{side}_{tag}"
                    g.group, g.contype, g.conaffinity, g.density = 2, 0, 0, 0
        return spec.compile()

    plain, split = build(False), build(True)
    assert split.ngeom > plain.ngeom, "the split did not actually add a geom"
    assert np.array_equal(plain.body_mass, split.body_mass)
    assert np.array_equal(plain.body_inertia, split.body_inertia)
    assert np.array_equal(plain.body_ipos, split.body_ipos)
    assert np.array_equal(plain.dof_armature, split.dof_armature)
    def colliders(model: mujoco.MjModel) -> list[str]:
        return sorted(model.geom(g).name for g in range(model.ngeom)
                      if model.geom_contype[g])

    assert colliders(plain) == colliders(split)

    def roll(model: mujoco.MjModel):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        rng = np.random.default_rng(0)
        ctrl = rng.normal(0, 0.05, (150, model.nu))
        for step in range(150):
            data.ctrl[:] = ctrl[step]
            for _ in range(10):
                mujoco.mj_step(model, data)
        return data.qpos.copy(), data.qvel.copy()

    qa, va = roll(plain)
    qb, vb = roll(split)
    # BIT-identical, not close: anything else means the look reached the sim.
    assert np.array_equal(qa, qb), f"qpos drifted by {np.abs(qa - qb).max():.3e}"
    assert np.array_equal(va, vb), f"qvel drifted by {np.abs(va - vb).max():.3e}"

# ------------------------------------------------ the cuts are CLEAN cuts

def _seam_smooth_fraction(verts, picked) -> tuple[int, float]:
    """`(seam edges, fraction of them through SMOOTH surface)` for one cut.

    A cut's seam is every edge whose two triangles land on opposite sides of
    it. A seam that follows a real feature has sharp dihedral angles across
    it; a seam sawn across a smooth surface has flat ones — and flat is what
    the eye sees as serrated teeth.
    """
    from collections import defaultdict

    key = np.round(verts.reshape(-1, 3), 5)
    _uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    faces = np.asarray(inverse).reshape(len(verts), 3)
    normals = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    length = np.linalg.norm(normals, axis=1)
    good = length > 0
    unit = np.zeros_like(normals)
    unit[good] = normals[good] / length[good, None]
    edges: dict = defaultdict(list)
    for i, tri in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            p, q = int(tri[a]), int(tri[b])
            edges[(p, q) if p < q else (q, p)].append(i)
    angles = [
        np.degrees(np.arccos(np.clip(float(unit[fs[0]] @ unit[fs[1]]), -1, 1)))
        for fs in edges.values()
        if len(fs) == 2 and good[fs[0]] and good[fs[1]]
        and picked[fs[0]] != picked[fs[1]]
    ]
    if not angles:
        return 0, 0.0
    a = np.asarray(angles)
    return len(a), float((a < 12).mean())


def test_every_cut_follows_an_edge_instead_of_sawing_across_one():
    """A cut must land on the model's own features, not slice through them.

    MEASURED as the share of the seam whose dihedral angle is under 12 deg —
    i.e. the share running through SMOOTH surface. A plane across a curved
    surface scores high and looks SERRATED on the stage: Asimov's soles
    shipped at `z < -28 mm` and wore a row of magenta teeth, and the chest
    shipped at `z > 120 mm` with 83 % of its seam through smooth surface.
    Replacing both with face-normal tests took them to 2.1 % and 11.4 %.

    30 % is the bar: comfortably above every clean cut here and far below
    every serrated one that shipped.
    """
    for link, name, keep, _cut, _rest in W.CUT_PARTS:
        normals, verts = W._read_stl(W._mesh_file(link))
        unit = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True),
                                 1e-12, None)
        picked = np.asarray(
            keep(verts.mean(1), W._triangle_shells(verts), unit), dtype=bool)
        count, smooth = _seam_smooth_fraction(verts, picked)
        if count == 0:
            continue          # a whole-shell cut has no seam at all: ideal
        assert smooth < 0.30, (
            f"{name}: {100 * smooth:.1f}% of its {count}-edge seam runs "
            "through smooth surface — that is a serrated cut")

def test_a_cut_selects_one_part_and_not_confetti():
    """A cut must land on a PART, not scatter flecks over the whole shell.

    The seam test above asks whether the boundary follows creases. It does
    not ask whether there IS one boundary, and for weeks the chest passed it
    at 11.4 % while actually being 39 separate islands: a front plate, a back
    plate, and 36 specks strewn down the sides and up at the shoulder line
    (z = 344 mm), plus 35 lone black triangles stranded inside the red. On
    the stage that reads as a ragged plate with red flecks around the
    shoulders, which is how it was reported.

    This reads the STL `cut_part` actually WROTE, not a selection recomputed
    here. The first version of this test called `W._tidy_selection` itself
    and then asserted on its own output — so it passed happily against a
    planted `cut_part` with the tidy pass deleted, which is the whole thing
    it exists to catch.
    """
    for link, name, keep, _cut, _rest in W.CUT_PARTS:
        kept, _rest_path = W.cut_part(link, name, keep)
        _normals, verts = W._read_stl(kept)

        # 1. what was written is the TIDIED selection, not the raw predicate
        src_normals, src_verts = W._read_stl(W._mesh_file(link))
        unit = src_normals / np.clip(
            np.linalg.norm(src_normals, axis=1, keepdims=True), 1e-12, None)
        raw = np.asarray(keep(src_verts.mean(1),
                              W._triangle_shells(src_verts), unit), dtype=bool)
        want = int(W._tidy_selection(src_verts, raw).sum())
        assert len(verts) == want, (
            f"{name}: the written half has {len(verts)} triangles but the "
            f"tidied selection is {want} (the raw predicate is {int(raw.sum())}) "
            "— cut_part is not tidying what it writes")

        # 2. and it is made of PARTS: nothing in it is a speck
        left, right = W._edge_pairs(verts)
        area = np.linalg.norm(np.cross(verts[:, 1] - verts[:, 0],
                                       verts[:, 2] - verts[:, 0]), axis=1) / 2
        label = W._islands(len(verts), np.ones(len(verts), dtype=bool),
                           left, right)
        areas = np.bincount(label, weights=area) * 1e6
        assert areas.min() >= W.SPECK_MM2, (
            f"{name}: an island of {areas.min():.1f} mm^2 was written — "
            f"anything under {W.SPECK_MM2:.0f} mm^2 is a speck, not a part")


def test_both_shoulders_wear_the_same_pad():
    """The two shoulder pads are one part, mirrored — so they must measure
    the same.

    They did not. `keep = s != 0` means "every shell but the biggest", and
    the two links do not decompose alike: the right rotator barrel is a
    single component, the left arrives already split in half. Ranked by
    TRIANGLE COUNT, "not shell 0" therefore took both plates on the right
    and both plates PLUS half a barrel on the left — 30 143 mm^2 against
    14 654, so the robot's left shoulder wore twice the bronze of its right
    and only the left showed any from the front.
    """
    got = {}
    for side in ("left", "right"):
        link, name, keep, _cut, _rest = next(
            c for c in W.CUT_PARTS if c[1] == f"asimov_shoulder_pad_{side[0]}")
        normals, verts = W._read_stl(W._mesh_file(link))
        unit = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True),
                                 1e-12, None)
        picked = W._tidy_selection(verts, np.asarray(
            keep(verts.mean(1), W._triangle_shells(verts), unit), dtype=bool))
        v = verts[picked]
        got[side] = float(np.linalg.norm(
            np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2) * 1e6
    lo, hi = sorted(got.values())
    assert hi / lo < 1.01, (
        f"the shoulder pads differ by {100 * (hi / lo - 1):.1f}% — "
        f"left {got['left']:.0f} mm^2, right {got['right']:.0f} mm^2; "
        "they are the same part mirrored and must measure alike")
