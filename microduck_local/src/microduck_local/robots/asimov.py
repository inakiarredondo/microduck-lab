"""Menlo Research's Asimov 1 — a 1.2 m, 23-joint open-source biped.

    uv run fetch-robot asimov1      # the MJCF, the URDF, 25 meshes, 2 licences

[Asimov 1](https://github.com/menloresearch/asimov-1) is an open-source
humanoid: published BOM, mechanical and electrical CAD, and — the part this
file wants — a MuJoCo model. It is the FOURTH body here and the first that
belongs to strangers who also published the training config their policies
run under (`menloresearch/cyclotron`), which is what makes it the test case
`docs/mars-roadmap.md` §7.4 was written for: that table lists the `RobotSpec`
fields a stranger's MJCF cannot supply, and Asimov supplies every one of them
(a `<gyro>` the Go2 lacks, two feet rather than four, a published 70 deg fall
cut, published command ranges and a published obs layout).

**MEASURED at the pinned sha, in this repo's MuJoCo 3.10.0** — not read off
the README, which says 35 kg where the model says 32:

    nq 30   nv 29   23 hinges + a free `floating_base`   27 bodies   48 geoms
    mass 32.225 kg      qpos0 base z 0.630 m      5 sensors, no keyframes
    nu 0 — NO actuators: cyclotron adds them in Python, so a level-1 spec
           here has to inject position servos before anything can act
    0.098 ms/step as shipped (19.5 ms per simulated second); 0.028 ms/step
           under this repo's solver options (5.56 ms/sim-s, against the
           duck's 7.15 and the G1's 8.66) — the shipped `cone="elliptic"`,
           `impratio="10"` and `noslip_iterations="5"` are a foot-contact
           fidelity choice and they cost 3.5x. An `MjSpec.attach` into a
           room keeps the PARENT scene's options, so a lab duck and a lab
           Asimov already share one solver.

**Nothing is vendored, and that is a licence decision rather than a size one.**
`asimov-1` is GPL-2.0 (software) and CERN-OHL-S-2.0 (hardware) — both strongly
reciprocal — and this repo is Apache-2.0 and public. So the model is
DOWNLOADED, both licence files travel with it into the cache, and `fetch`
prints their first lines, because a licence that stayed on GitHub is a licence
nobody read (`robots/menagerie.py`'s rule, and the reason it is repeated here).

**The sha is pinned because a revision is a different robot.** The pinned one
is `fix(sim-model): Update urdf and xml motion range to align with the
hardware limits` (#25, 2026-09-28): the joint ranges in it ARE the hardware's,
which is the whole premise of training something deployable on it. An earlier
sha has wider ranges and would quietly teach a policy to ask for travel the
servos do not have.

This file is the SECOND copy of `robots/menagerie.py`'s download technique
(pinned listing, git-blob-sha1 refusal, sha256 manifest, atomic replace). It
is a copy on purpose: that one's tests replace its `_get` seam, and moving a
tested download path to avoid a duplicate is the refactor this repo's history
warns about. `docs/mars-roadmap.md` §6.7 ("One asset story") is where the four
fetched bodies merge, and this is the fourth reason to do it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]       # microduck_local/

#: The upstream repository and the revision every file below was read at.
#: `fix(sim-model): Update urdf and xml motion range to align with the
#: hardware limits` (#25, 2026-09-28) — see the module docstring on why a
#: different sha is a different robot.
REPO = "menloresearch/asimov-1"
ASIMOV_SHA = "35ae7b3581ce36762bfddb00b8fc5e7c957ee7fa"

BODY_ID = "asimov1"
API_BASE = f"https://api.github.com/repos/{REPO}/contents"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{ASIMOV_SHA}"

#: The subtree that IS the model. Its layout is preserved in the cache
#: because the MJCF spells `meshdir="../assets/meshes"` relative to `xmls/`:
#: flattening the download would compile to `Error opening file`.
SUBDIR = "sim-model"

#: Files outside `SUBDIR` worth having, as `{path in the cache: path in the
#: repo}`. Both licences, because the model is under one of them and which
#: one a simulation model falls under is the reader's call, not ours.
EXTRA_FILES: Mapping[str, str] = {
    "SOFTWARE-LICENSE.txt": "SOFTWARE-LICENSE.txt",
    "HARDWARE-LICENSE.txt": "HARDWARE-LICENSE.txt",
}

CACHE_DIR = Path(os.environ.get("MICRODUCK_ASIMOV_DIR")
                 or _ROOT / ".cache" / "asimov_1")

#: 41 MB over 28 files, so the whole download is one slow call, not many.
DOWNLOAD_TIMEOUT_S = 120.0

_WANTED_SUFFIXES = (".xml", ".urdf", ".md", ".stl", ".txt")

ROBOT_XML_REL = "xmls/asimov_1.xml"
URDF_REL = "urdf/asimov_1.urdf"


# --------------------------------------------------------------- the paths

def cache_dir() -> Path:
    return CACHE_DIR


def manifest_path(dest: Path | None = None) -> Path:
    return (dest or CACHE_DIR) / "body.json"


def xml_path(dest: Path | None = None) -> Path:
    return (dest or CACHE_DIR) / ROBOT_XML_REL


def urdf_path(dest: Path | None = None) -> Path:
    """The URDF Isaac Lab spawns (cyclotron's `ASIMOV_1_URDF_PATH`).

    Downloaded but unused here: MuJoCo reads the MJCF, and the two are the
    same robot only as long as upstream keeps them in step — which the pinned
    sha's own commit message says it is doing. Kept because a sim2sim check
    against cyclotron needs the file Isaac actually loaded.
    """
    return (dest or CACHE_DIR) / URDF_REL


def read_manifest(dest: Path | None = None) -> dict[str, Any] | None:
    p = manifest_path(dest)
    if not p.is_file():
        return None
    try:
        man = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return man if isinstance(man, dict) else None


def ready(dest: Path | None = None) -> bool:
    """Is the download present? Presence, not sha256 — `robots/mars.ready`'s
    reason: this is asked on every registry build and hashing 41 MB to answer
    a palette chip would be felt."""
    man = read_manifest(dest)
    if not man:
        return False
    d = dest or CACHE_DIR
    files = man.get("files") or ()
    return bool(files) and all((d / rel).is_file() for rel, _sha in files)


def setup_hint() -> str:
    return f"uv run fetch-robot {BODY_ID}"


def require_asimov(dest: Path | None = None) -> Path:
    """The cache directory, or a RuntimeError naming the command that fills
    it — the G1's `require_g1` contract, so a missing download is never an
    `Error opening file` from three layers down."""
    if not ready(dest):
        raise RuntimeError(
            f"Asimov 1's model is not downloaded — run `{setup_hint()}` "
            f"(~41 MB from {REPO} at {ASIMOV_SHA[:12]}; it is GPL-2.0 / "
            f"CERN-OHL-S-2.0 and is NOT vendored in this repo)")
    return dest or CACHE_DIR


# ----------------------------------------------------------- the download

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    """git's own content hash: `sha1("blob <len>\\0" + data)`, which is what
    the contents API publishes per file and the only thing that can tell the
    pinned tree apart from a moved branch or a captive portal's login page."""
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def _get(url: str) -> bytes:
    """One HTTP GET. The seam the tests replace, so the listing, the filter,
    the hash refusal, the atomic replace and the idempotence are all testable
    with no network."""
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "microduck-local/fetch-robot",
    })
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith(API_BASE):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as r:
        return r.read()


def _listing(path: str) -> list[dict[str, Any]]:
    url = f"{API_BASE}/{path}?ref={ASIMOV_SHA}"
    data = json.loads(_get(url).decode())
    if not isinstance(data, list):
        raise RuntimeError(
            f"{url} did not list a directory — {REPO} at "
            f"{ASIMOV_SHA[:12]} has no {path!r}")
    return data


def _wanted(rel: str) -> bool:
    return Path(rel).suffix.lower() in _WANTED_SUFFIXES


def _walk() -> list[tuple[str, str, str]]:
    """Every wanted file as `(path in the cache, path in the repo, blob sha1)`.

    Recursive over `SUBDIR` — `assets/meshes/` is a level down — plus
    `EXTRA_FILES`, which are named rather than discovered because they live at
    the repo root beside 229 MB of CAD nobody here wants.
    """
    out: list[tuple[str, str, str]] = []
    todo = [SUBDIR]
    while todo:
        path = todo.pop(0)
        for entry in _listing(path):
            repo_rel = str(entry["path"])
            rel = repo_rel[len(SUBDIR) + 1:]
            if entry["type"] == "dir":
                todo.append(repo_rel)
                continue
            if entry["type"] != "file" or not _wanted(rel):
                continue
            out.append((rel, repo_rel, str(entry["sha"])))
    for rel, repo_rel in EXTRA_FILES.items():
        entry = _listing_file(repo_rel)
        out.append((rel, repo_rel, str(entry["sha"])))
    return sorted(out)


def _listing_file(repo_rel: str) -> dict[str, Any]:
    """The contents API on ONE file (it answers an object, not a list)."""
    url = f"{API_BASE}/{repo_rel}?ref={ASIMOV_SHA}"
    data = json.loads(_get(url).decode())
    if not isinstance(data, dict) or "sha" not in data:
        raise RuntimeError(f"{url} did not describe a file")
    return data


def _verify_blob(rel: str, data: bytes, want_sha1: str) -> None:
    got = _git_blob_sha1(data)
    if got != want_sha1:
        raise RuntimeError(
            f"{rel} from {REPO} {ASIMOV_SHA[:12]} hashes {got[:12]}, the "
            f"listing says {want_sha1[:12]} — refusing it")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def licence_lines(dest: Path | None = None) -> tuple[str, ...]:
    """The first meaningful line of each licence that came with the model."""
    d = dest or CACHE_DIR
    out = []
    for rel in EXTRA_FILES:
        p = d / rel
        if not p.is_file():
            continue
        for line in p.read_text(errors="replace").splitlines():
            if line.strip():
                out.append(f"{rel}: {line.strip()}")
                break
    return tuple(out)


def fetch(dest: Path | None = None) -> Path:
    """Download the model into the cache. Returns its directory.

    Idempotent: a file that is present and hashes to what the manifest
    recorded is skipped, so a second run costs the listings and no bytes.
    Every file lands on a temp name and is `os.replace`d into place only after
    its git blob sha1 matches the pinned listing, so a half-written mesh can
    never be imported by a parallel worker (`AGENTS.md`, "Atomic writes and
    live imports").
    """
    d = dest or CACHE_DIR
    entries = _walk()
    if not entries:
        raise RuntimeError(
            f"{REPO} at {ASIMOV_SHA[:12]} listed no usable files under "
            f"{SUBDIR!r}")
    known = {rel: sha for rel, sha in (read_manifest(dest) or {}).get(
        "files", ())}
    fresh = bytes_new = 0
    recorded: list[list[str]] = []
    for rel, repo_rel, blob_sha1 in entries:
        out = d / rel
        want256 = known.get(rel)
        if out.is_file() and want256 and _sha256(out) == want256:
            recorded.append([rel, want256])
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.{os.getpid()}.part")
        print(f"[asimov] {rel}")
        try:
            data = _get(f"{RAW_BASE}/{repo_rel}")
            _verify_blob(rel, data, blob_sha1)
            tmp.write_bytes(data)
            recorded.append([rel, _sha256(tmp)])
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        fresh += 1
        bytes_new += len(data)
    _write_json(manifest_path(dest), {
        "id": BODY_ID,
        "repo": REPO,
        "sha": ASIMOV_SHA,
        "files": recorded,
        "robot_xml": ROBOT_XML_REL,
        "urdf": URDF_REL,
        "licences": list(licence_lines(dest)),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _ASIMOV.clear()
    print(f"[asimov] {fresh} file(s) downloaded ({bytes_new / 1e6:.1f} MB), "
          f"{len(entries) - fresh} already verified — {d}")
    for line in licence_lines(dest):
        print(f"[asimov] {line}")
    return d


# --------------------------------------------------------------- the body

#: One built body per cache directory, invalidated by `fetch`.
_ASIMOV: dict[str, Any] = {}


def body(dest: Path | None = None):
    """Asimov 1 as a `Body`. Level 1 — `robots/asimov_walk.spec()`.

    It arrived at level 0 (an `MjcfBody` read straight off the MJCF) and
    climbed the same day, because this robot supplies every field
    `docs/mars-roadmap.md` §7.4 says a stranger's model cannot: a `<gyro>`
    sensor, two feet whose side is legible in their geom names, a published
    fall cut, published command ranges and a published observation layout.
    The two that still had to be MEASURED rather than read were the ones that
    section predicts — `ground_tol` off the shipped policy's own gait, and
    `fall_height` off a limp drop.

    `dest` is honoured by the asset paths; the spec itself is a singleton
    because the env, the lab and the exporter must all be describing one
    robot.
    """
    from . import asimov_walk

    if not read_manifest(dest):
        raise FileNotFoundError(
            f"Asimov 1 is not fetched — run `{setup_hint()}`")
    return asimov_walk.spec()


def joint_groups(names) -> tuple[str, ...]:
    """One 🎬-panel section label PER JOINT, in `joint_names` order.

    The shape `Body.joint_groups` wants, and the shape this file got wrong
    first time: a tuple of five `(label, joints)` pairs read as "joint_groups
    is 5 for 23 joints" in `tests/test_body_conformance.py`, which is the
    off-by-a-shape the suite exists to catch — nothing would have crashed, the
    panel would just have labelled 23 sliders from a 5-long list.

    DERIVED from the names rather than counted out as `("left leg",) * 6 +
    ...` (the G1's form), because this model's joints arrive in an order no
    one would guess — left leg, right leg, waist, then the RIGHT arm before
    the left — and a hand-written run-length table silently mislabels every
    slider the day upstream reorders one. An unrecognised joint raises.
    """
    out: list[str] = []
    for n in names:
        side = ("left" if n.startswith("left_")
                else "right" if n.startswith("right_") else "")
        if any(k in n for k in ("hip", "knee", "ankle")):
            part = "leg"
        elif any(k in n for k in ("shoulder", "elbow", "wrist")):
            part = "arm"
        elif "waist" in n:
            side, part = "", "waist"
        elif "neck" in n or "head" in n:
            side, part = "", "head"
        else:
            raise RuntimeError(
                f"{BODY_ID}: no panel group for joint {n!r} — a revision of "
                f"{REPO} added a limb this does not know, and guessing would "
                f"mislabel the 🎬 panel rather than fail")
        out.append(f"{side} {part}".strip())
    return tuple(out)


class _LazyAsimov:
    """`ASIMOV1` without opening a 41 MB download at import time —
    `g1._LazyG1Spec`'s pattern, and the registry needs the same four
    attributes before any model is compiled."""

    __slots__ = ("id", "title", "noun", "kind", "default_task")

    def __init__(self) -> None:
        self.id = BODY_ID
        self.title = "Asimov 1"
        self.noun = "Asimov"
        self.kind = "legged"           # level 1: see `body()`
        self.default_task = "walk"

    # Answered WITHOUT resolving, because the whole point of the proxy is a
    # body that is not downloaded yet: `fetch-robot` with no arguments asks
    # every body `ready()` to print the table, and the ⤓ button asks `fetch()`
    # precisely when there is no model to build. Routing either through
    # `_resolve` raised `FileNotFoundError` and took the listing down with it.
    def ready(self) -> bool:
        return ready()

    def fetch(self) -> Path:
        return fetch()

    def setup_hint(self) -> str:
        return setup_hint()

    def _resolve(self):
        return body()

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return getattr(self._resolve(), item)

    def __repr__(self) -> str:
        return f"LazyBody(id={BODY_ID!r}, {'fetched' if ready() else 'not fetched'})"


ASIMOV1 = _LazyAsimov()
