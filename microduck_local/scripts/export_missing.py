"""Export `policy.onnx` for trained runs that have none, so the lab can see them.

A run without `policy.onnx` is invisible: `viz_server.discover_policies` lists
a run only `if (run / "policy.onnx").exists()`. Nothing on the teach path ever
wrote one, so a job that trained its full budget and saved every artefact
never appeared in the palette — 338 of 1085 trained runs on this machine on
2026-10-06, including every stage but one of the Asimov get-up chains, which
is what made "Get up (Asimov 1)" read as 5 runs instead of 30.

The lab exports on completion now (`viz_server.start_export`). This is the
backfill for everything trained before that.

    uv run python scripts/export_missing.py --match asimov      # one family
    uv run python scripts/export_missing.py --all               # everything
    uv run python scripts/export_missing.py --match asimov -n   # just list
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microduck_local.export_onnx import export  # noqa: E402


def runs_dir() -> Path:
    import os
    env = os.environ.get("MICRODUCK_RUNS_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "runs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default=None,
                    help="only runs whose name contains this")
    ap.add_argument("--all", action="store_true",
                    help="every run with a model.zip and no policy.onnx")
    ap.add_argument("-n", "--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.match and not a.all:
        ap.error("pass --match <substring> or --all")

    todo = []
    for r in sorted(runs_dir().iterdir()):
        if not r.is_dir():
            continue
        if a.match and a.match not in r.name:
            continue
        if not (r / "model.zip").exists():
            continue
        if (r / "policy.onnx").exists():
            continue
        todo.append(r)

    print(f"{len(todo)} run(s) trained with no policy.onnx")
    if a.dry_run:
        for r in todo:
            print("  would export", r.name)
        return 0

    ok = []
    bad: list[tuple[str, str]] = []
    for i, r in enumerate(todo, 1):
        try:
            export(r, r / "policy.onnx")
            ok.append(r.name)
            print(f"[{i}/{len(todo)}] {r.name}", flush=True)
        except Exception as exc:
            # A refusal is INFORMATION, not noise: a run whose recorded
            # contract disagrees with its checkpoint is a run that trained one
            # body and is being exported as another, and export_contract is
            # the thing that catches it. Keep going and name them all at the
            # end rather than stopping on the first.
            bad.append((r.name, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(todo)}] {r.name}  REFUSED: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if "--debug" in sys.argv:
                traceback.print_exc()

    print(f"\nexported {len(ok)}, refused {len(bad)}")
    for name, why in bad:
        print(f"  {name}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
