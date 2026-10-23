"""Merge clip sets into one, because a rung is a SET of starts.

Fine-tuning on the post-stand clips ALONE would repeat the mistake this repo
has already paid for: training each curriculum rung only on its newest start
made Asimov forget, and its hold collapsed 7.76 s -> 0.24 s across five rungs
while every per-rung score still read 12/12. A clip set that covers only the
hold would buy the hold and lose the rise.

    uv run python scripts/merge_getup_clips.py \
        .cache/asimov_1/getup_clips.npz .cache/asimov_1/poststand_clips.npz \
        --out .cache/asimov_1/getup_clips_mixed.npz
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    qpos, lengths, starts, tts, names, srcs = [], [], [], [], None, []
    for p in a.inputs:
        d = np.load(p, allow_pickle=True)
        lens = d["lengths"].astype(int)
        if len(d["starts"]) != len(lens):
            raise SystemExit(f"{p}: {len(lens)} clips but {len(d['starts'])} starts")
        if names is None:
            names = d["joint_names"]
        elif list(map(str, names)) != list(map(str, d["joint_names"])):
            raise SystemExit(f"{p}: joint order differs — merging would relabel "
                             "every frame, which is a different robot")
        qpos.append(d["qpos"]); lengths.append(lens); starts.append(d["starts"])
        tts.append(d["time_to_stand"] if "time_to_stand" in d
                   else np.zeros(len(lens)))
        srcs.append(f"{p.name}({len(lens)})")
        print(f"  {p.name:28} {len(lens):4d} clips, "
              f"{lens.min()}-{lens.max()} steps")

    L = np.concatenate(lengths)
    np.savez(a.out, qpos=np.concatenate(qpos, axis=0), lengths=L,
             starts=np.concatenate(starts, axis=0),
             time_to_stand=np.concatenate(tts), joint_names=names,
             source=" + ".join(srcs))
    print(f"\n-> {a.out}: {len(L)} clips, {L.sum()} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
