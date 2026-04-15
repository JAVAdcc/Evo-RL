#!/usr/bin/env python
"""Rescale reward_seq in cached chunk-transition .pt files in place.

Loads each <cache_dir>/chunk_transitions_<split>.pt produced by
src/lerobot/rlt/offline_dataset.py:save_transition_cache (a list of dicts with
key "reward_seq" holding (C,) torch tensors), multiplies every transition's
reward_seq by --scale, and writes the file back. The original is renamed to
<file><backup-suffix> the first time only, so re-running never clobbers the
true original.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--backup-suffix", default=".bak10")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def patch_split(
    cache_dir: Path, split: str, scale: float, backup_suffix: str, dry_run: bool,
) -> None:
    file = cache_dir / f"chunk_transitions_{split}.pt"
    if not file.exists():
        print(f"[{split}] SKIP (not found): {file}")
        return

    transitions = torch.load(file, weights_only=False)
    n = len(transitions)
    if n == 0:
        print(f"[{split}] EMPTY: {file}")
        return

    nz_before = sum(1 for t in transitions if bool((t["reward_seq"] != 0).any().item()))
    min_before = min(float(t["reward_seq"].min().item()) for t in transitions)
    max_before = max(float(t["reward_seq"].max().item()) for t in transitions)
    dtype_before = transitions[0]["reward_seq"].dtype

    for t in transitions:
        t["reward_seq"] = t["reward_seq"] * scale

    nz_after = sum(1 for t in transitions if bool((t["reward_seq"] != 0).any().item()))
    min_after = min(float(t["reward_seq"].min().item()) for t in transitions)
    max_after = max(float(t["reward_seq"].max().item()) for t in transitions)
    dtype_after = transitions[0]["reward_seq"].dtype

    print(
        f"[{split}] n={n} nz {nz_before}->{nz_after} "
        f"min {min_before:.4f}->{min_after:.4f} "
        f"max {max_before:.4f}->{max_after:.4f} "
        f"dtype {dtype_before}->{dtype_after}"
    )

    if dtype_before != dtype_after:
        raise RuntimeError(f"dtype changed for {file}: {dtype_before} -> {dtype_after}")

    if dry_run:
        print(f"[{split}] DRY-RUN: not writing")
        return

    backup = file.with_name(file.name + backup_suffix)
    if not backup.exists():
        file.rename(backup)
        print(f"[{split}] backed up original -> {backup.name}")
    else:
        print(f"[{split}] backup {backup.name} exists, skipping backup")

    torch.save(transitions, file)
    print(f"[{split}] wrote {file.name}")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)
    print(f"cache_dir={cache_dir} scale={args.scale} dry_run={args.dry_run}")
    for split in args.splits:
        patch_split(cache_dir, split, args.scale, args.backup_suffix, args.dry_run)


if __name__ == "__main__":
    main()
