"""Rewrite complementary_info.is_intervention -> 1.0 for teleop-only datasets.

Used in two places:
  * record_teleop_critical_phase.py — post-record hook so every new teleop
    recording is stamped consistently.
  * CLI — one-off fix for legacy datasets recorded before the rule was added
    (e.g. 0414_teleop_cp).

Policy rule (decided 2026-04-14): every frame of a pure-teleop critical-phase
dataset must have complementary_info.is_intervention=1.0, because
conceptually the human is "intervening" over a null policy for the entire
duration. extract_cp_datasets.py verifies this invariant before extraction.

Usage:
    PYTHONPATH=src python -m scripts.dataset.fix_teleop_is_intervention \
        --dataset-dir ~/.roboclaw/workspace/embodied/datasets/0414_teleop_cp
    PYTHONPATH=src python -m scripts.dataset.fix_teleop_is_intervention \
        --dataset-dir ~/.roboclaw/workspace/embodied/datasets/0414_teleop_cp/teleop_cp_172257
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

_COL = "complementary_info.is_intervention"


def _is_dataset_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").exists()


def _find_datasets(root: Path) -> list[Path]:
    if _is_dataset_dir(root):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and _is_dataset_dir(d))


def rewrite_parquet(parquet_path: Path) -> tuple[int, int]:
    """Rewrite is_intervention column to 1.0 in-place. Returns (changed, total)."""
    table = pq.read_table(parquet_path)
    if _COL not in table.column_names:
        return 0, table.num_rows

    old_col = table.column(_COL)
    old_vals = old_col.to_pylist()
    new_vals = [1.0] * len(old_vals)
    changed = sum(1 for v in old_vals if float(v) != 1.0)

    new_array = pa.array(new_vals, type=old_col.type)
    idx = table.column_names.index(_COL)
    new_table = table.set_column(idx, _COL, new_array)

    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    pq.write_table(new_table, tmp_path)
    tmp_path.replace(parquet_path)
    return changed, table.num_rows


def rewrite_recovery_jsonl(jsonl_path: Path) -> int:
    """Rewrite is_intervention in recovery_frames.jsonl. Returns #lines changed."""
    if not jsonl_path.exists():
        return 0
    lines = jsonl_path.read_text().splitlines()
    changed = 0
    new_lines: list[str] = []
    for raw in lines:
        stripped = raw.replace("\x00", "").strip()
        if not stripped:
            new_lines.append(raw)
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            new_lines.append(raw)
            continue
        if _COL in row:
            old = row[_COL]
            new_val = [1.0] if isinstance(old, list) else 1.0
            if old != new_val:
                row[_COL] = new_val
                changed += 1
            new_lines.append(json.dumps(row))
        else:
            new_lines.append(raw)
    jsonl_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    return changed


def rewrite_dataset(dataset_dir: Path) -> dict:
    """Rewrite every parquet + recovery jsonl in a single dataset dir."""
    stats = {"parquet_files": 0, "parquet_changed": 0, "parquet_rows": 0, "jsonl_changed": 0}
    parquet_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    for fp in parquet_files:
        changed, rows = rewrite_parquet(fp)
        stats["parquet_files"] += 1
        stats["parquet_changed"] += changed
        stats["parquet_rows"] += rows
    stats["jsonl_changed"] = rewrite_recovery_jsonl(dataset_dir / "recovery_frames.jsonl")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="A single dataset dir or a parent containing multiple")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.dataset_dir.exists():
        log.error("Path does not exist: %s", args.dataset_dir)
        sys.exit(1)

    datasets = _find_datasets(args.dataset_dir)
    if not datasets:
        log.error("No datasets (meta/info.json) found under %s", args.dataset_dir)
        sys.exit(1)

    log.info("Found %d dataset(s)", len(datasets))
    total_changed = 0
    total_rows = 0
    for ds in datasets:
        parquet_files = sorted((ds / "data").rglob("*.parquet"))
        if not parquet_files:
            log.info("  %s: no parquet files (empty shell?) — skipped", ds.name)
            continue
        if args.dry_run:
            changed = 0
            rows = 0
            for fp in parquet_files:
                t = pq.read_table(fp, columns=[_COL]) if _COL in pq.read_schema(fp).names else None
                if t is None:
                    continue
                vals = t.column(_COL).to_pylist()
                rows += len(vals)
                changed += sum(1 for v in vals if float(v) != 1.0)
            log.info("  %s: would rewrite %d/%d rows", ds.name, changed, rows)
            total_changed += changed
            total_rows += rows
        else:
            stats = rewrite_dataset(ds)
            log.info("  %s: rewrote %d/%d parquet rows, %d jsonl lines",
                     ds.name, stats["parquet_changed"], stats["parquet_rows"], stats["jsonl_changed"])
            total_changed += stats["parquet_changed"]
            total_rows += stats["parquet_rows"]

    log.info("Total: %d rows changed out of %d", total_changed, total_rows)


if __name__ == "__main__":
    main()
