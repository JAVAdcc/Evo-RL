"""Extract success/fail datasets from recorded datasets with CP annotations.

Three source dataset types are recognized via directory-name prefix:

    CPDataType.SFT        eval_autonomy_*               VLA auto-rollout + human CP annotation
    CPDataType.RLT_HIL    eval_rlt_hil_wo_prefix_*      RL-executed CP + intermittent human intervention
    CPDataType.TELEOP     teleop_cp_*                   Pure human-teleop CP

The extractor looks for any sub-dir that contains critical_phase_intervals.json
*and* matches one of the known prefixes; the matched prefix is stripped to get
the time_tag used in the extracted output name (cp_{outcome}_{time_tag}).

Usage:
    PYTHONPATH=src python scripts/extract_cp_datasets.py --date 0414_rlt_hil_wo_prefix
    PYTHONPATH=src python scripts/extract_cp_datasets.py --all
    PYTHONPATH=src python scripts/extract_cp_datasets.py --dataset-dir /path/to/source_dir
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from enum import Enum
from pathlib import Path

from lerobot.utils.critical_phase_extraction_fast import extract_critical_phase_dataset_direct as extract_critical_phase_dataset

from .setup_helpers import load_setup_json, resolve_dataset_root


class CPDataType(Enum):
    SFT = "eval_autonomy_"
    RLT_HIL = "eval_rlt_hil_wo_prefix_"
    TELEOP = "teleop_cp_"


def _match_type(dir_name: str) -> tuple[CPDataType, str] | None:
    """Return (type, time_tag) if dir_name matches a known prefix, else None.

    time_tag = dir_name with the matched prefix stripped and any trailing
    _repaired suffix removed. Example:
        eval_rlt_hil_wo_prefix_022750_repaired -> (RLT_HIL, "022750")
    """
    for t in CPDataType:
        if dir_name.startswith(t.value):
            tag = dir_name[len(t.value):]
            if tag.endswith("_repaired"):
                tag = tag[: -len("_repaired")]
            return t, tag
    return None


def find_cp_datasets_in_dir(parent: Path) -> list[Path]:
    """Find child dirs with a known prefix + critical_phase_intervals.json."""
    if not parent.is_dir():
        return []
    return sorted(
        d for d in parent.iterdir()
        if d.is_dir()
        and _match_type(d.name) is not None
        and (d / "critical_phase_intervals.json").exists()
    )


def find_all_cp_datasets(ds_root_base: Path) -> list[Path]:
    """Search all date folders (MMDD_*) for CP-annotated datasets."""
    results = []
    for day_dir in sorted(ds_root_base.iterdir()):
        if day_dir.is_dir() and len(day_dir.name) >= 4 and day_dir.name[:4].isdigit():
            results.extend(find_cp_datasets_in_dir(day_dir))
    return results


def _cp_output_dir(day_dir: Path) -> Path:
    cp_dir = day_dir.parent / f"{day_dir.name}_cp"
    cp_dir.mkdir(exist_ok=True)
    return cp_dir


def _verify_teleop_is_intervention(source_dir: Path) -> None:
    """Fail if a teleop_cp source dataset has any frame with is_intervention != 1."""
    import pyarrow.parquet as pq

    col_name = "complementary_info.is_intervention"
    for fp in sorted((source_dir / "data").rglob("*.parquet")):
        schema_names = pq.read_schema(fp).names
        if col_name not in schema_names:
            raise RuntimeError(
                f"{source_dir.name}: {col_name} column missing in {fp.name}"
            )
        col = pq.read_table(fp, columns=[col_name]).column(col_name).to_pylist()
        bad = sum(1 for v in col if float(v) != 1.0)
        if bad:
            raise RuntimeError(
                f"{source_dir.name}: teleop dataset has {bad}/{len(col)} frames with "
                f"is_intervention != 1 in {fp.name}. "
                f"Run: python -m scripts.dataset.fix_teleop_is_intervention "
                f"--dataset-dir {source_dir}"
            )


def _run_extract(
    source_dir: Path,
    source_repo_id: str,
    cp_dir: Path,
    out_name: str,
    matching: list,
    task: str,
) -> bool:
    out_dir = cp_dir / out_name
    if out_dir.exists():
        print(f"  [skip] {out_name} already exists")
        return False
    extract_critical_phase_dataset(
        source_repo_id=source_repo_id,
        source_root=source_dir,
        output_repo_id=f"local/{out_name}",
        output_root=out_dir,
        intervals=matching,
        task=task,
    )
    return True


def extract_from_dataset(source_dir: Path, task: str) -> None:
    """Extract success/fail CP datasets from a single source dataset."""
    cp_json = source_dir / "critical_phase_intervals.json"
    if not cp_json.exists():
        print(f"  [skip] No critical_phase_intervals.json in {source_dir.name}")
        return

    match = _match_type(source_dir.name)
    if match is None:
        print(f"  [skip] Unknown dataset type: {source_dir.name}")
        return
    data_type, time_tag = match

    if data_type == CPDataType.TELEOP:
        _verify_teleop_is_intervention(source_dir)

    with open(cp_json) as f:
        raw_intervals = json.load(f)

    intervals = [
        (iv["episode_index"], iv["start_frame"], iv["end_frame"], iv.get("outcome"))
        for iv in raw_intervals
    ]

    if not intervals:
        print(f"  [skip] Empty intervals in {source_dir.name}")
        return

    day_dir = source_dir.parent
    cp_dir = _cp_output_dir(day_dir)
    source_repo_id = f"local/{source_dir.name}"

    print(
        f"\n  Source: {day_dir.name}/{source_dir.name} "
        f"({data_type.name}, tag={time_tag}, {len(intervals)} intervals)"
    )

    has_outcomes = any(iv[3] is not None for iv in intervals)

    if has_outcomes:
        for outcome in ("success", "failure"):
            matching = [iv for iv in intervals if iv[3] == outcome]
            if not matching:
                print(f"  [skip] No {outcome} intervals")
                continue
            out_name = f"cp_{outcome}_{time_tag}"
            if _run_extract(source_dir, source_repo_id, cp_dir, out_name, matching, task):
                print(f"  [done] {outcome}: {len(matching)} segments -> {cp_dir / out_name}")

        unlabeled = [iv for iv in intervals if iv[3] is None]
        if unlabeled:
            out_name = f"cp_{time_tag}"
            if _run_extract(source_dir, source_repo_id, cp_dir, out_name, unlabeled, task):
                print(f"  [done] unlabeled: {len(unlabeled)} segments -> {cp_dir / out_name}")
    else:
        out_name = f"cp_{time_tag}"
        if _run_extract(source_dir, source_repo_id, cp_dir, out_name, intervals, task):
            print(f"  [done] all: {len(intervals)} segments -> {cp_dir / out_name}")


def main():
    parser = argparse.ArgumentParser(description="Extract CP success/fail datasets from recorded data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Date folder name (e.g. 0414_rlt_hil_wo_prefix)")
    group.add_argument("--all", action="store_true", help="Process all date folders with CP annotations")
    group.add_argument("--dataset-dir", type=Path, help="Path to a specific dataset directory")
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve")
    parser.add_argument("--setup-json", default=None, help="Path to setup.json for dataset root")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset_dir:
        if not args.dataset_dir.exists():
            print(f"Error: {args.dataset_dir} does not exist")
            sys.exit(1)
        extract_from_dataset(args.dataset_dir, args.task)
        return

    setup = load_setup_json(args.setup_json)
    ds_root_base = resolve_dataset_root(setup)

    if not ds_root_base.exists():
        print(f"Error: Dataset root {ds_root_base} does not exist")
        sys.exit(1)

    if args.all:
        datasets = find_all_cp_datasets(ds_root_base)
    else:
        day_dir = ds_root_base / args.date
        if not day_dir.exists():
            print(f"Error: Date folder {day_dir} does not exist")
            sys.exit(1)
        datasets = find_cp_datasets_in_dir(day_dir)

    if not datasets:
        print("No CP-annotated datasets found.")
        sys.exit(1)

    print(f"Found {len(datasets)} dataset(s) to process:")
    for ds in datasets:
        extract_from_dataset(ds, args.task)

    print("\nDone.")


if __name__ == "__main__":
    main()
