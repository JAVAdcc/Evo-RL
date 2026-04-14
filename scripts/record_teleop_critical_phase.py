"""Record pure-teleop critical phase episodes.

Teleop-only HIL variant of `record_rlt_hil_wo_prefix.py` with **no VLA
and no RL inference**. The leader arms drive the follower arms the
entire time; the only episode-control input is the r key.

Lifecycle per episode:

  * Before first r: teleop drives, frames are NOT recorded
  * First r      : start writing frames to the dataset (critical phase)
  * Second r     : end the episode, mark success
                   (two r presses inside `--double-tap-window-s` = failure)
  * After end    : back to teleop, repeat for next episode

Keyboard controls during recording:
    r     - Pre-episode: start recording an episode
            During episode: end it (single press = success;
                             second press within window = failure)
    →     - End current episode without an outcome label
    ←     - Discard and re-record episode
    ESC   - Stop all recording

No SPACE intervention, no s/f explicit label keys, no policy loading.

Recorded annotation schema:
    complementary_info.is_intervention    - always 0 in this mode
    complementary_info.phase              - 0 before r / 1 between r presses
    complementary_info.collector_policy_id - 0 = human (teleop)

Episode metadata:
    episode_success                       - success / failure (from r state machine)
    rl_intervals                          - one interval per episode:
                                            start=0, end=<ep_len>, outcome=<success|failure>

Usage (on zhaobo-4090-1):
    cd ~/code/hsy/Evo-RL
    conda activate evo-rl
    PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/record_teleop_critical_phase.py --num-episodes 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dataset.setup_helpers import (
    get_sorted_followers,
    get_sorted_leaders,
    load_setup_json,
    resolve_dataset_root,
)

log = logging.getLogger(__name__)

# SO101 bilateral: 6 joints per arm × 2 = 12 DOF
_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record pure-teleop critical phase episodes (r key only)")
    p.add_argument("--task", type=str, default="Insert the copper screw into the black sleeve.")
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--episode-time-s", type=int, default=3000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--setup-json", default=None)
    p.add_argument("--dataset-tag", default="teleop_cp")
    p.add_argument("--vcodec", default="h264")
    p.add_argument("--double-tap-window-s", type=float, default=0.6,
                    help="Window after first r-end press inside which a second r tap marks the episode as failure")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_camera_configs(cameras: list[dict]) -> tuple[dict, dict]:
    """Split cameras into left/right dicts for BiSOFollower."""
    CAM_RENAME = {"left_wrist": "wrist", "right_wrist": "wrist", "top": "front"}
    LEFT_CAMS = {"left_wrist"}
    RIGHT_CAMS = {"right_wrist", "top"}

    left_cameras, right_cameras = {}, {}
    for cam in cameras:
        alias = cam["alias"]
        new_name = CAM_RENAME.get(alias, alias)
        cam_cfg = {
            "type": "opencv",
            "index_or_path": cam["port"],
            "width": cam.get("width", 640),
            "height": cam.get("height", 480),
            "fps": cam.get("fps", 30),
        }
        if cam.get("fourcc"):
            cam_cfg["fourcc"] = cam["fourcc"]
        if alias in LEFT_CAMS:
            left_cameras[new_name] = cam_cfg
        elif alias in RIGHT_CAMS:
            right_cameras[new_name] = cam_cfg
    return left_cameras, right_cameras


def main():
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    setup = load_setup_json(args.setup_json)
    followers = get_sorted_followers(setup)
    leaders = get_sorted_leaders(setup)

    if len(followers) < 2:
        log.error("Need at least 2 follower arms, got %d", len(followers))
        sys.exit(1)
    if len(leaders) < 2:
        log.error("Need at least 2 leader arms for teleop, got %d", len(leaders))
        sys.exit(1)

    now = datetime.now()
    date_folder = now.strftime("%m%d") + f"_{args.dataset_tag}"
    time_tag = now.strftime("%H%M%S")
    dataset_leaf = f"teleop_cp_{time_tag}"

    day_dir = resolve_dataset_root(setup) / date_folder
    day_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = day_dir / dataset_leaf
    dataset_name = f"local/{dataset_leaf}"

    log_file = day_dir / f"{dataset_leaf}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    log.info("=== record_teleop_critical_phase started ===")
    log.info("Args: %s", vars(args))
    log.info("Dataset: %s -> %s", dataset_name, dataset_root)

    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)

    left_cameras, right_cameras = _build_camera_configs(setup.get("cameras", []))

    teleop_id = "bimanual_leader"
    teleop_argv = [
        "--teleop.type=bi_so_leader",
        f"--teleop.left_arm_config.port={leaders[0]['port']}",
        "--teleop.left_arm_config.use_degrees=true",
        f"--teleop.right_arm_config.port={leaders[1]['port']}",
        "--teleop.right_arm_config.use_degrees=true",
        f"--teleop.id={teleop_id}",
    ]
    log.info("Teleop: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])

    with TemporaryDirectory(prefix="teleop-cp-") as cal_dir:
        # Follower calibration
        for side, arm in [("left", followers[0]), ("right", followers[1])]:
            serial = Path(arm["calibration_dir"]).name
            src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
            dst = Path(cal_dir) / f"bimanual_{side}.json"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                log.warning("Calibration file not found: %s", src)

        # Leader calibration (BiSOLeader looks for {teleop_id}_{side}.json)
        leader_cal_dir = TemporaryDirectory(prefix="teleop-cp-leader-cal-")
        for side, arm in [("left", leaders[0]), ("right", leaders[1])]:
            serial = Path(arm["calibration_dir"]).name
            src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
            dst = Path(leader_cal_dir.name) / f"{teleop_id}_{side}.json"
            if src.exists():
                shutil.copy2(src, dst)
                log.info("Leader calibration staged: %s -> %s", src, dst)
            else:
                log.warning("Leader calibration file not found: %s", src)
        teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")

        sys.argv = [
            "record_teleop_critical_phase",
            "--robot.type=bi_so_follower",
            "--robot.id=bimanual",
            f"--robot.calibration_dir={cal_dir}",
            f"--robot.left_arm_config.port={followers[0]['port']}",
            "--robot.left_arm_config.use_degrees=true",
            f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
            f"--robot.right_arm_config.port={followers[1]['port']}",
            "--robot.right_arm_config.use_degrees=true",
            f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
            *teleop_argv,
            f"--dataset.repo_id={dataset_name}",
            f"--dataset.root={dataset_root}",
            f"--dataset.single_task={args.task}",
            f"--dataset.num_episodes={args.num_episodes}",
            f"--dataset.episode_time_s={args.episode_time_s}",
            f"--dataset.fps={args.fps}",
            f"--dataset.vcodec={args.vcodec}",
            "--dataset.push_to_hub=false",
            # Defer video encoding to end-of-recording
            f"--dataset.video_encoding_batch_size={args.num_episodes + 1}",
            # Pure-teleop r-key-driven episode mode
            "--teleop_r_key_episodes=true",
            f"--rlt.rl_phase_double_tap_window_s={args.double_tap_window_s}",
            # Resolve episode_success from the r-key state machine outcome
            "--enable_episode_outcome_labeling=true",
            # Teleop-only: SPACE intervention is irrelevant (no policy to override)
            "--intervention_state_machine_enabled=false",
            "--play_sounds=true",
        ]

        log.info("Calling record() with %d argv entries", len(sys.argv))
        print(f"\nDataset: {dataset_name} -> {dataset_root}")
        print(f"Log: {log_file}")
        print(
            "Teleop critical-phase mode: r=start episode; "
            "r again=end success; "
            f"r+r within {args.double_tap_window_s:.1f}s=end failure"
        )
        print()

        from lerobot.scripts.lerobot_record import record
        record()

    leader_cal_dir.cleanup()

    # Post-record: stamp every frame as is_intervention=1 for teleop-only data.
    # The recording loop writes is_intervention=0 because the intervention state
    # machine is disabled in this mode; semantically every frame here is human
    # action, so we rewrite the column in place before exit.
    from scripts.dataset.fix_teleop_is_intervention import rewrite_dataset
    if dataset_root.exists() and (dataset_root / "meta" / "info.json").exists():
        stats = rewrite_dataset(dataset_root)
        log.info(
            "Post-record is_intervention rewrite: %d/%d parquet rows, %d jsonl lines",
            stats["parquet_changed"], stats["parquet_rows"], stats["jsonl_changed"],
        )

    log.info("=== record_teleop_critical_phase finished ===")


if __name__ == "__main__":
    main()
