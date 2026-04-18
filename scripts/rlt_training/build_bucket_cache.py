#!/usr/bin/env python
"""Build a 3-bucket chunk-transition cache for RLT online-offline training.

For each bucket:
  - warmup_vla:   ref_chunk = VLA-proposed action (no replacement)
  - human_expert: ref_chunk = exec_chunk (always, teleop-only dataset)
  - rl_rollout:   ref_chunk = exec_chunk only for chunks where the dominant
                  per-frame source is human intervention (chunk-level vote,
                  matching online_collector.py semantics)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import configure_logging, load_training_config

logger = configure_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-dataset-path", default=None)
    parser.add_argument("--transition-cache-dir", required=True)
    parser.add_argument("--bucket-mode", choices=["warmup_vla", "human_expert", "rl_rollout"], required=True)
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument("--rl-token-checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mark-critical", action="store_true")
    parser.add_argument("--token-pool-size", type=int, default=64)
    parser.add_argument("--task-instruction", default="Insert the copper screw into the black sleeve.")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--repo-id", default="rlt_demo")
    return parser.parse_args()


def _collect_intervention_array(dataset) -> torch.Tensor | None:
    """Return (N,) float tensor of is_intervention flags if present, else None."""
    hf = getattr(dataset._dataset, "hf_dataset", None)
    if hf is None:
        return None
    cols = getattr(hf, "column_names", [])
    key = "complementary_info.is_intervention"
    if key not in cols:
        return None
    raw = hf[key]
    return torch.tensor([float(v) for v in raw], dtype=torch.float32)


def _dominant_is_human(interventions: torch.Tensor, start: int, length: int) -> bool:
    """Return True when human frames win the majority vote over [start, start+length)."""
    window = interventions[start : start + length]
    human = int((window > 0.5).sum().item())
    return human >= (length - human)  # human wins ties, matches online_collector rule


def main() -> None:
    args = parse_args()

    from common import build_pi05_policy
    from lerobot.rlt.demo_loader import RLTDemoDataset, rlt_demo_collate
    from lerobot.rlt.offline_dataset import (
        _count_episodes,
        _episode_frame_range,
        build_overlap_frame_indices,
        build_transitions_from_demos,
        save_transition_cache,
        split_episode_indices,
    )

    config = load_training_config(args.config)
    config.offline_rl.frame_stride = args.frame_stride
    config.offline_rl.train_ratio = args.train_ratio
    config.offline_rl.val_ratio = args.val_ratio

    logger.info("Loading pi0.5 from %s", args.model_path)
    policy = build_pi05_policy(
        config=config,
        model_path=args.model_path,
        task_instruction=args.task_instruction,
        device=args.device,
        token_pool_size=args.token_pool_size,
        dtype=args.dtype,
        rl_token_checkpoint=args.rl_token_checkpoint,
    )
    policy.freeze_vla()
    policy.freeze_rl_token_encoder()
    policy.eval()

    dataset = RLTDemoDataset(
        dataset_path=args.demo_dataset_path,
        repo_id=args.repo_id,
        chunk_length=config.vla_horizon,
        normalize_actions=True,
    )
    num_episodes = _count_episodes(dataset)
    splits = split_episode_indices(
        num_episodes,
        train_ratio=config.offline_rl.train_ratio,
        val_ratio=config.offline_rl.val_ratio,
        seed=config.seed,
    )
    logger.info(
        "[%s] %d episodes -> train=%d val=%d test=%d",
        args.bucket_mode, num_episodes,
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    interventions: torch.Tensor | None = None
    if args.bucket_mode == "rl_rollout":
        interventions = _collect_intervention_array(dataset)
        if interventions is None:
            raise RuntimeError("rl_rollout bucket requires complementary_info.is_intervention column")
        logger.info(
            "rl_rollout: loaded %d intervention flags, human_frac=%.2f",
            interventions.numel(), float(interventions.mean()),
        )

    total_start = time.time()
    for split_name, episode_ids in splits.items():
        if not episode_ids:
            continue

        split_start = time.time()
        transitions = []
        n_ref_replaced = 0
        n_total = 0
        for ep_index, episode_id in enumerate(sorted(episode_ids), start=1):
            frame_start, frame_stop = _episode_frame_range(dataset, episode_id)
            frame_indices = build_overlap_frame_indices(
                episode_start=frame_start,
                episode_stop=frame_stop,
                chunk_length=config.chunk_length,
                stride=config.offline_rl.frame_stride,
            )
            loader = DataLoader(
                Subset(dataset, frame_indices),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=rlt_demo_collate,
                num_workers=0,
                drop_last=False,
            )
            ep_transitions = build_transitions_from_demos(
                policy=policy,
                demo_loader=loader,
                frame_indices=frame_indices,
                episode_last_frame=frame_stop - 1,
                chunk_length=config.chunk_length,
                device=args.device,
                episode_id=episode_id,
                is_critical=float(args.mark_critical),
                stride=config.offline_rl.frame_stride,
                source=_bucket_source_id(args.bucket_mode),
            )

            start_anchors = [
                f for f in sorted(frame_indices)
                if f + config.chunk_length <= frame_stop - 1
            ]
            if len(start_anchors) != len(ep_transitions):
                raise RuntimeError(
                    f"Episode {episode_id}: start_anchors={len(start_anchors)} "
                    f"vs transitions={len(ep_transitions)}"
                )

            replaced_flags = [False] * len(ep_transitions)
            for tr_idx, t in enumerate(ep_transitions):
                n_total += 1
                if args.bucket_mode == "human_expert":
                    t.ref_chunk = t.exec_chunk.clone()
                    t.intervention = torch.tensor(1.0)
                    replaced_flags[tr_idx] = True
                    n_ref_replaced += 1
                elif args.bucket_mode == "rl_rollout":
                    start_frame = start_anchors[tr_idx]
                    if _dominant_is_human(interventions, start_frame, config.chunk_length):
                        t.ref_chunk = t.exec_chunk.clone()
                        t.intervention = torch.tensor(1.0)
                        replaced_flags[tr_idx] = True
                        n_ref_replaced += 1

            # Second pass: propagate next_ref_chunk using the anchor map.
            anchor_to_idx = {a: i for i, a in enumerate(start_anchors)}
            for tr_idx, t in enumerate(ep_transitions):
                next_anchor = start_anchors[tr_idx] + config.chunk_length
                nxt = anchor_to_idx.get(next_anchor)
                if nxt is not None and replaced_flags[nxt]:
                    t.next_ref_chunk = ep_transitions[nxt].ref_chunk.clone()

            transitions.extend(ep_transitions)
            if ep_index % 20 == 0:
                logger.info(
                    "[%s/%s] %d/%d ep, %d transitions, %d ref-replaced",
                    args.bucket_mode, split_name, ep_index, len(episode_ids),
                    len(transitions), n_ref_replaced,
                )

        save_transition_cache(transitions, args.transition_cache_dir, split_name)
        logger.info(
            "[%s/%s] split done: %d transitions, %d ref-replaced (%.1f%%), %.1fs",
            args.bucket_mode, split_name, len(transitions), n_ref_replaced,
            100.0 * n_ref_replaced / max(n_total, 1),
            time.time() - split_start,
        )

    logger.info(
        "[%s] total cache built at %s in %.1fs",
        args.bucket_mode, args.transition_cache_dir, time.time() - total_start,
    )


def _bucket_source_id(mode: str) -> int:
    return {"warmup_vla": 1, "human_expert": 3, "rl_rollout": 2}[mode]


if __name__ == "__main__":
    main()
