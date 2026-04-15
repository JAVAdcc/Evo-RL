#!/usr/bin/env python
"""Offline-mode RLT training over a 3-bucket weighted mix of transitions.

Loads bucket caches (warmup_vla, human_expert, rl_rollout), warm-starts the
actor-critic from an existing AC checkpoint, freezes the RL-token encoder, and
runs offline_rl_loop on a WeightedMixReplayBuffer sampled per user-specified
probabilities.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import configure_logging, load_training_config
from train_chunk_actor_critic import create_algorithm_with_cached_transitions

logger = configure_logging(__name__)


DEFAULT_BUCKET_PATHS = {
    "warmup_vla":   "outputs/cache_bucket1_warmup_v2",
    "human_expert": "outputs/cache_bucket2_human",
    "rl_rollout":   "outputs/cache_bucket3_rl",
}

DEFAULT_PROBS = {"warmup_vla": 0.2, "human_expert": 0.4, "rl_rollout": 0.4}


@dataclass
class ExperimentConfig:
    name: str
    probs: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PROBS))
    actor_hidden: int = 1024
    actor_layers: int = 3
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = True
    actor_lr: float = 3e-4
    ref_dropout_p: float = 0.5
    fixed_std: float = 0.05
    critic_hidden: int = 1024
    critic_layers: int = 3
    critic_activation: str = "relu"
    critic_layer_norm: bool = False
    critic_residual: bool = True
    critic_lr: float = 3e-4
    beta: float = 5.0
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    actor_update_interval: int = 2
    gradient_steps: int = 30000
    warm_start_ckpt: str | None = "outputs/ac_0412_278cp_refdrop05/rl_checkpoint.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/lerobot/rlt/configs/pi05_rlt.yaml")
    parser.add_argument("--rl-token-checkpoint", default="outputs/rlt_demo_adapt_271ep_sft_fp32/demo_adapt_checkpoint.pt")
    parser.add_argument("--output-dir", default="outputs/online_offline_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sweep-file", default=None, help="JSON list of experiment dicts; if omitted, run a default sweep")
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--bucket-warmup-vla-path", default=DEFAULT_BUCKET_PATHS["warmup_vla"])
    parser.add_argument("--bucket-human-expert-path", default=DEFAULT_BUCKET_PATHS["human_expert"])
    parser.add_argument("--bucket-rl-rollout-path", default=DEFAULT_BUCKET_PATHS["rl_rollout"])
    return parser.parse_args()


def apply_experiment(config, exp: ExperimentConfig) -> None:
    config.actor.hidden_dim = exp.actor_hidden
    config.actor.num_layers = exp.actor_layers
    config.actor.activation = exp.actor_activation
    config.actor.layer_norm = exp.actor_layer_norm
    config.actor.residual = exp.actor_residual
    config.actor.lr = exp.actor_lr
    config.actor.ref_dropout_p = exp.ref_dropout_p
    config.actor.fixed_std = exp.fixed_std
    config.critic.hidden_dim = exp.critic_hidden
    config.critic.num_layers = exp.critic_layers
    config.critic.activation = exp.critic_activation
    config.critic.layer_norm = exp.critic_layer_norm
    config.critic.residual = exp.critic_residual
    config.critic.lr = exp.critic_lr
    config.training.beta = exp.beta
    config.training.gamma = exp.gamma
    config.training.tau = exp.tau
    config.training.batch_size = exp.batch_size
    config.training.actor_update_interval = exp.actor_update_interval
    config.offline_rl.num_gradient_steps = exp.gradient_steps


def load_warm_start(algorithm, checkpoint_path: str, device: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    algorithm.policy.actor.load_state_dict(ckpt["actor_state_dict"])
    algorithm.critic.load_state_dict(ckpt["critic_state_dict"])
    algorithm.target_critic.load_state_dict(ckpt["target_critic_state_dict"])
    logger.info("Warm-started actor/critic from %s", checkpoint_path)


def build_mix_buffers(bucket_paths: dict[str, str], probs: dict[str, float], config):
    from lerobot.rlt.mix_dataset import WeightedMixReplayBuffer, load_bucket_cache

    train_buckets, val_buckets = {}, {}
    for name, path in bucket_paths.items():
        train_buckets[name] = load_bucket_cache(path, "train", capacity=config.replay.capacity)
        val_buckets[name] = load_bucket_cache(path, "val", capacity=config.replay.capacity)

    train_mix = WeightedMixReplayBuffer(train_buckets, probs)
    val_mix = WeightedMixReplayBuffer(val_buckets, probs)
    return train_mix, val_mix, train_buckets, val_buckets


def evaluate_per_bucket(algorithm, config, val_buckets, batches: int) -> dict[str, dict]:
    from lerobot.rlt.evaluator import evaluate_offline

    out = {}
    for name, buf in val_buckets.items():
        if len(buf) == 0:
            continue
        m = evaluate_offline(algorithm, buf, config, num_batches=batches)
        out[name] = {
            "ref_mse": float(m.ref_action_mse),
            "expert_mse": float(m.expert_action_mse),
            "q_gap": float(m.q_gap),
            "mean_q_policy": float(m.mean_q_policy),
            "mean_q_expert": float(m.mean_q_expert),
            "td_error": float(m.mean_critic_td_error),
        }
    return out


def run_experiment(
    exp: ExperimentConfig,
    args: argparse.Namespace,
    bucket_paths: dict[str, str],
    results_file: Path,
) -> dict:
    from lerobot.rlt.trainer import offline_rl_loop

    logger.info("=" * 80)
    logger.info("EXPERIMENT %s", exp.name)
    logger.info("config: %s", json.dumps(exp.__dict__, default=str))

    config = load_training_config(args.config)
    apply_experiment(config, exp)
    config.offline_rl.eval_every = args.eval_every
    config.offline_rl.save_every = 10**9  # disable mid-run save, final save only
    config.offline_rl.log_every = args.log_every

    train_mix, val_mix, train_buckets, val_buckets = build_mix_buffers(
        bucket_paths, exp.probs, config,
    )
    logger.info(
        "Mix sizes: train=%s val=%s probs=%s",
        {k: len(v) for k, v in train_buckets.items()},
        {k: len(v) for k, v in val_buckets.items()},
        exp.probs,
    )

    algorithm = create_algorithm_with_cached_transitions(config, args.rl_token_checkpoint, args.device)
    if exp.warm_start_ckpt is not None:
        load_warm_start(algorithm, exp.warm_start_ckpt, args.device)

    actor_opt = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=config.actor.lr)
    critic_opt = torch.optim.Adam(algorithm.critic.parameters(), lr=config.critic.lr)

    save_dir = Path(args.output_dir) / exp.name
    start = time.time()
    metrics = offline_rl_loop(
        algorithm=algorithm,
        config=config,
        replay_buffer=train_mix,
        val_buffer=val_mix,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        save_dir=str(save_dir),
    )
    elapsed = time.time() - start

    per_bucket_eval = evaluate_per_bucket(algorithm, config, val_buckets, args.eval_batches)
    mix_eval = {}
    if len(val_mix) > 0:
        from lerobot.rlt.evaluator import evaluate_offline
        m = evaluate_offline(algorithm, val_mix, config, num_batches=args.eval_batches)
        mix_eval = {
            "ref_mse": float(m.ref_action_mse),
            "expert_mse": float(m.expert_action_mse),
            "q_gap": float(m.q_gap),
            "td_error": float(m.mean_critic_td_error),
        }

    result = {
        "name": exp.name,
        "config": exp.__dict__,
        "elapsed_sec": elapsed,
        "final_actor_loss": _mean_tail(metrics.actor_losses, 500),
        "final_critic_loss": _mean_tail(metrics.critic_losses, 500),
        "mix_eval": mix_eval,
        "per_bucket_eval": per_bucket_eval,
        "save_dir": str(save_dir),
    }
    _append_result(results_file, result)
    logger.info(
        "[%s] DONE elapsed=%.1fs mix_ref_mse=%.6f mix_q_gap=%.6f",
        exp.name, elapsed,
        mix_eval.get("ref_mse", float("nan")),
        mix_eval.get("q_gap", float("nan")),
    )
    del algorithm, actor_opt, critic_opt
    torch.cuda.empty_cache()
    return result


def _mean_tail(values, k: int) -> float | None:
    if not values:
        return None
    tail = values[-k:]
    return sum(tail) / len(tail)


def _append_result(results_file: Path, result: dict) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if results_file.exists():
        existing = json.loads(results_file.read_text())
    existing.append(result)
    results_file.write_text(json.dumps(existing, indent=2, default=str))


def default_sweep() -> list[ExperimentConfig]:
    exps: list[ExperimentConfig] = []

    # === Block A0: baseline eval (0 training steps, just warm-start eval) ===
    exps.append(ExperimentConfig(
        name="A0_baseline_eval",
        probs={"warmup_vla": 0.34, "human_expert": 0.33, "rl_rollout": 0.33},
        gradient_steps=0,
    ))

    # === Block A: mixture ratio @ beta=5 ===
    for name, probs in [
        ("A1_all_warmup",       {"warmup_vla": 1.0, "human_expert": 0.0, "rl_rollout": 0.0}),
        ("A2_mix_equal",        {"warmup_vla": 0.33, "human_expert": 0.33, "rl_rollout": 0.34}),
        ("A3_mix_doc",          {"warmup_vla": 0.2, "human_expert": 0.4, "rl_rollout": 0.4}),
        ("A4_mix_human_heavy",  {"warmup_vla": 0.1, "human_expert": 0.6, "rl_rollout": 0.3}),
        ("A5_mix_rl_heavy",     {"warmup_vla": 0.1, "human_expert": 0.3, "rl_rollout": 0.6}),
        ("A6_no_warmup",        {"warmup_vla": 0.0, "human_expert": 0.5, "rl_rollout": 0.5}),
        ("A7_warmup_plus_rl",   {"warmup_vla": 0.5, "human_expert": 0.0, "rl_rollout": 0.5}),
        ("A8_warmup_plus_human",{"warmup_vla": 0.5, "human_expert": 0.5, "rl_rollout": 0.0}),
    ]:
        exps.append(ExperimentConfig(name=name, probs=probs))

    # === Block B: beta sweep at doc ratio ===
    for beta in [0.3, 1.0, 2.0, 10.0]:
        exps.append(ExperimentConfig(
            name=f"B_beta_{beta}", beta=beta,
            probs={"warmup_vla": 0.2, "human_expert": 0.4, "rl_rollout": 0.4},
        ))

    # === Block D: actor LR sweep (lower LR for fine-tune) ===
    for lr in [1e-4, 3e-5]:
        exps.append(ExperimentConfig(
            name=f"D_actor_lr_{lr}", actor_lr=lr, critic_lr=lr,
            probs={"warmup_vla": 0.2, "human_expert": 0.4, "rl_rollout": 0.4},
        ))

    # === Block E: longer training with best-guess mix ===
    exps.append(ExperimentConfig(
        name="E_long_doc_mix",
        probs={"warmup_vla": 0.2, "human_expert": 0.4, "rl_rollout": 0.4},
        gradient_steps=50000,
    ))

    return exps


def main() -> None:
    args = parse_args()
    bucket_paths = {
        "warmup_vla": args.bucket_warmup_vla_path,
        "human_expert": args.bucket_human_expert_path,
        "rl_rollout": args.bucket_rl_rollout_path,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"

    if args.sweep_file is not None:
        raw = json.loads(Path(args.sweep_file).read_text())
        experiments = [ExperimentConfig(**d) for d in raw]
    else:
        experiments = default_sweep()

    completed = set()
    if results_file.exists():
        for r in json.loads(results_file.read_text()):
            completed.add(r["name"])

    for exp in experiments:
        if exp.name in completed:
            logger.info("skip %s (already completed)", exp.name)
            continue
        run_experiment(exp, args, bucket_paths, results_file)


if __name__ == "__main__":
    main()
