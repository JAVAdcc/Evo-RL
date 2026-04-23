#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import configure_logging, load_training_config
from train_chunk_actor_critic import create_algorithm_with_cached_transitions
from train_online_offline import build_mix_buffers, load_warm_start

from lerobot.rlt.evaluator import evaluate_offline
from lerobot.rlt.losses import discounted_chunk_return
from lerobot.rlt.trainer import TrainingMetrics, _save_rl_checkpoint
from lerobot.rlt.utils import infer_actor_architecture

logger = configure_logging(__name__)


DEFAULT_BUCKET_PATHS = {
    "cp405": "outputs/cache_cp405_post_p0",
    "teleop141": "outputs/cache_teleop141_renorm_cp405",
    "intervene156": "outputs/cache_intervene156_post_p0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase A critic warm-start with cached next_ref bootstrap and frozen actor.",
    )
    parser.add_argument("--name", default="w_reduced_cp405_refboot_phase_a")
    parser.add_argument("--config", default="src/lerobot/rlt/configs/pi05_rlt.yaml")
    parser.add_argument(
        "--rl-token-checkpoint",
        default="outputs/rlt_demo_adapt_271ep_sft_fp32/demo_adapt_checkpoint.pt",
    )
    parser.add_argument(
        "--warm-start-ckpt",
        default="outputs/qviz_replay_20260422/w_reduced_cp405_replay/rl_checkpoint.pt",
    )
    parser.add_argument("--output-dir", default="outputs/phase_a_ref_bootstrap")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--bucket-cp405-path", default=DEFAULT_BUCKET_PATHS["cp405"])
    parser.add_argument("--bucket-teleop-path", default=DEFAULT_BUCKET_PATHS["teleop141"])
    parser.add_argument("--bucket-intervene-path", default=DEFAULT_BUCKET_PATHS["intervene156"])
    parser.add_argument("--prob-cp405", type=float, default=0.30)
    parser.add_argument("--prob-teleop", type=float, default=0.35)
    parser.add_argument("--prob-intervene", type=float, default=0.35)
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace) -> None:
    config.offline_rl.num_gradient_steps = args.gradient_steps
    config.offline_rl.eval_every = args.eval_every
    config.offline_rl.log_every = args.log_every
    config.offline_rl.save_every = 10**9
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.critic_lr is not None:
        config.critic.lr = args.critic_lr
    if args.gamma is not None:
        config.training.gamma = args.gamma
    if args.tau is not None:
        config.training.tau = args.tau


def infer_block_count(prefix: str, state_dict: dict[str, torch.Tensor]) -> int:
    block_indices = set()
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix):]
        block_id = remainder.split(".", 1)[0]
        if block_id.isdigit():
            block_indices.add(int(block_id))
    return (max(block_indices) + 1) if block_indices else 0


def infer_critic_architecture(
    critic_state_dict: dict[str, torch.Tensor],
    *,
    default_activation: str = "relu",
) -> dict[str, int | bool | str]:
    q1_keys = {k.removeprefix("q1."): v for k, v in critic_state_dict.items() if k.startswith("q1.")}
    if "net.input_proj.weight" in q1_keys:
        hidden_dim = q1_keys["net.input_proj.weight"].shape[0]
        block_indices = {
            int(k.split(".")[2])
            for k in q1_keys
            if k.startswith("net.blocks.") and k.endswith(".0.weight")
        }
        layer_norm = any(k.startswith("net.blocks.") and k.endswith(".1.weight") for k in q1_keys)
        return {
            "hidden_dim": hidden_dim,
            "num_layers": len(block_indices),
            "activation": default_activation,
            "layer_norm": layer_norm,
            "residual": True,
        }

    linear_keys = sorted(
        k for k, v in q1_keys.items()
        if k.startswith("net.") and k.endswith(".weight") and v.ndim == 2
    )
    if not linear_keys:
        raise ValueError("Could not infer critic architecture from warm-start checkpoint")

    hidden_dim = q1_keys[linear_keys[0]].shape[0]
    layer_norm = any(
        k.startswith("net.") and k.endswith(".weight") and v.ndim == 1
        for k, v in q1_keys.items()
    )
    return {
        "hidden_dim": hidden_dim,
        "num_layers": len(linear_keys) - 1,
        "activation": default_activation,
        "layer_norm": layer_norm,
        "residual": False,
    }


def apply_warm_start_architecture(config, warm_start_ckpt: str) -> dict[str, object]:
    ckpt = torch.load(warm_start_ckpt, map_location="cpu", weights_only=False)
    actor_state = ckpt["actor_state_dict"]
    critic_state = ckpt["critic_state_dict"]
    metrics_path = Path(warm_start_ckpt).with_name("metrics.json")
    metrics_cfg = {}
    if metrics_path.exists():
        metrics_cfg = json.loads(metrics_path.read_text()).get("config", {})

    actor_arch = infer_actor_architecture(
        actor_state,
        default_activation=metrics_cfg.get("actor_activation", config.actor.activation),
        default_fixed_std=metrics_cfg.get("fixed_std", config.actor.fixed_std),
        default_ref_dropout_p=metrics_cfg.get("ref_dropout_p", config.actor.ref_dropout_p),
    )
    critic_arch = infer_critic_architecture(
        critic_state,
        default_activation=metrics_cfg.get("critic_activation", metrics_cfg.get("actor_activation", config.critic.activation)),
    )

    config.actor.hidden_dim = int(actor_arch["hidden_dim"])
    config.actor.num_layers = int(actor_arch["num_layers"])
    config.actor.residual = bool(actor_arch["residual"])
    config.actor.activation = str(metrics_cfg.get("actor_activation", actor_arch["activation"]))
    config.actor.layer_norm = bool(metrics_cfg.get("actor_layer_norm", actor_arch["layer_norm"]))
    config.actor.fixed_std = float(metrics_cfg.get("fixed_std", actor_arch["fixed_std"]))
    config.actor.ref_dropout_p = float(metrics_cfg.get("ref_dropout_p", actor_arch["ref_dropout_p"]))

    config.critic.hidden_dim = int(critic_arch["hidden_dim"])
    config.critic.num_layers = int(critic_arch["num_layers"])
    config.critic.residual = bool(critic_arch["residual"])
    config.critic.activation = str(metrics_cfg.get("critic_activation", critic_arch["activation"]))
    config.critic.layer_norm = bool(metrics_cfg.get("critic_layer_norm", critic_arch["layer_norm"]))

    return {
        "actor_hidden": config.actor.hidden_dim,
        "actor_layers": config.actor.num_layers,
        "actor_residual": config.actor.residual,
        "actor_activation": config.actor.activation,
        "actor_layer_norm": config.actor.layer_norm,
        "actor_fixed_std": config.actor.fixed_std,
        "actor_ref_dropout_p": config.actor.ref_dropout_p,
        "critic_hidden": config.critic.hidden_dim,
        "critic_layers": config.critic.num_layers,
        "critic_residual": config.critic.residual,
        "critic_activation": config.critic.activation,
        "critic_layer_norm": config.critic.layer_norm,
        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
    }


def build_metrics_sidecar(config, warm_start_arch: dict[str, object], metadata: dict | None = None) -> dict[str, object]:
    payload = {
        "actor_hidden": int(config.actor.hidden_dim),
        "actor_layers": int(config.actor.num_layers),
        "actor_activation": str(config.actor.activation),
        "actor_layer_norm": bool(config.actor.layer_norm),
        "actor_residual": bool(config.actor.residual),
        "fixed_std": float(config.actor.fixed_std),
        "ref_dropout_p": float(config.actor.ref_dropout_p),
        "critic_hidden": int(config.critic.hidden_dim),
        "critic_layers": int(config.critic.num_layers),
        "critic_activation": str(config.critic.activation),
        "critic_layer_norm": bool(config.critic.layer_norm),
        "critic_residual": bool(config.critic.residual),
        "bootstrap_mode": "cached_next_ref",
        "actor_frozen": True,
        "warm_start_architecture": warm_start_arch,
    }
    if metadata:
        payload["metadata"] = metadata
    return {"config": payload}


def set_phase_a_train_mode(algorithm) -> None:
    algorithm.train()
    algorithm.policy.actor.eval()


def run_dual_eval(algorithm, buffer, config, num_batches: int) -> tuple[dict[str, float], dict[str, float]]:
    phase_a_eval = evaluate_offline_ref_bootstrap(algorithm, buffer, config, num_batches=num_batches)
    standard_eval = summarize_standard_eval(
        evaluate_offline(algorithm, buffer, config, num_batches=num_batches),
    )
    return phase_a_eval, standard_eval


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def mean_tail(values: list[float], k: int) -> float | None:
    if not values:
        return None
    tail = values[-k:]
    return sum(tail) / len(tail)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def append_worklog(path: Path, heading: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"\n## {heading}\n")
        for line in lines:
            f.write(f"- {line}\n")


def critic_loss_ref_bootstrap(
    critic,
    target_critic,
    batch: dict[str, torch.Tensor],
    gamma: float,
    C: int,
) -> torch.Tensor:
    x = batch["state_vec"]
    a = batch["exec_chunk_flat"]
    x_next = batch["next_state_vec"]
    ref_next = batch["next_ref_flat"]
    reward_seq = batch["reward_seq"]
    done = batch["done"]
    actual_steps = batch.get("actual_steps")

    with torch.no_grad():
        a_next = ref_next.clamp(-1.0, 1.0)
        q_next = target_critic.min_q(x_next, a_next)
        q_next = q_next.clamp(-100.0, 100.0)
        r = discounted_chunk_return(reward_seq, gamma, actual_steps)

        if actual_steps is not None:
            bootstrap_exp = actual_steps.unsqueeze(-1).float()
        else:
            bootstrap_exp = torch.full_like(done.unsqueeze(-1), C, dtype=torch.float32)
        bootstrap = (gamma ** bootstrap_exp) * (1.0 - done.unsqueeze(-1)) * q_next
        target = r + bootstrap

    q1, q2 = critic(x, a)
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def evaluate_offline_ref_bootstrap(algorithm, val_buffer, config, num_batches: int = 10) -> dict[str, float]:
    algorithm.eval()
    policy = algorithm.policy
    device = next(algorithm.parameters()).device

    totals = {
        "expert_mse": 0.0,
        "ref_mse": 0.0,
        "ref_dropped_mse": 0.0,
        "mean_q_policy": 0.0,
        "mean_q_expert": 0.0,
        "mean_q_ref": 0.0,
        "mean_ref_td_error": 0.0,
    }

    for _ in range(num_batches):
        batch = _batch_to_device(val_buffer.sample(config.training.batch_size), device)
        state_vec = batch["state_vec"]
        exec_chunk_flat = batch["exec_chunk_flat"]
        ref_chunk_flat = batch["ref_chunk_flat"]

        with torch.no_grad():
            mu, _ = policy.actor.forward(state_vec, ref_chunk_flat)
            mu_dropped, _ = policy.actor.forward(state_vec, torch.zeros_like(ref_chunk_flat))

            totals["expert_mse"] += F.mse_loss(mu, exec_chunk_flat).item()
            totals["ref_mse"] += F.mse_loss(mu, ref_chunk_flat).item()
            totals["ref_dropped_mse"] += F.mse_loss(mu_dropped, exec_chunk_flat).item()
            totals["mean_q_policy"] += algorithm.critic.min_q(state_vec, mu).mean().item()
            totals["mean_q_expert"] += algorithm.critic.min_q(state_vec, exec_chunk_flat).mean().item()
            totals["mean_q_ref"] += algorithm.critic.min_q(state_vec, ref_chunk_flat.clamp(-1.0, 1.0)).mean().item()
            totals["mean_ref_td_error"] += critic_loss_ref_bootstrap(
                algorithm.critic,
                algorithm.target_critic,
                batch,
                config.training.gamma,
                config.chunk_length,
            ).item()

    n = max(num_batches, 1)
    out = {k: v / n for k, v in totals.items()}
    out["q_gap_policy_expert"] = out["mean_q_policy"] - out["mean_q_expert"]
    out["q_gap_policy_ref"] = out["mean_q_policy"] - out["mean_q_ref"]
    return out


def summarize_standard_eval(metrics) -> dict[str, float]:
    return {
        "expert_mse": float(metrics.expert_action_mse),
        "ref_mse": float(metrics.ref_action_mse),
        "ref_dropped_mse": float(metrics.ref_dropped_mse),
        "mean_q_policy": float(metrics.mean_q_policy),
        "mean_q_expert": float(metrics.mean_q_expert),
        "q_gap": float(metrics.q_gap),
        "mean_critic_td_error": float(metrics.mean_critic_td_error),
    }


def evaluate_per_bucket(algorithm, config, val_buckets, eval_batches: int) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, buf in val_buckets.items():
        if len(buf) == 0:
            continue
        phase_a = evaluate_offline_ref_bootstrap(algorithm, buf, config, num_batches=eval_batches)
        standard = summarize_standard_eval(evaluate_offline(algorithm, buf, config, num_batches=eval_batches))
        out[name] = {"phase_a": phase_a, "standard": standard}
    return out


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    apply_overrides(config, args)
    warm_start_arch = apply_warm_start_architecture(config, args.warm_start_ckpt)

    probs = {
        "cp405": args.prob_cp405,
        "teleop141": args.prob_teleop,
        "intervene156": args.prob_intervene,
    }
    prob_sum = sum(probs.values())
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(f"Sampling probabilities must sum to 1.0, got {prob_sum:.6f}")

    bucket_paths = {
        "cp405": args.bucket_cp405_path,
        "teleop141": args.bucket_teleop_path,
        "intervene156": args.bucket_intervene_path,
    }

    save_dir = Path(args.output_dir) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    worklog_path = save_dir / "worklog.md"
    worklog_path.write_text("# Phase A Worklog\n")

    write_json(
        save_dir / "phase_a_config.json",
        {
            "args": vars(args),
            "bucket_paths": bucket_paths,
            "probs": probs,
            "warm_start_architecture": warm_start_arch,
        },
    )
    append_worklog(
        worklog_path,
        "Kickoff",
        [
            "Objective: freeze actor and re-train critic with cached next_ref bootstrap only.",
            "Completion criteria: final checkpoint, metrics json, eval history, and bucket-wise diagnostics are written.",
            f"Warm start checkpoint: {args.warm_start_ckpt}",
            f"Warm start architecture: {warm_start_arch}",
            f"Bucket paths: {bucket_paths}",
            f"Sampling probabilities: {probs}",
            "Open risks: mixed caches remain heterogeneous; current HTML visualizer is only trusted qualitatively on cp405.",
        ],
    )

    train_mix, val_mix, train_buckets, val_buckets = build_mix_buffers(bucket_paths, probs, config)
    logger.info(
        "Phase A buffers loaded. train=%s val=%s probs=%s",
        {k: len(v) for k, v in train_buckets.items()},
        {k: len(v) for k, v in val_buckets.items()},
        probs,
    )
    append_worklog(
        worklog_path,
        "Buffers Ready",
        [
            f"Train bucket sizes: { {k: len(v) for k, v in train_buckets.items()} }",
            f"Val bucket sizes: { {k: len(v) for k, v in val_buckets.items()} }",
        ],
    )

    algorithm = create_algorithm_with_cached_transitions(config, args.rl_token_checkpoint, args.device)
    load_warm_start(algorithm, args.warm_start_ckpt, args.device)

    for param in algorithm.policy.actor.parameters():
        param.requires_grad_(False)
    set_phase_a_train_mode(algorithm)

    actor_optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=config.actor.lr)
    critic_optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=config.critic.lr)
    write_json(
        save_dir / "metrics.json",
        build_metrics_sidecar(
            config,
            warm_start_arch,
            metadata={
                "phase": "A",
                "bootstrap_mode": "cached_next_ref",
                "actor_frozen": True,
                "warm_start_ckpt": args.warm_start_ckpt,
                "bucket_paths": bucket_paths,
                "probs": probs,
            },
        ),
    )

    metrics = TrainingMetrics()
    eval_history: list[dict] = []
    device = next(algorithm.parameters()).device
    gamma = config.training.gamma
    C = config.chunk_length
    tau = config.training.tau
    batch_size = config.training.batch_size
    utd = config.training.utd_ratio
    start = time.time()

    append_worklog(
        worklog_path,
        "Training Start",
        [
            f"Gradient steps: {config.offline_rl.num_gradient_steps}",
            f"Batch size: {batch_size}",
            f"Gamma: {gamma}",
            f"Tau: {tau}",
            f"UTD ratio: {utd}",
            f"Critic LR: {config.critic.lr}",
            "Actor updates: disabled (frozen actor, no actor optimizer step).",
        ],
    )

    phase_a_eval, standard_eval = run_dual_eval(algorithm, val_mix, config, num_batches=args.eval_batches)
    baseline_record = {
        "step": 0,
        "elapsed_sec": 0.0,
        "train_critic_loss_tail_200": None,
        "phase_a_eval": phase_a_eval,
        "standard_eval": standard_eval,
    }
    eval_history.append(baseline_record)
    write_json(save_dir / "eval_history.json", eval_history)
    append_worklog(
        worklog_path,
        "Eval Step 0",
        [
            f"Phase A mean_q_expert: {phase_a_eval['mean_q_expert']:.6f}",
            f"Phase A mean_q_ref: {phase_a_eval['mean_q_ref']:.6f}",
            f"Phase A mean_ref_td_error: {phase_a_eval['mean_ref_td_error']:.6f}",
            f"Standard td_error: {standard_eval['mean_critic_td_error']:.6f}",
        ],
    )
    set_phase_a_train_mode(algorithm)

    for step in range(1, config.offline_rl.num_gradient_steps + 1):
        step_losses: list[float] = []
        for _ in range(utd):
            batch = _batch_to_device(train_mix.sample(batch_size), device)
            critic_optimizer.zero_grad(set_to_none=True)
            c_loss = critic_loss_ref_bootstrap(
                algorithm.critic,
                algorithm.target_critic,
                batch,
                gamma,
                C,
            )
            c_loss.backward()
            critic_optimizer.step()
            metrics.critic_losses.append(float(c_loss.item()))
            step_losses.append(float(c_loss.item()))

        algorithm.soft_update_target(tau)

        if step % config.offline_rl.log_every == 0:
            logger.info(
                "Phase A step %d/%d critic=%.6f actor=frozen",
                step,
                config.offline_rl.num_gradient_steps,
                sum(step_losses) / max(len(step_losses), 1),
            )

        if step % config.offline_rl.eval_every == 0:
            phase_a_eval, standard_eval = run_dual_eval(algorithm, val_mix, config, num_batches=args.eval_batches)
            record = {
                "step": step,
                "elapsed_sec": time.time() - start,
                "train_critic_loss_tail_200": mean_tail(metrics.critic_losses, 200),
                "phase_a_eval": phase_a_eval,
                "standard_eval": standard_eval,
            }
            eval_history.append(record)
            write_json(save_dir / "eval_history.json", eval_history)
            append_worklog(
                worklog_path,
                f"Eval Step {step}",
                [
                    f"Tail critic loss (200): {record['train_critic_loss_tail_200']}",
                    f"Phase A mean_q_expert: {phase_a_eval['mean_q_expert']:.6f}",
                    f"Phase A mean_q_ref: {phase_a_eval['mean_q_ref']:.6f}",
                    f"Phase A mean_ref_td_error: {phase_a_eval['mean_ref_td_error']:.6f}",
                    f"Standard td_error: {standard_eval['mean_critic_td_error']:.6f}",
                ],
            )
            logger.info(
                "Eval step %d phaseA_td=%.6f q_exp=%.6f q_ref=%.6f std_td=%.6f",
                step,
                phase_a_eval["mean_ref_td_error"],
                phase_a_eval["mean_q_expert"],
                phase_a_eval["mean_q_ref"],
                standard_eval["mean_critic_td_error"],
            )
            set_phase_a_train_mode(algorithm)

    elapsed = time.time() - start
    final_phase_a_eval, final_standard_eval = run_dual_eval(algorithm, val_mix, config, num_batches=args.eval_batches)
    per_bucket = evaluate_per_bucket(algorithm, config, val_buckets, args.eval_batches)

    metadata = {
        "phase": "A",
        "bootstrap_mode": "cached_next_ref",
        "actor_frozen": True,
        "warm_start_ckpt": args.warm_start_ckpt,
        "rl_token_checkpoint": args.rl_token_checkpoint,
        "bucket_paths": bucket_paths,
        "probs": probs,
    }
    _save_rl_checkpoint(
        algorithm,
        actor_optimizer,
        critic_optimizer,
        config.offline_rl.num_gradient_steps,
        metrics,
        str(save_dir),
        metadata=metadata,
    )

    result = {
        "name": args.name,
        "elapsed_sec": elapsed,
        "metadata": metadata,
        "final_train_critic_loss_tail_500": mean_tail(metrics.critic_losses, 500),
        "phase_a_eval": final_phase_a_eval,
        "standard_eval": final_standard_eval,
        "per_bucket_eval": per_bucket,
        "save_dir": str(save_dir),
    }
    write_json(save_dir / "phase_a_results.json", result)
    append_worklog(
        worklog_path,
        "Run Complete",
        [
            f"Elapsed seconds: {elapsed:.1f}",
            f"Final tail critic loss (500): {result['final_train_critic_loss_tail_500']}",
            f"Final Phase A mean_q_expert: {final_phase_a_eval['mean_q_expert']:.6f}",
            f"Final Phase A mean_q_ref: {final_phase_a_eval['mean_q_ref']:.6f}",
            f"Final Phase A mean_ref_td_error: {final_phase_a_eval['mean_ref_td_error']:.6f}",
            f"Final standard td_error: {final_standard_eval['mean_critic_td_error']:.6f}",
            f"Residual risk: Q visualization still has known cache-faithfulness limits outside cp405.",
            "Next action: run cp405 episode visualization against this final checkpoint.",
        ],
    )

    logger.info("Phase A complete. Results written to %s", save_dir)
    logger.info("Final Phase A eval: %s", json.dumps(final_phase_a_eval, indent=2))
    logger.info("Final standard eval: %s", json.dumps(final_standard_eval, indent=2))

    del algorithm, actor_optimizer, critic_optimizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
