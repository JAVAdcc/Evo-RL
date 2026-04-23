#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import configure_logging, load_training_config
from train_chunk_actor_critic import create_algorithm_with_cached_transitions
from train_online_offline import build_mix_buffers, load_warm_start
from train_phase_a_ref_bootstrap import (
    DEFAULT_BUCKET_PATHS,
    _batch_to_device,
    append_worklog,
    apply_warm_start_architecture,
    mean_tail,
    summarize_standard_eval,
    write_json,
)

from lerobot.rlt.evaluator import evaluate_offline
from lerobot.rlt.losses import actor_loss, discounted_chunk_return
from lerobot.rlt.trainer import TrainingMetrics, _save_rl_checkpoint
from lerobot.rlt.utils import soft_update

logger = configure_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase B: reference-anchored target actor TD with slow actor updates.",
    )
    parser.add_argument("--name", default="w_reduced_cp405_phase_b_anchor_10k")
    parser.add_argument("--config", default="src/lerobot/rlt/configs/pi05_rlt.yaml")
    parser.add_argument(
        "--rl-token-checkpoint",
        default="outputs/rlt_demo_adapt_271ep_sft_fp32/demo_adapt_checkpoint.pt",
    )
    parser.add_argument(
        "--warm-start-ckpt",
        default="outputs/phase_a_ref_bootstrap/w_reduced_cp405_refboot_phase_a_10k/rl_checkpoint.pt",
    )
    parser.add_argument("--output-dir", default="outputs/phase_b_target_anchor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--actor-lr-scale", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--target-actor-tau", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--actor-update-interval", type=int, default=8)
    parser.add_argument("--bootstrap-lambda-start", type=float, default=0.0)
    parser.add_argument("--bootstrap-lambda-max", type=float, default=0.30)
    parser.add_argument("--bootstrap-ramp-steps", type=int, default=10000)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--bucket-cp405-path", default=DEFAULT_BUCKET_PATHS["cp405"])
    parser.add_argument("--bucket-teleop-path", default=DEFAULT_BUCKET_PATHS["teleop141"])
    parser.add_argument("--bucket-intervene-path", default=DEFAULT_BUCKET_PATHS["intervene156"])
    parser.add_argument("--prob-cp405", type=float, default=0.30)
    parser.add_argument("--prob-teleop", type=float, default=0.35)
    parser.add_argument("--prob-intervene", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace) -> dict[str, float]:
    config.offline_rl.num_gradient_steps = args.gradient_steps
    config.offline_rl.eval_every = args.eval_every
    config.offline_rl.log_every = args.log_every
    config.offline_rl.save_every = 10**9
    config.training.actor_update_interval = args.actor_update_interval
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.critic_lr is not None:
        config.critic.lr = args.critic_lr
    if args.actor_lr is not None:
        config.actor.lr = args.actor_lr
    else:
        config.actor.lr = float(config.actor.lr) * float(args.actor_lr_scale)
    if args.gamma is not None:
        config.training.gamma = args.gamma
    if args.tau is not None:
        config.training.tau = args.tau
    if args.beta is not None:
        config.training.beta = args.beta
    target_actor_tau = (
        float(args.target_actor_tau)
        if args.target_actor_tau is not None
        else float(config.training.tau)
    )
    return {"target_actor_tau": target_actor_tau}


def blend_lambda(step: int, start: float, max_value: float, ramp_steps: int) -> float:
    if ramp_steps <= 0:
        return float(max_value)
    progress = min(max(step, 0) / ramp_steps, 1.0)
    return float(start + (max_value - start) * progress)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preserve_rng_state() -> dict[str, object]:
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_with_eval_seed(eval_seed: int, fn):
    state = preserve_rng_state()
    try:
        set_random_seed(eval_seed)
        return fn()
    finally:
        restore_rng_state(state)


def build_phase_b_metrics_sidecar(config, warm_start_arch: dict[str, object], metadata: dict | None = None) -> dict[str, object]:
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
        "bootstrap_mode": "target_actor_ref_anchor",
        "actor_frozen": False,
        "warm_start_architecture": warm_start_arch,
    }
    if metadata:
        payload["metadata"] = metadata
    return {"config": payload}


def build_bootstrap_action(
    target_actor,
    next_state_vec: torch.Tensor,
    next_ref_flat: torch.Tensor,
    lambda_blend: float,
) -> torch.Tensor:
    with torch.no_grad():
        ref_anchor = next_ref_flat.clamp(-1.0, 1.0)
        mu_target, _ = target_actor.forward(next_state_vec, ref_anchor)
        mixed = (1.0 - lambda_blend) * ref_anchor + lambda_blend * mu_target
        return mixed.clamp(-1.0, 1.0)


def critic_loss_anchor_bootstrap(
    critic,
    target_critic,
    target_actor,
    batch: dict[str, torch.Tensor],
    gamma: float,
    chunk_length: int,
    lambda_blend: float,
) -> torch.Tensor:
    x = batch["state_vec"]
    a = batch["exec_chunk_flat"]
    x_next = batch["next_state_vec"]
    ref_next = batch["next_ref_flat"]
    reward_seq = batch["reward_seq"]
    done = batch["done"]
    actual_steps = batch.get("actual_steps")

    with torch.no_grad():
        a_next = build_bootstrap_action(target_actor, x_next, ref_next, lambda_blend)
        q_next = target_critic.min_q(x_next, a_next).clamp(-100.0, 100.0)
        reward = discounted_chunk_return(reward_seq, gamma, actual_steps)
        if actual_steps is not None:
            bootstrap_exp = actual_steps.unsqueeze(-1).float()
        else:
            bootstrap_exp = torch.full_like(done.unsqueeze(-1), chunk_length, dtype=torch.float32)
        bootstrap = (gamma ** bootstrap_exp) * (1.0 - done.unsqueeze(-1)) * q_next
        target = reward + bootstrap

    q1, q2 = critic(x, a)
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def evaluate_offline_anchor_bootstrap(
    algorithm,
    target_actor,
    val_buffer,
    config,
    lambda_blend: float,
    num_batches: int = 10,
) -> dict[str, float]:
    algorithm.eval()
    target_actor.eval()
    policy = algorithm.policy
    device = next(algorithm.parameters()).device

    totals = {
        "expert_mse": 0.0,
        "ref_mse": 0.0,
        "ref_dropped_mse": 0.0,
        "mean_q_policy": 0.0,
        "mean_q_expert": 0.0,
        "mean_q_ref": 0.0,
        "mean_anchor_td_error": 0.0,
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
            totals["mean_anchor_td_error"] += critic_loss_anchor_bootstrap(
                algorithm.critic,
                algorithm.target_critic,
                target_actor,
                batch,
                config.training.gamma,
                config.chunk_length,
                lambda_blend,
            ).item()

    n = max(num_batches, 1)
    out = {k: v / n for k, v in totals.items()}
    out["q_gap_policy_expert"] = out["mean_q_policy"] - out["mean_q_expert"]
    out["q_gap_policy_ref"] = out["mean_q_policy"] - out["mean_q_ref"]
    out["bootstrap_lambda"] = float(lambda_blend)
    return out


def evaluate_per_bucket_anchor(
    algorithm,
    target_actor,
    config,
    val_buckets,
    eval_batches: int,
    lambda_blend: float,
    eval_seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    def _eval():
        out: dict[str, dict[str, dict[str, float]]] = {}
        for name, buf in val_buckets.items():
            if len(buf) == 0:
                continue
            anchor_eval = evaluate_offline_anchor_bootstrap(
                algorithm,
                target_actor,
                buf,
                config,
                lambda_blend=lambda_blend,
                num_batches=eval_batches,
            )
            standard_eval = summarize_standard_eval(
                evaluate_offline(algorithm, buf, config, num_batches=eval_batches),
            )
            out[name] = {"phase_b": anchor_eval, "standard": standard_eval}
        return out

    return run_with_eval_seed(eval_seed, _eval)


def run_dual_eval(
    algorithm,
    target_actor,
    buffer,
    config,
    num_batches: int,
    lambda_blend: float,
    eval_seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    def _eval():
        phase_b_eval = evaluate_offline_anchor_bootstrap(
            algorithm,
            target_actor,
            buffer,
            config,
            lambda_blend=lambda_blend,
            num_batches=num_batches,
        )
        standard_eval = summarize_standard_eval(
            evaluate_offline(algorithm, buffer, config, num_batches=num_batches),
        )
        return phase_b_eval, standard_eval

    return run_with_eval_seed(eval_seed, _eval)


def set_phase_b_train_mode(algorithm, target_actor) -> None:
    algorithm.train()
    target_actor.eval()


def reset_best_dirs(save_dir: Path) -> None:
    for tag in ("best_q_expert", "best_anchor_td"):
        path = save_dir / tag
        if path.exists():
            shutil.rmtree(path)


def save_phase_b_checkpoint(
    algorithm,
    target_actor,
    actor_optimizer,
    critic_optimizer,
    step: int,
    metrics: TrainingMetrics,
    save_dir: Path,
    metadata: dict[str, object],
) -> None:
    _save_rl_checkpoint(
        algorithm,
        actor_optimizer,
        critic_optimizer,
        step,
        metrics,
        str(save_dir),
        metadata=metadata,
    )
    torch.save({"target_actor_state_dict": target_actor.state_dict(), "step": step}, save_dir / "target_actor.pt")


def save_eval_best_checkpoint(
    algorithm,
    target_actor,
    actor_optimizer,
    critic_optimizer,
    step: int,
    metrics: TrainingMetrics,
    root_dir: Path,
    tag: str,
    metadata: dict[str, object],
    record: dict[str, object],
) -> None:
    ckpt_dir = root_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_phase_b_checkpoint(
        algorithm,
        target_actor,
        actor_optimizer,
        critic_optimizer,
        step,
        metrics,
        ckpt_dir,
        metadata,
    )
    write_json(ckpt_dir / "eval_record.json", record)


def maybe_update_best_checkpoints(
    *,
    algorithm,
    target_actor,
    actor_optimizer,
    critic_optimizer,
    step: int,
    metrics: TrainingMetrics,
    save_dir: Path,
    metadata: dict[str, object],
    record: dict[str, object],
    best_q_record: dict[str, object] | None,
    best_td_record: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    phase_b_eval = record["phase_b_eval"]
    next_best_q = best_q_record
    next_best_td = best_td_record
    best_events: list[str] = []

    if next_best_q is None or phase_b_eval["mean_q_expert"] > next_best_q["phase_b_eval"]["mean_q_expert"]:
        next_best_q = copy.deepcopy(record)
        save_eval_best_checkpoint(
            algorithm,
            target_actor,
            actor_optimizer,
            critic_optimizer,
            step,
            metrics,
            save_dir,
            "best_q_expert",
            {**metadata, "selection": "best_q_expert"},
            next_best_q,
        )
        best_events.append(f"best_q_expert updated: step={step}, q_exp={phase_b_eval['mean_q_expert']:.6f}")

    if next_best_td is None or phase_b_eval["mean_anchor_td_error"] < next_best_td["phase_b_eval"]["mean_anchor_td_error"]:
        next_best_td = copy.deepcopy(record)
        save_eval_best_checkpoint(
            algorithm,
            target_actor,
            actor_optimizer,
            critic_optimizer,
            step,
            metrics,
            save_dir,
            "best_anchor_td",
            {**metadata, "selection": "best_anchor_td"},
            next_best_td,
        )
        best_events.append(
            f"best_anchor_td updated: step={step}, td={phase_b_eval['mean_anchor_td_error']:.6f}"
        )

    return next_best_q, next_best_td, best_events


def main() -> None:
    args = parse_args()
    eval_seed = int(args.seed) if args.seed is not None else 0
    if args.seed is not None:
        set_random_seed(args.seed)
    config = load_training_config(args.config)
    runtime_cfg = apply_overrides(config, args)
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
    reset_best_dirs(save_dir)
    worklog_path = save_dir / "worklog.md"
    worklog_path.write_text("# Phase B Worklog\n")

    write_json(
        save_dir / "phase_b_config.json",
        {
            "args": vars(args),
            "bucket_paths": bucket_paths,
            "probs": probs,
            "warm_start_architecture": warm_start_arch,
            "resolved_actor_lr": config.actor.lr,
            "resolved_target_actor_tau": runtime_cfg["target_actor_tau"],
            "eval_seed": eval_seed,
        },
    )

    append_worklog(
        worklog_path,
        "Kickoff",
        [
            "Objective: continue from Phase A with target actor + reference-anchor TD bootstrap.",
            "Completion criteria: final checkpoint, target_actor.pt, eval history, and bucket-wise diagnostics are written.",
            f"Warm start checkpoint: {args.warm_start_ckpt}",
            f"Warm start architecture: {warm_start_arch}",
            f"Sampling probabilities: {probs}",
            f"Bootstrap lambda schedule: start={args.bootstrap_lambda_start} max={args.bootstrap_lambda_max} ramp_steps={args.bootstrap_ramp_steps}",
            f"Actor update interval: {config.training.actor_update_interval}",
            f"Target actor tau: {runtime_cfg['target_actor_tau']}",
            "Target actor EMA scope: on_actor_update (TD3-style delayed policy update).",
            f"Seed: {args.seed}",
            f"Eval seed: {eval_seed}",
        ],
    )

    train_mix, val_mix, train_buckets, val_buckets = build_mix_buffers(bucket_paths, probs, config)
    logger.info(
        "Phase B buffers loaded. train=%s val=%s probs=%s",
        {k: len(v) for k, v in train_buckets.items()},
        {k: len(v) for k, v in val_buckets.items()},
        probs,
    )

    algorithm = create_algorithm_with_cached_transitions(config, args.rl_token_checkpoint, args.device)
    load_warm_start(algorithm, args.warm_start_ckpt, args.device)
    target_actor = copy.deepcopy(algorithm.policy.actor).to(args.device)
    for param in target_actor.parameters():
        param.requires_grad_(False)
    set_phase_b_train_mode(algorithm, target_actor)

    actor_optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=config.actor.lr)
    critic_optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=config.critic.lr)

    metadata = {
        "phase": "B",
        "bootstrap_mode": "target_actor_ref_anchor",
        "bootstrap_lambda_start": args.bootstrap_lambda_start,
        "bootstrap_lambda_max": args.bootstrap_lambda_max,
        "bootstrap_ramp_steps": args.bootstrap_ramp_steps,
        "target_actor_tau": runtime_cfg["target_actor_tau"],
        "target_actor_update_scope": "on_actor_update",
        "actor_update_interval": config.training.actor_update_interval,
        "warm_start_ckpt": args.warm_start_ckpt,
        "rl_token_checkpoint": args.rl_token_checkpoint,
        "bucket_paths": bucket_paths,
        "probs": probs,
        "seed": args.seed,
        "eval_seed": eval_seed,
    }
    write_json(
        save_dir / "metrics.json",
        build_phase_b_metrics_sidecar(
            config,
            warm_start_arch,
            metadata=metadata,
        ),
    )

    metrics = TrainingMetrics()
    eval_history: list[dict[str, object]] = []
    device = next(algorithm.parameters()).device
    gamma = config.training.gamma
    chunk_length = config.chunk_length
    tau = config.training.tau
    target_actor_tau = runtime_cfg["target_actor_tau"]
    beta = config.training.beta
    batch_size = config.training.batch_size
    utd = config.training.utd_ratio
    actor_interval = config.training.actor_update_interval
    critic_update_count = 0
    start = time.time()

    lambda_0 = blend_lambda(0, args.bootstrap_lambda_start, args.bootstrap_lambda_max, args.bootstrap_ramp_steps)
    phase_b_eval, standard_eval = run_dual_eval(
        algorithm,
        target_actor,
        val_mix,
        config,
        num_batches=args.eval_batches,
        lambda_blend=lambda_0,
        eval_seed=eval_seed,
    )
    baseline_record = {
        "step": 0,
        "elapsed_sec": 0.0,
        "bootstrap_lambda": lambda_0,
        "train_critic_loss_tail_200": None,
        "train_actor_loss_tail_50": None,
        "phase_b_eval": phase_b_eval,
        "standard_eval": standard_eval,
    }
    eval_history.append(baseline_record)
    write_json(save_dir / "eval_history.json", eval_history)
    best_q_record: dict[str, object] | None = None
    best_td_record: dict[str, object] | None = None
    best_q_record, best_td_record, baseline_best_events = maybe_update_best_checkpoints(
        algorithm=algorithm,
        target_actor=target_actor,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        step=0,
        metrics=metrics,
        save_dir=save_dir,
        metadata=metadata,
        record=baseline_record,
        best_q_record=best_q_record,
        best_td_record=best_td_record,
    )
    append_worklog(
        worklog_path,
        "Eval Step 0",
        [
            f"Bootstrap lambda: {lambda_0:.6f}",
            f"Phase B mean_q_expert: {phase_b_eval['mean_q_expert']:.6f}",
            f"Phase B mean_q_ref: {phase_b_eval['mean_q_ref']:.6f}",
            f"Phase B mean_anchor_td_error: {phase_b_eval['mean_anchor_td_error']:.6f}",
            f"Standard td_error: {standard_eval['mean_critic_td_error']:.6f}",
            *baseline_best_events,
        ],
    )
    set_phase_b_train_mode(algorithm, target_actor)

    for step in range(1, config.offline_rl.num_gradient_steps + 1):
        lambda_blend = blend_lambda(
            step,
            args.bootstrap_lambda_start,
            args.bootstrap_lambda_max,
            args.bootstrap_ramp_steps,
        )
        step_critic_losses: list[float] = []
        last_actor_loss = None
        actor_updates_this_step = 0

        for _ in range(utd):
            batch = _batch_to_device(train_mix.sample(batch_size), device)

            critic_optimizer.zero_grad(set_to_none=True)
            c_loss = critic_loss_anchor_bootstrap(
                algorithm.critic,
                algorithm.target_critic,
                target_actor,
                batch,
                gamma,
                chunk_length,
                lambda_blend,
            )
            c_loss.backward()
            critic_optimizer.step()
            metrics.critic_losses.append(float(c_loss.item()))
            step_critic_losses.append(float(c_loss.item()))
            critic_update_count += 1

            if critic_update_count % actor_interval == 0:
                actor_optimizer.zero_grad(set_to_none=True)
                a_loss = actor_loss(algorithm.policy.actor, algorithm.critic, batch, beta)
                a_loss.backward()
                actor_optimizer.step()
                last_actor_loss = float(a_loss.item())
                metrics.actor_losses.append(last_actor_loss)
                actor_updates_this_step += 1
                soft_update(target_actor, algorithm.policy.actor, target_actor_tau)

        algorithm.soft_update_target(tau)

        if step % config.offline_rl.log_every == 0:
            logger.info(
                "Phase B step %d/%d critic=%.6f actor=%s lambda=%.4f actor_updates=%d",
                step,
                config.offline_rl.num_gradient_steps,
                sum(step_critic_losses) / max(len(step_critic_losses), 1),
                f"{last_actor_loss:.6f}" if last_actor_loss is not None else "N/A",
                lambda_blend,
                actor_updates_this_step,
            )

        if step % config.offline_rl.eval_every == 0:
            phase_b_eval, standard_eval = run_dual_eval(
                algorithm,
                target_actor,
                val_mix,
                config,
                num_batches=args.eval_batches,
                lambda_blend=lambda_blend,
                eval_seed=eval_seed,
            )
            record = {
                "step": step,
                "elapsed_sec": time.time() - start,
                "bootstrap_lambda": lambda_blend,
                "train_critic_loss_tail_200": mean_tail(metrics.critic_losses, 200),
                "train_actor_loss_tail_50": mean_tail(metrics.actor_losses, 50),
                "phase_b_eval": phase_b_eval,
                "standard_eval": standard_eval,
            }
            eval_history.append(record)
            write_json(save_dir / "eval_history.json", eval_history)
            best_q_record, best_td_record, best_events = maybe_update_best_checkpoints(
                algorithm=algorithm,
                target_actor=target_actor,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                step=step,
                metrics=metrics,
                save_dir=save_dir,
                metadata=metadata,
                record=record,
                best_q_record=best_q_record,
                best_td_record=best_td_record,
            )
            append_worklog(
                worklog_path,
                f"Eval Step {step}",
                [
                    f"Bootstrap lambda: {lambda_blend:.6f}",
                    f"Tail critic loss (200): {record['train_critic_loss_tail_200']}",
                    f"Tail actor loss (50): {record['train_actor_loss_tail_50']}",
                    f"Phase B mean_q_expert: {phase_b_eval['mean_q_expert']:.6f}",
                    f"Phase B mean_q_ref: {phase_b_eval['mean_q_ref']:.6f}",
                    f"Phase B mean_anchor_td_error: {phase_b_eval['mean_anchor_td_error']:.6f}",
                    f"Standard td_error: {standard_eval['mean_critic_td_error']:.6f}",
                    *best_events,
                ],
            )
            logger.info(
                "Eval step %d lambda=%.4f phaseB_td=%.6f q_exp=%.6f q_ref=%.6f std_td=%.6f",
                step,
                lambda_blend,
                phase_b_eval["mean_anchor_td_error"],
                phase_b_eval["mean_q_expert"],
                phase_b_eval["mean_q_ref"],
                standard_eval["mean_critic_td_error"],
            )
            set_phase_b_train_mode(algorithm, target_actor)

    elapsed = time.time() - start
    final_lambda = blend_lambda(
        config.offline_rl.num_gradient_steps,
        args.bootstrap_lambda_start,
        args.bootstrap_lambda_max,
        args.bootstrap_ramp_steps,
    )
    final_phase_b_eval, final_standard_eval = run_dual_eval(
        algorithm,
        target_actor,
        val_mix,
        config,
        num_batches=args.eval_batches,
        lambda_blend=final_lambda,
        eval_seed=eval_seed,
    )
    final_record = {
        "step": config.offline_rl.num_gradient_steps,
        "elapsed_sec": elapsed,
        "bootstrap_lambda": final_lambda,
        "train_critic_loss_tail_200": mean_tail(metrics.critic_losses, 200),
        "train_actor_loss_tail_50": mean_tail(metrics.actor_losses, 50),
        "phase_b_eval": final_phase_b_eval,
        "standard_eval": final_standard_eval,
    }
    best_q_record, best_td_record, final_best_events = maybe_update_best_checkpoints(
        algorithm=algorithm,
        target_actor=target_actor,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        step=config.offline_rl.num_gradient_steps,
        metrics=metrics,
        save_dir=save_dir,
        metadata=metadata,
        record=final_record,
        best_q_record=best_q_record,
        best_td_record=best_td_record,
    )
    per_bucket = evaluate_per_bucket_anchor(
        algorithm,
        target_actor,
        config,
        val_buckets,
        eval_batches=args.eval_batches,
        lambda_blend=final_lambda,
        eval_seed=eval_seed,
    )

    save_phase_b_checkpoint(
        algorithm,
        target_actor,
        actor_optimizer,
        critic_optimizer,
        config.offline_rl.num_gradient_steps,
        metrics,
        save_dir,
        metadata,
    )

    result = {
        "name": args.name,
        "elapsed_sec": elapsed,
        "metadata": metadata,
        "final_lambda": final_lambda,
        "final_train_critic_loss_tail_500": mean_tail(metrics.critic_losses, 500),
        "final_train_actor_loss_tail_100": mean_tail(metrics.actor_losses, 100),
        "phase_b_eval": final_phase_b_eval,
        "standard_eval": final_standard_eval,
        "per_bucket_eval": per_bucket,
        "best_q_expert_record": best_q_record,
        "best_anchor_td_record": best_td_record,
        "save_dir": str(save_dir),
    }
    write_json(save_dir / "phase_b_results.json", result)
    append_worklog(
        worklog_path,
        "Completed",
        [
            f"Elapsed sec: {elapsed:.2f}",
            f"Final lambda: {final_lambda:.6f}",
            f"Final tail critic loss (500): {result['final_train_critic_loss_tail_500']}",
            f"Final tail actor loss (100): {result['final_train_actor_loss_tail_100']}",
            f"Final Phase B mean_q_expert: {final_phase_b_eval['mean_q_expert']:.6f}",
            f"Final Phase B mean_q_ref: {final_phase_b_eval['mean_q_ref']:.6f}",
            f"Final Phase B mean_anchor_td_error: {final_phase_b_eval['mean_anchor_td_error']:.6f}",
            f"Final standard td_error: {final_standard_eval['mean_critic_td_error']:.6f}",
            *final_best_events,
            f"Checkpoint dir: {save_dir}",
        ],
    )
    logger.info(
        "Phase B done in %.2fs. q_exp=%.6f q_ref=%.6f anchor_td=%.6f std_td=%.6f",
        elapsed,
        final_phase_b_eval["mean_q_expert"],
        final_phase_b_eval["mean_q_ref"],
        final_phase_b_eval["mean_anchor_td_error"],
        final_standard_eval["mean_critic_td_error"],
    )


if __name__ == "__main__":
    main()
