#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

log = logging.getLogger(__name__)

JOINT_TYPES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
WINDOW = 60
DEFAULT_VLA_MODEL = "Elvinky/pi05_full_mix_562ep_sft_fp32"
DEFAULT_RL_VLA_MODEL = "Elvinky/pi05_screw_271ep_sft_fp32"
DEFAULT_HF_REPO = "Shiki42/rlt_pi0.5_screw"
DEFAULT_RL_TOKEN_PATH = "rl_token/271ep_pi0.5_screw_sft_rltoken/demo_adapt_checkpoint.pt"
DEFAULT_RL_CONFIG_PATH = "rl_token/271ep_pi0.5_screw_sft_rltoken/pi05_rlt.yaml"
DEFAULT_AC_PATH = "rl_token/271ep_pi0.5_screw_sft_rltoken/actor_critic/0412_278cp_warmup/rl_checkpoint.pt"
DEFAULT_AC_METRICS_PATH = "rl_token/271ep_pi0.5_screw_sft_rltoken/actor_critic/0412_278cp_warmup/metrics.json"
VIDEO_FRAME_WIDTH = 320


@dataclass
class RLModelPaths:
    vla_model: str
    rl_token_ckpt: str
    ac_ckpt: str
    config_path: str
    metrics_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GT / VLA / RL actions on one dataset episode and export an interactive HTML.",
    )
    parser.add_argument("--dataset-path", required=True, help="Local LeRobot dataset root.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument("--repo-id", default="rlt_viz", help="LeRobot local repo id placeholder.")
    parser.add_argument("--output", default="warmup_compare.html", help="Output HTML path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    parser.add_argument("--stride", type=int, default=1, help="Run inference every N frames.")
    parser.add_argument("--window-size", type=int, default=WINDOW, help="Visible window in frames.")
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument(
        "--vla-model",
        default=DEFAULT_VLA_MODEL,
        help="Standalone VLA model used for the VLA curves.",
    )
    parser.add_argument(
        "--no-vla",
        action="store_true",
        help="Skip standalone VLA inference and only visualize GT + RL.",
    )

    parser.add_argument(
        "--rl-vla-model",
        default=DEFAULT_RL_VLA_MODEL,
        help="VLA backbone used inside the RL policy.",
    )
    parser.add_argument(
        "--rl-token-ckpt",
        default="",
        help="RL token checkpoint. Empty means use latest warmup default from HF.",
    )
    parser.add_argument(
        "--ac-ckpt",
        default="",
        help="Actor-critic checkpoint. Empty means use latest warmup default from HF.",
    )
    parser.add_argument(
        "--rl-config",
        default="",
        help="Optional RLT yaml config. Empty means use latest warmup default from HF.",
    )
    parser.add_argument(
        "--ac-metrics",
        default="",
        help="Optional metrics.json next to the AC checkpoint for architecture metadata.",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help="HF repo used to resolve default RL warmup checkpoints.",
    )
    parser.add_argument(
        "--no-rl",
        action="store_true",
        help="Skip RL inference and only visualize GT + VLA.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video frame extraction (faster, no video player in output).",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=VIDEO_FRAME_WIDTH,
        help="Width for extracted video frames (height auto-scaled).",
    )
    return parser.parse_args()


def load_dataset(dataset_path: str, repo_id: str, episode_index: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
        episodes=[episode_index],
        video_backend="pyav",
    )


def detect_camera_keys(dataset) -> list[str]:
    return [key.removeprefix("observation.images.") for key in dataset.features if key.startswith("observation.images.")]


def load_action_quantiles(dataset) -> tuple[np.ndarray, np.ndarray]:
    stats = dataset.meta.stats["action"]
    q01 = stats["q01"].numpy() if isinstance(stats["q01"], torch.Tensor) else np.array(stats["q01"])
    q99 = stats["q99"].numpy() if isinstance(stats["q99"], torch.Tensor) else np.array(stats["q99"])
    return q01, q99


def load_state_quantiles(dataset) -> tuple[np.ndarray, np.ndarray]:
    stats = dataset.meta.stats["observation.state"]
    q01 = stats["q01"].numpy() if isinstance(stats["q01"], torch.Tensor) else np.array(stats["q01"])
    q99 = stats["q99"].numpy() if isinstance(stats["q99"], torch.Tensor) else np.array(stats["q99"])
    return q01, q99


def normalize_proprio(proprio: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    denom = q99 - q01
    denom = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
    proprio = (proprio - q01) / denom * 2.0 - 1.0
    return proprio.clamp(-1.0, 1.0)


def unnormalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    return (actions + 1.0) / 2.0 * (q99 - q01) + q01


def extract_ground_truth(dataset, stride: int) -> np.ndarray:
    return np.array([dataset[index]["action"].numpy() for index in range(0, len(dataset), stride)])


def load_vla_adapter(
    vla_model: str,
    task: str,
    action_dim: int,
    proprio_dim: int,
    device: str,
):
    from lerobot.rlt.pi05_adapter import Pi05VLAAdapter

    return Pi05VLAAdapter(
        model_path=vla_model,
        actual_action_dim=action_dim,
        actual_proprio_dim=proprio_dim,
        task_instruction=task,
        device=device,
        token_pool_size=0,
    )


def run_vla_inference(
    vla,
    dataset,
    camera_full_keys: list[str],
    device: str,
    stride: int,
    state_q01: torch.Tensor,
    state_q99: torch.Tensor,
) -> np.ndarray:
    from lerobot.rlt.interfaces import Observation

    actions = []
    frame_indices = list(range(0, len(dataset), stride))
    log.info("VLA inference over %d frames", len(frame_indices))
    start = time.monotonic()
    for index in tqdm(frame_indices, desc="VLA"):
        item = dataset[index]
        images = {}
        for full_key in camera_full_keys:
            images[full_key.split(".")[-1]] = item[full_key].unsqueeze(0).to(device)
        proprio = item["observation.state"].unsqueeze(0).to(device=device, dtype=torch.float32)
        proprio = normalize_proprio(proprio, state_q01, state_q99)
        obs = Observation(images=images, proprio=proprio)
        with torch.inference_mode():
            vla_out = vla.forward_vla(obs)
        actions.append(vla_out.sampled_action_chunk[0, 0, :].cpu().numpy())
    elapsed = time.monotonic() - start
    log.info("VLA done in %.1fs (%.0f ms/frame)", elapsed, elapsed / max(len(actions), 1) * 1000)
    return np.array(actions)


def resolve_hf_file(repo_id: str, path_in_repo: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=path_in_repo)


def resolve_rl_model_paths(args: argparse.Namespace) -> RLModelPaths:
    if args.no_rl:
        raise ValueError("RL model resolution requested while --no-rl is set")

    rl_token_ckpt = args.rl_token_ckpt or resolve_hf_file(args.hf_repo, DEFAULT_RL_TOKEN_PATH)
    ac_ckpt = args.ac_ckpt or resolve_hf_file(args.hf_repo, DEFAULT_AC_PATH)
    config_path = args.rl_config or resolve_hf_file(args.hf_repo, DEFAULT_RL_CONFIG_PATH)
    metrics_path = args.ac_metrics or None
    if metrics_path is None:
        sibling_metrics = Path(ac_ckpt).with_name("metrics.json")
        if sibling_metrics.exists():
            metrics_path = str(sibling_metrics)
        else:
            try:
                metrics_path = resolve_hf_file(args.hf_repo, DEFAULT_AC_METRICS_PATH)
            except Exception:
                metrics_path = None

    return RLModelPaths(
        vla_model=args.rl_vla_model,
        rl_token_ckpt=rl_token_ckpt,
        ac_ckpt=ac_ckpt,
        config_path=config_path,
        metrics_path=metrics_path,
    )


def load_ac_metadata(ac_ckpt_path: str, metrics_path: str | None) -> tuple[dict, dict]:
    ac_ckpt = torch.load(ac_ckpt_path, map_location="cpu", weights_only=False)
    metrics_metadata = {}
    if metrics_path:
        metrics = json.loads(Path(metrics_path).read_text())
        metrics_metadata = metrics.get("config", {})
    return ac_ckpt, {
        "checkpoint": ac_ckpt.get("metadata", {}),
        "metrics": metrics_metadata,
    }


def infer_critic_architecture(
    critic_state_dict: dict,
    *,
    default_activation: str = "relu",
) -> dict:
    """Infer TwinCritic construction kwargs from q1 sub-network."""
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


def load_critic_from_ckpt(ac_ckpt: dict, config, state_dim: int, chunk_dim: int, device: str):
    """Create TwinCritic from AC checkpoint dict and load weights."""
    from lerobot.rlt.critic import TwinCritic

    checkpoint_metadata = ac_ckpt.get("metadata", {})
    checkpoint_critic = checkpoint_metadata.get("critic", {})
    inferred = infer_critic_architecture(
        ac_ckpt["critic_state_dict"],
        default_activation=config.critic.activation,
    )
    arch = {
        "hidden_dim": int(checkpoint_critic.get("hidden_dim", inferred["hidden_dim"])),
        "num_layers": int(checkpoint_critic.get("num_layers", inferred["num_layers"])),
        "activation": str(checkpoint_critic.get("activation", inferred["activation"])),
        "layer_norm": bool(checkpoint_critic.get("layer_norm", inferred["layer_norm"])),
        "residual": bool(checkpoint_critic.get("residual", inferred["residual"])),
    }
    critic = TwinCritic(state_dim=state_dim, chunk_dim=chunk_dim, **arch)
    critic.load_state_dict(ac_ckpt["critic_state_dict"])
    return critic.to(device).eval()


def load_rl_policy(paths: RLModelPaths, task: str, device: str):
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.pi05_adapter import Pi05VLAAdapter
    from lerobot.rlt.policy import RLTPolicy
    from lerobot.rlt.utils import filter_encoder_only, infer_actor_architecture

    config = RLTConfig.from_yaml(paths.config_path)
    ac_ckpt, ac_metadata = load_ac_metadata(paths.ac_ckpt, paths.metrics_path)
    checkpoint_metadata = ac_metadata.get("checkpoint", {})
    metrics_metadata = ac_metadata.get("metrics", {})
    checkpoint_actor = checkpoint_metadata.get("actor", {})
    checkpoint_critic = checkpoint_metadata.get("critic", {})
    inferred = infer_actor_architecture(
        ac_ckpt["actor_state_dict"],
        default_activation=config.actor.activation,
        default_fixed_std=config.actor.fixed_std,
        default_ref_dropout_p=config.actor.ref_dropout_p,
    )

    config.actor.hidden_dim = int(checkpoint_actor.get("hidden_dim", metrics_metadata.get("actor_hidden", inferred["hidden_dim"])))
    config.actor.num_layers = int(checkpoint_actor.get("num_layers", metrics_metadata.get("actor_layers", inferred["num_layers"])))
    config.actor.activation = str(checkpoint_actor.get("activation", metrics_metadata.get("actor_activation", inferred["activation"])))
    config.actor.layer_norm = bool(checkpoint_actor.get("layer_norm", metrics_metadata.get("actor_layer_norm", inferred["layer_norm"])))
    config.actor.residual = bool(checkpoint_actor.get("residual", metrics_metadata.get("actor_residual", inferred["residual"])))
    config.actor.fixed_std = float(checkpoint_actor.get("fixed_std", metrics_metadata.get("fixed_std", inferred["fixed_std"])))
    config.actor.ref_dropout_p = float(checkpoint_actor.get("ref_dropout_p", metrics_metadata.get("ref_dropout_p", inferred["ref_dropout_p"])))

    if checkpoint_critic:
        config.critic.hidden_dim = int(checkpoint_critic.get("hidden_dim", config.critic.hidden_dim))
        config.critic.num_layers = int(checkpoint_critic.get("num_layers", config.critic.num_layers))
        config.critic.activation = str(checkpoint_critic.get("activation", config.critic.activation))
        config.critic.layer_norm = bool(checkpoint_critic.get("layer_norm", config.critic.layer_norm))
        config.critic.residual = bool(checkpoint_critic.get("residual", config.critic.residual))

    vla = Pi05VLAAdapter(
        model_path=paths.vla_model,
        actual_action_dim=config.action_dim,
        actual_proprio_dim=config.proprio_dim,
        task_instruction=task,
        device=device,
        token_pool_size=64,
    )
    policy = RLTPolicy(config, vla).to(device)

    rl_ckpt = torch.load(paths.rl_token_ckpt, map_location="cpu", weights_only=False)
    filtered, _ = filter_encoder_only(rl_ckpt["rl_token_state_dict"])
    policy.rl_token.load_state_dict(filtered, strict=False)
    if "rl_token_state_dict" in ac_ckpt:
        filtered, _ = filter_encoder_only(ac_ckpt["rl_token_state_dict"])
        policy.rl_token.load_state_dict(filtered, strict=False)
    policy.actor.load_state_dict(ac_ckpt["actor_state_dict"])
    policy.freeze_vla()
    policy.freeze_rl_token_encoder()
    policy.eval()

    state_dim = config.rl_token.token_dim + config.proprio_dim
    chunk_dim = config.chunk_length * config.action_dim
    critic = load_critic_from_ckpt(ac_ckpt, config, state_dim, chunk_dim, device)
    return policy, critic


def run_rl_inference(
    policy,
    critic,
    dataset,
    camera_full_keys: list[str],
    stride: int,
    device: str,
    state_q01: torch.Tensor,
    state_q99: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    from lerobot.rlt.interfaces import Observation
    from lerobot.rlt.utils import flatten_chunk

    actions = []
    q_values = []
    frame_indices = list(range(0, len(dataset), stride))
    log.info("RL inference over %d frames", len(frame_indices))
    start = time.monotonic()
    for index in tqdm(frame_indices, desc="RL"):
        item = dataset[index]
        images = {}
        for full_key in camera_full_keys:
            images[full_key.split(".")[-1]] = item[full_key].unsqueeze(0).to(device)
        proprio = item["observation.state"].unsqueeze(0).to(device=device, dtype=torch.float32)
        proprio = normalize_proprio(proprio, state_q01, state_q99)
        obs = Observation(images=images, proprio=proprio)
        with torch.inference_mode():
            action_chunk, _, state_vec, _ = policy.select_action(obs, deterministic=True)
            action_flat = flatten_chunk(action_chunk)
            q_val = critic.min_q(state_vec, action_flat)
            q_values.append(q_val.item())
        actions.append(action_chunk[0, 0, :].cpu().numpy())
    elapsed = time.monotonic() - start
    log.info("RL done in %.1fs (%.0f ms/frame)", elapsed, elapsed / max(len(actions), 1) * 1000)
    return np.array(actions), np.array(q_values)


def extract_video_frames(
    dataset,
    camera_full_keys: list[str],
    stride: int,
    target_width: int,
) -> tuple[dict[str, list[str]], tuple[int, int]]:
    """Extract video frames as base64 JPEG strings per camera key.

    Cameras whose first frame has near-zero variance (blank/disconnected)
    are automatically skipped.

    Returns (frames_dict, (frame_w, frame_h)).
    """
    from PIL import Image

    # detect blank cameras by checking first frame variance
    item0 = dataset[0]
    active_keys = []
    for full_key in camera_full_keys:
        img_t = item0[full_key]
        if img_t.std().item() < 0.001:
            log.warning("Skipping blank camera: %s (std=%.4f)", full_key, img_t.std().item())
        else:
            active_keys.append(full_key)

    frames: dict[str, list[str]] = {key: [] for key in active_keys}
    frame_indices = list(range(0, len(dataset), stride))
    h_orig, w_orig = item0[active_keys[0]].shape[1], item0[active_keys[0]].shape[2]
    target_h = int(h_orig * target_width / w_orig)
    log.info(
        "Extracting %d video frames from %d cameras (%d blank skipped), %dx%d",
        len(frame_indices), len(active_keys),
        len(camera_full_keys) - len(active_keys), target_width, target_h,
    )
    for index in tqdm(frame_indices, desc="Frames"):
        item = dataset[index]
        for full_key in active_keys:
            img_t = item[full_key]  # (C, H, W) float [0,1]
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            img = img.resize((target_width, target_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            frames[full_key].append(base64.b64encode(buf.getvalue()).decode())
    return frames, (target_width, target_h)


def build_stats_html(
    gt: np.ndarray,
    vla: np.ndarray | None,
    rl: np.ndarray | None,
) -> str:
    names = JOINT_TYPES
    rows = []
    for joint_index, joint_name in enumerate(names):
        left_index = joint_index
        right_index = joint_index + 6
        row = [f"<td>{joint_name}</td>"]
        for label, array in [("GT-L", gt), ("GT-R", gt), ("VLA-L", vla), ("VLA-R", vla), ("RL-L", rl), ("RL-R", rl)]:
            if array is None:
                row.append("<td>-</td><td>-</td>")
                continue
            dim_index = left_index if label.endswith("-L") else right_index
            row.append(f"<td>{array[:, dim_index].mean():.2f}</td><td>{array[:, dim_index].std():.2f}</td>")
        rows.append("<tr>" + "".join(row) + "</tr>")
    header_cells = [
        "<th>Joint</th>",
        "<th>GT-L μ</th><th>GT-L σ</th>",
        "<th>GT-R μ</th><th>GT-R σ</th>",
        "<th>VLA-L μ</th><th>VLA-L σ</th>",
        "<th>VLA-R μ</th><th>VLA-R σ</th>",
        "<th>RL-L μ</th><th>RL-L σ</th>",
        "<th>RL-R μ</th><th>RL-R σ</th>",
    ]
    return (
        '<h3 style="font-family:sans-serif;">Per-joint summary</h3>'
        '<table style="border-collapse:collapse;font-family:monospace;font-size:13px;">'
        f'<thead style="background:#f0f0f0;"><tr>{"".join(header_cells)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _build_video_section(
    camera_keys: list[str],
    video_frames: dict[str, list[str]],
    q_values: np.ndarray | None,
    fps: float,
    stride: int,
    frame_size: tuple[int, int],
) -> tuple[str, str]:
    """Return (video_html, video_js) for the camera + Q-value overlay."""
    first_key = list(video_frames.keys())[0]
    n_frames = len(video_frames[first_key])
    fw, fh = frame_size

    frames_json = json.dumps({
        k.split(".")[-1]: v for k, v in video_frames.items()
    })
    q_json = json.dumps(q_values.tolist()) if q_values is not None else "null"

    cam_labels = [k.split(".")[-1] for k in video_frames]
    canvases_html = "\n".join(
        f'<div style="text-align:center;">'
        f'<div style="font-size:13px;font-weight:600;margin-bottom:4px;">{lbl}</div>'
        f'<canvas id="cam-{lbl}" width="{fw}" height="{fh}"'
        f' style="border:1px solid #ccc;max-width:100%;"></canvas>'
        f'</div>'
        for lbl in cam_labels
    )

    video_html = f"""
    <div id="video-section" style="margin-bottom:24px;">
      <h3 style="font-family:sans-serif;">Camera Views + Critic Q-value</h3>
      <div id="cam-row" style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
        {canvases_html}
      </div>
      <div style="margin:12px 0;display:flex;align-items:center;gap:12px;font-family:monospace;font-size:13px;">
        <button id="play-btn" style="width:60px;">Play</button>
        <input id="frame-slider" type="range" min="0" max="{n_frames - 1}" value="0" step="1" style="flex:1;">
        <span id="frame-info"></span>
      </div>
    </div>
    """

    video_js = f"""
const VID_FRAMES = {frames_json};
const Q_VALUES = {q_json};
const VID_N = {n_frames};
const VID_FPS = {fps};
const VID_STRIDE = {stride};
const CAM_LABELS = {json.dumps(cam_labels)};
const FRAME_W = {fw};
const FRAME_H = {fh};

const ctxMap = {{}};
CAM_LABELS.forEach(lbl => {{
  ctxMap[lbl] = document.getElementById('cam-' + lbl).getContext('2d');
}});

// precompute Q min/max once
let qMin = 0, qMax = 0;
if (Q_VALUES) {{
  qMin = Q_VALUES[0]; qMax = Q_VALUES[0];
  for (let i = 1; i < Q_VALUES.length; i++) {{
    if (Q_VALUES[i] < qMin) qMin = Q_VALUES[i];
    if (Q_VALUES[i] > qMax) qMax = Q_VALUES[i];
  }}
}}

let vidPlaying = false;
let vidFrame = 0;
const playBtn = document.getElementById('play-btn');
const frameSlider = document.getElementById('frame-slider');
const frameInfo = document.getElementById('frame-info');

function drawVideoFrame(idx) {{
  CAM_LABELS.forEach(lbl => {{
    const ctx = ctxMap[lbl];
    const img = new Image();
    img.onload = function() {{
      ctx.clearRect(0, 0, FRAME_W, FRAME_H);
      ctx.drawImage(this, 0, 0, FRAME_W, FRAME_H);
      if (Q_VALUES) drawQOverlay(ctx, FRAME_W, FRAME_H, idx);
    }};
    img.src = 'data:image/jpeg;base64,' + VID_FRAMES[lbl][idx];
  }});
  const t = (idx * VID_STRIDE / VID_FPS).toFixed(2);
  frameInfo.textContent = 'frame ' + idx + '/' + (VID_N - 1) + '  t=' + t + 's'
    + (Q_VALUES ? '  Q=' + Q_VALUES[idx].toFixed(4) : '');
  frameSlider.value = idx;
}}

function drawQOverlay(ctx, w, h, currentIdx) {{
  const barH = 50;
  const barY = h - barH;
  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.fillRect(0, barY, w, barH);

  const qRange = qMax - qMin || 1;
  const pad = 5;

  ctx.beginPath();
  ctx.strokeStyle = '#FFD700';
  ctx.lineWidth = 2;
  for (let i = 0; i < VID_N; i++) {{
    const x = (i / (VID_N - 1)) * w;
    const y = barY + barH - pad - ((Q_VALUES[i] - qMin) / qRange) * (barH - 2 * pad);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  const curX = (currentIdx / (VID_N - 1)) * w;
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255, 80, 80, 0.9)';
  ctx.lineWidth = 2;
  ctx.moveTo(curX, barY);
  ctx.lineTo(curX, h);
  ctx.stroke();

  ctx.fillStyle = '#FFD700';
  ctx.font = 'bold 12px monospace';
  ctx.fillText('Q: ' + Q_VALUES[currentIdx].toFixed(4), 4, barY + 14);
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = '10px monospace';
  ctx.fillText(qMax.toFixed(3), w - 52, barY + 12);
  ctx.fillText(qMin.toFixed(3), w - 52, h - 4);
}}

let vidInterval = null;

frameSlider.addEventListener('input', () => {{
  vidFrame = parseInt(frameSlider.value, 10);
  drawVideoFrame(vidFrame);
}});

playBtn.addEventListener('click', () => {{
  vidPlaying = !vidPlaying;
  playBtn.textContent = vidPlaying ? 'Pause' : 'Play';
  if (vidPlaying) {{
    vidInterval = setInterval(() => {{
      vidFrame++;
      if (vidFrame >= VID_N) vidFrame = 0;
      drawVideoFrame(vidFrame);
    }}, Math.round(1000 * VID_STRIDE / VID_FPS));
  }} else {{
    clearInterval(vidInterval);
    vidInterval = null;
  }}
}});

// initial draw
drawVideoFrame(0);
"""
    return video_html, video_js


def build_html(
    gt: np.ndarray,
    vla: np.ndarray | None,
    rl: np.ndarray | None,
    q_values: np.ndarray | None,
    video_frames: dict[str, list[str]] | None,
    camera_keys: list[str],
    *,
    stride: int,
    fps: float,
    window_size: int,
    stats_html: str,
    title: str,
    frame_size: tuple[int, int] = (320, 240),
) -> str:
    time_s = (np.arange(gt.shape[0]) * stride / fps).tolist()
    series = {
        "gt-left": gt[:, :6].tolist(),
        "gt-right": gt[:, 6:12].tolist(),
    }
    if vla is not None:
        series["vla-left"] = vla[:, :6].tolist()
        series["vla-right"] = vla[:, 6:12].tolist()
    if rl is not None:
        series["rl-left"] = rl[:, :6].tolist()
        series["rl-right"] = rl[:, 6:12].tolist()

    colors = {
        "gt-left": "#1f77b4",
        "gt-right": "#1f77b4",
        "vla-left": "#ff7f0e",
        "vla-right": "#ff7f0e",
        "rl-left": "#2ca02c",
        "rl-right": "#2ca02c",
    }
    dashes = {
        "gt-left": "solid",
        "gt-right": "dash",
        "vla-left": "solid",
        "vla-right": "dash",
        "rl-left": "solid",
        "rl-right": "dash",
    }

    data_json = json.dumps(
        {
            "time": time_s,
            "series": series,
            "joint_types": JOINT_TYPES,
            "window": window_size,
            "n": gt.shape[0],
            "colors": colors,
            "dashes": dashes,
        }
    )

    video_html = ""
    video_js = ""
    if video_frames:
        video_html, video_js = _build_video_section(
            camera_keys, video_frames, q_values, fps, stride, frame_size,
        )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    #plot {{ width: 100%; height: 960px; }}
    #controls {{ margin: 16px 0; }}
    #time-slider {{ width: 100%; }}
    #frame-label {{ margin-bottom: 8px; font-size: 14px; }}
    table td, table th {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
    table td:first-child, table th:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  {video_html}
  <div id="plot"></div>
  <div id="controls">
    <div id="frame-label"></div>
    <input id="time-slider" type="range" min="0" max="{max(0, gt.shape[0] - window_size)}" value="0" step="1">
  </div>
  <div>{stats_html}</div>
<script>
const DATA = {data_json};
const plotDiv = document.getElementById("plot");
const frameLabel = document.getElementById("frame-label");
const slider = document.getElementById("time-slider");

function buildTraces(start) {{
  const end = Math.min(DATA.n, start + DATA.window);
  const t = DATA.time.slice(start, end);
  const traces = [];
  const seriesKeys = Object.keys(DATA.series);
  for (let joint = 0; joint < 6; joint++) {{
    const axisSuffix = joint === 0 ? "" : String(joint + 1);
    for (const key of seriesKeys) {{
      const values = DATA.series[key].slice(start, end).map(row => row[joint]);
      traces.push({{
        x: t,
        y: values,
        type: "scatter",
        mode: "lines",
        name: key,
        legendgroup: key,
        showlegend: joint === 0,
        line: {{ color: DATA.colors[key], dash: DATA.dashes[key], width: 1.8 }},
        xaxis: "x" + axisSuffix,
        yaxis: "y" + axisSuffix,
        hovertemplate: key + " " + DATA.joint_types[joint] + ": %{{y:.2f}}<extra></extra>",
      }});
    }}
  }}
  return traces;
}}

function buildLayout() {{
  const layout = {{
    template: "plotly_white",
    height: 960,
    dragmode: "pan",
    grid: {{ rows: 3, columns: 2, pattern: "independent", roworder: "top to bottom" }},
    margin: {{ t: 40, b: 80 }},
    legend: {{ orientation: "h", y: -0.08, x: 0.5, xanchor: "center" }},
    annotations: [],
  }};
  for (let joint = 0; joint < 6; joint++) {{
    const suffix = joint === 0 ? "" : String(joint + 1);
    layout["xaxis" + suffix] = {{ title: "time (s)" }};
    layout["yaxis" + suffix] = {{ title: DATA.joint_types[joint].replace(/_/g, " ") }};
    if (joint >= 4) {{
      layout["xaxis" + suffix].rangeslider = {{ visible: true, thickness: 0.06 }};
    }}
    layout.annotations.push({{
      text: DATA.joint_types[joint].replace(/_/g, " "),
      xref: "paper",
      yref: "paper",
      x: (joint % 2) * 0.5 + 0.22,
      y: 1.0 - Math.floor(joint / 2) * 0.34,
      showarrow: false,
      font: {{ size: 13 }},
    }});
  }}
  return layout;
}}

function updateFrameLabel(start) {{
  const end = Math.min(DATA.n, start + DATA.window) - 1;
  const tStart = DATA.time[start].toFixed(2);
  const tEnd = DATA.time[end].toFixed(2);
  frameLabel.textContent = `frames ${{start}}-${{end}} / ${{DATA.n - 1}}   time ${{tStart}}s-${{tEnd}}s`;
}}

function render(start) {{
  updateFrameLabel(start);
  Plotly.react(plotDiv, buildTraces(start), buildLayout(), {{ responsive: true, scrollZoom: true }});
}}

slider.addEventListener("input", () => render(parseInt(slider.value, 10)));
render(0);

{video_js}
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.no_vla and args.no_rl:
        raise ValueError("At least one of VLA or RL must be enabled.")

    dataset = load_dataset(args.dataset_path, args.repo_id, args.episode_index)
    fps = dataset.meta.fps
    camera_keys = detect_camera_keys(dataset)
    camera_full_keys = [f"observation.images.{key}" for key in camera_keys]
    q01, q99 = load_action_quantiles(dataset)
    state_q01_np, state_q99_np = load_state_quantiles(dataset)
    state_q01 = torch.as_tensor(state_q01_np, dtype=torch.float32, device=args.device)
    state_q99 = torch.as_tensor(state_q99_np, dtype=torch.float32, device=args.device)
    gt_actions = extract_ground_truth(dataset, args.stride)
    gt_actions = gt_actions[:, :12]

    video_frames = None
    frame_size = (args.video_width, 240)
    if not args.no_video:
        video_frames, frame_size = extract_video_frames(dataset, camera_full_keys, args.stride, args.video_width)

    vla_actions = None
    if not args.no_vla:
        log.info("Loading VLA model: %s", args.vla_model)
        vla_policy = load_vla_adapter(
            args.vla_model,
            args.task,
            action_dim=gt_actions.shape[1],
            proprio_dim=dataset[0]["observation.state"].shape[0],
            device=args.device,
        )
        vla_norm = run_vla_inference(
            vla_policy,
            dataset,
            camera_full_keys,
            args.device,
            args.stride,
            state_q01,
            state_q99,
        )
        vla_actions = unnormalize_actions(vla_norm[:, :12], q01[:12], q99[:12])
        del vla_policy
        torch.cuda.empty_cache()

    rl_actions = None
    q_values = None
    if not args.no_rl:
        paths = resolve_rl_model_paths(args)
        log.info("Loading RL model with backbone=%s ac=%s", paths.vla_model, paths.ac_ckpt)
        rl_policy, critic = load_rl_policy(paths, args.task, args.device)
        rl_norm, q_values = run_rl_inference(
            rl_policy,
            critic,
            dataset,
            camera_full_keys,
            args.stride,
            args.device,
            state_q01,
            state_q99,
        )
        rl_actions = unnormalize_actions(rl_norm[:, :12], q01[:12], q99[:12])
        del rl_policy, critic
        torch.cuda.empty_cache()

    stats_html = build_stats_html(gt_actions, vla_actions, rl_actions)
    html = build_html(
        gt_actions,
        vla_actions,
        rl_actions,
        q_values,
        video_frames,
        camera_keys,
        stride=args.stride,
        fps=fps,
        window_size=args.window_size,
        stats_html=stats_html,
        title=f"Episode {args.episode_index}: GT vs VLA vs RL",
        frame_size=frame_size,
    )
    output_path = Path(args.output)
    output_path.write_text(html, encoding="utf-8")
    log.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
