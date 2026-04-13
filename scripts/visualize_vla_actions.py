#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
TRAINING_ROOT = REPO_ROOT / "scripts" / "rlt_training"
for root in [SRC_ROOT, TRAINING_ROOT]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from common import build_pi05_policy, load_training_config

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
DEFAULT_VLA_MODEL = "Elvinky/pi05_screw_271ep_sft_fp32"
DEFAULT_HF_REPO = "Shiki42/rlt_pi0.5_screw"
DEFAULT_RL_TOKEN_PATH = "rl_token/271ep_pi0.5_screw_sft_rltoken/demo_adapt_checkpoint.pt"
DEFAULT_RL_CONFIG_PATH = "rl_token/271ep_pi0.5_screw_sft_rltoken/pi05_rlt.yaml"
DEFAULT_AC_PATH = "rl_token/271ep_pi0.5_screw_sft_rltoken/actor_critic/0412_278cp_warmup/rl_checkpoint.pt"
DEFAULT_AC_METRICS_PATH = "rl_token/271ep_pi0.5_screw_sft_rltoken/actor_critic/0412_278cp_warmup/metrics.json"
COLORS = {
    "gt_left": "#1f4e79",
    "gt_right": "#6fa8dc",
    "vla_left": "#7f6000",
    "vla_right": "#f6b26b",
    "rl_left": "#274e13",
    "rl_right": "#93c47d",
}


@dataclass
class RLModelPaths:
    rl_token_ckpt: str
    ac_ckpt: str
    config_path: str
    metrics_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GT/VLA/RL actions using the same inference path as RLT training.",
    )
    parser.add_argument("--dataset-path", required=True, help="Local LeRobot dataset root.")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=-1,
        help="-1 means choose the longest episode. Non-negative values use that episode index.",
    )
    parser.add_argument("--repo-id", default="rlt_viz")
    parser.add_argument("--output", default="action_viz.html")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=WINDOW)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--action-space",
        choices=["normalized", "raw"],
        default="normalized",
        help="Plot in training space or unnormalized physical space.",
    )
    parser.add_argument("--vla-model", default=DEFAULT_VLA_MODEL)
    parser.add_argument("--config", default="")
    parser.add_argument("--token-pool-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--no-rl", action="store_true")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--rl-token-ckpt", default="")
    parser.add_argument("--ac-ckpt", default="")
    parser.add_argument("--ac-metrics", default="")
    return parser.parse_args()


def resolve_hf_file(repo_id: str, path_in_repo: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=path_in_repo)


def resolve_rl_model_paths(args: argparse.Namespace) -> RLModelPaths:
    return RLModelPaths(
        rl_token_ckpt=args.rl_token_ckpt or resolve_hf_file(args.hf_repo, DEFAULT_RL_TOKEN_PATH),
        ac_ckpt=args.ac_ckpt or resolve_hf_file(args.hf_repo, DEFAULT_AC_PATH),
        config_path=args.config or resolve_hf_file(args.hf_repo, DEFAULT_RL_CONFIG_PATH),
        metrics_path=args.ac_metrics or resolve_hf_file(args.hf_repo, DEFAULT_AC_METRICS_PATH),
    )


def load_demo_dataset(dataset_path: str, repo_id: str, chunk_length: int):
    from lerobot.rlt.demo_loader import RLTDemoDataset

    return RLTDemoDataset(
        dataset_path=dataset_path,
        repo_id=repo_id,
        chunk_length=chunk_length,
        normalize_actions=True,
    )


def select_episode(dataset, episode_index: int) -> tuple[int, list[int]]:
    from lerobot.rlt.offline_dataset import _count_episodes, _episode_frame_range

    num_episodes = _count_episodes(dataset)
    if episode_index >= num_episodes:
        raise ValueError(f"episode_index={episode_index} out of range for {num_episodes} episodes")

    if episode_index >= 0:
        start, stop = _episode_frame_range(dataset, episode_index)
        return episode_index, list(range(start, stop))

    longest: tuple[int, int, int] | None = None
    for ep_idx in range(num_episodes):
        start, stop = _episode_frame_range(dataset, ep_idx)
        length = stop - start
        candidate = (length, ep_idx, start)
        if longest is None or candidate > longest:
            longest = candidate

    if longest is None:
        raise ValueError("dataset has no episodes")

    length, ep_idx, start = longest
    return ep_idx, list(range(start, start + length))


def load_action_quantiles(dataset) -> tuple[np.ndarray, np.ndarray]:
    stats = dataset._dataset.meta.stats["action"]
    q01 = stats["q01"].numpy() if isinstance(stats["q01"], torch.Tensor) else np.array(stats["q01"])
    q99 = stats["q99"].numpy() if isinstance(stats["q99"], torch.Tensor) else np.array(stats["q99"])
    return q01[:12], q99[:12]


def unnormalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    return (actions + 1.0) / 2.0 * (q99 - q01) + q01


def convert_action_space(actions: np.ndarray, action_space: str, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    if action_space == "normalized":
        return actions
    return unnormalize_actions(actions, q01, q99)


def load_ac_metadata(ac_ckpt_path: str, metrics_path: str) -> tuple[dict, dict]:
    from lerobot.rlt.utils import infer_actor_architecture

    ac_ckpt = torch.load(ac_ckpt_path, map_location="cpu", weights_only=False)
    metrics = json.loads(Path(metrics_path).read_text())
    inferred = infer_actor_architecture(ac_ckpt["actor_state_dict"])
    return ac_ckpt, metrics.get("config", {}) or inferred


def build_policy(args: argparse.Namespace, config_path: str, rl_paths: RLModelPaths | None):
    from lerobot.rlt.utils import filter_encoder_only, infer_actor_architecture

    config = load_training_config(config_path or None)
    if rl_paths is not None:
        ac_ckpt, ac_metadata = load_ac_metadata(rl_paths.ac_ckpt, rl_paths.metrics_path)
        inferred = infer_actor_architecture(ac_ckpt["actor_state_dict"])
        config.actor.hidden_dim = int(ac_metadata.get("actor_hidden", inferred["hidden_dim"]))
        config.actor.num_layers = int(ac_metadata.get("actor_layers", inferred["num_layers"]))
        config.actor.activation = str(ac_metadata.get("actor_activation", inferred["activation"]))
        config.actor.layer_norm = bool(ac_metadata.get("actor_layer_norm", inferred["layer_norm"]))
        config.actor.residual = bool(ac_metadata.get("actor_residual", inferred["residual"]))
        config.actor.fixed_std = float(ac_metadata.get("fixed_std", inferred["fixed_std"]))
        config.actor.ref_dropout_p = float(ac_metadata.get("ref_dropout_p", inferred["ref_dropout_p"]))
        rl_token_ckpt = rl_paths.rl_token_ckpt
    else:
        ac_ckpt = None
        rl_token_ckpt = None

    policy = build_pi05_policy(
        config=config,
        model_path=args.vla_model,
        task_instruction=args.task,
        device=args.device,
        token_pool_size=args.token_pool_size,
        dtype=args.dtype,
        rl_token_checkpoint=rl_token_ckpt,
    )

    if ac_ckpt is not None:
        policy.actor.load_state_dict(ac_ckpt["actor_state_dict"])
        if "rl_token_state_dict" in ac_ckpt:
            filtered, _ = filter_encoder_only(ac_ckpt["rl_token_state_dict"])
            policy.rl_token.load_state_dict(filtered, strict=False)

    policy.freeze_vla()
    policy.freeze_rl_token_encoder()
    policy.eval()
    return policy, config


def build_observation(item: dict, device: str):
    from lerobot.rlt.interfaces import Observation

    images = {
        key: item[key].unsqueeze(0).to(device)
        for key in item
        if key not in ("proprio", "expert_actions")
    }
    proprio = item["proprio"].unsqueeze(0).to(device)
    return Observation(images=images, proprio=proprio)


def run_episode_inference(
    policy,
    dataset,
    frame_indices: list[int],
    chunk_length: int,
    stride: int,
    device: str,
    include_rl: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    gt_step0 = []
    vla_step0 = []
    rl_step0 = []
    chunk_ref_mse = []
    step0_ref_mse = []

    sampled_indices = frame_indices[::stride]
    log.info("Episode frames=%d, sampled=%d, stride=%d", len(frame_indices), len(sampled_indices), stride)
    start_time = time.monotonic()
    for frame_idx in tqdm(sampled_indices, desc="episode"):
        item = dataset[frame_idx]
        obs = build_observation(item, device)
        expert_chunk = item["expert_actions"][:chunk_length, :12]
        gt_step0.append(expert_chunk[0].cpu().numpy())

        with torch.inference_mode():
            if include_rl:
                _, actor_chunk, _, ref_chunk = policy.select_action(obs, deterministic=True)
            else:
                ref_chunk = policy.get_reference_chunk(obs)
                actor_chunk = None

        ref_chunk_cpu = ref_chunk[0].cpu()
        vla_step0.append(ref_chunk_cpu[0].numpy())

        if actor_chunk is not None:
            actor_chunk_cpu = actor_chunk[0].cpu()
            rl_step0.append(actor_chunk_cpu[0].numpy())
            chunk_ref_mse.append(((actor_chunk_cpu - ref_chunk_cpu) ** 2).mean().item())
            step0_ref_mse.append(((actor_chunk_cpu[0] - ref_chunk_cpu[0]) ** 2).mean().item())

    elapsed = time.monotonic() - start_time
    log.info("Inference done in %.1fs (%.0f ms/frame)", elapsed, elapsed / max(len(sampled_indices), 1) * 1000)

    gt_arr = np.array(gt_step0)
    vla_arr = np.array(vla_step0)
    rl_arr = np.array(rl_step0) if rl_step0 else None
    chunk_mse_arr = np.array(chunk_ref_mse) if chunk_ref_mse else None
    step0_mse_arr = np.array(step0_ref_mse) if step0_ref_mse else None
    return gt_arr, vla_arr, rl_arr, chunk_mse_arr, step0_mse_arr


def build_stats_html(
    *,
    gt: np.ndarray,
    vla: np.ndarray,
    rl: np.ndarray | None,
    chunk_ref_mse: np.ndarray | None,
    step0_ref_mse: np.ndarray | None,
    episode_index: int,
    total_frames: int,
    sampled_frames: int,
    action_space: str,
    vla_model: str,
    ac_ckpt: str | None,
) -> str:
    rows = []
    for joint_index, joint_name in enumerate(JOINT_TYPES):
        left_index = joint_index
        right_index = joint_index + 6
        row = [f"<td>{joint_name}</td>"]
        for label, array in [("GT-L", gt), ("GT-R", gt), ("VLA-L", vla), ("VLA-R", vla), ("RL-L", rl), ("RL-R", rl)]:
            if array is None:
                row.append("<td>-</td><td>-</td><td>-</td>")
                continue
            dim_index = left_index if label.endswith("-L") else right_index
            mae = np.abs(array[:, dim_index] - gt[:, dim_index]).mean()
            row.append(f"<td>{array[:, dim_index].mean():.3f}</td>")
            row.append(f"<td>{array[:, dim_index].std():.3f}</td>")
            row.append(f"<td>{mae:.3f}</td>")
        rows.append("<tr>" + "".join(row) + "</tr>")

    header_cells = [
        "<th>Joint</th>",
        "<th>GT-L μ</th><th>GT-L σ</th><th>GT-L MAE</th>",
        "<th>GT-R μ</th><th>GT-R σ</th><th>GT-R MAE</th>",
        "<th>VLA-L μ</th><th>VLA-L σ</th><th>VLA-L MAE</th>",
        "<th>VLA-R μ</th><th>VLA-R σ</th><th>VLA-R MAE</th>",
        "<th>RL-L μ</th><th>RL-L σ</th><th>RL-L MAE</th>",
        "<th>RL-R μ</th><th>RL-R σ</th><th>RL-R MAE</th>",
    ]
    summary_lines = [
        f"<p><b>episode</b>: {episode_index} &nbsp; <b>frames</b>: {total_frames} &nbsp; <b>sampled</b>: {sampled_frames}</p>",
        f"<p><b>action_space</b>: {action_space} &nbsp; <b>vla_model</b>: {vla_model}</p>",
    ]
    if ac_ckpt is not None and chunk_ref_mse is not None and step0_ref_mse is not None:
        summary_lines.append(
            "<p><b>mean chunk ref_mse</b>: "
            f"{chunk_ref_mse.mean():.6f} &nbsp; <b>mean step0 ref_mse</b>: {step0_ref_mse.mean():.6f}</p>"
        )
        summary_lines.append(f"<p><b>ac_ckpt</b>: {ac_ckpt}</p>")

    return (
        '<div style="font-family:sans-serif;">'
        + "".join(summary_lines)
        + '<table style="border-collapse:collapse;font-family:monospace;font-size:12px;">'
        + f'<thead style="background:#f0f0f0;"><tr>{"".join(header_cells)}</tr></thead>'
        + f'<tbody>{"".join(rows)}</tbody></table>'
        + "</div>"
    )


def build_html(
    *,
    gt: np.ndarray,
    vla: np.ndarray,
    rl: np.ndarray | None,
    fps: float,
    stride: int,
    window_size: int,
    stats_html: str,
    title: str,
) -> str:
    num_frames = gt.shape[0]
    time_axis = (np.arange(num_frames) * stride / fps).tolist()
    data = {
        "time": time_axis,
        "window": window_size,
        "num_frames": num_frames,
        "gt": gt.tolist(),
        "vla": vla.tolist(),
        "rl": rl.tolist() if rl is not None else None,
        "joint_types": JOINT_TYPES,
        "colors": COLORS,
    }
    data_json = json.dumps(data)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
#plot {{ width: 100%; height: 920px; }}
#slider-container {{ margin: 16px 24px; }}
#frame-slider {{ width: 100%; }}
#frame-info {{ text-align: center; margin-bottom: 8px; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div id="plot"></div>
<div id="slider-container">
  <div id="frame-info"></div>
  <input type="range" id="frame-slider" min="0" max="{max(0, num_frames - window_size)}" value="0" step="1">
</div>
{stats_html}
<script>
const D = {data_json};
const plotDiv = document.getElementById("plot");
const slider = document.getElementById("frame-slider");
const info = document.getElementById("frame-info");

function trace(name, x, y, color, dash, axisIndex, showlegend) {{
  const suffix = axisIndex === 1 ? "" : String(axisIndex);
  return {{
    x: x,
    y: y,
    type: "scatter",
    mode: "lines",
    name: name,
    line: {{color: color, dash: dash, width: 1.6}},
    xaxis: "x" + suffix,
    yaxis: "y" + suffix,
    showlegend: showlegend,
  }};
}}

function buildTraces(start) {{
  const end = Math.min(start + D.window, D.num_frames);
  const x = D.time.slice(start, end);
  const traces = [];
  for (let joint = 0; joint < 6; joint++) {{
    const axisIndex = joint + 1;
    const left = joint;
    const right = joint + 6;
    const gt = D.gt.slice(start, end);
    const vla = D.vla.slice(start, end);
    traces.push(trace("gt-left", x, gt.map(row => row[left]), D.colors.gt_left, "solid", axisIndex, joint === 0));
    traces.push(trace("gt-right", x, gt.map(row => row[right]), D.colors.gt_right, "solid", axisIndex, joint === 0));
    traces.push(trace("vla-left", x, vla.map(row => row[left]), D.colors.vla_left, "solid", axisIndex, joint === 0));
    traces.push(trace("vla-right", x, vla.map(row => row[right]), D.colors.vla_right, "solid", axisIndex, joint === 0));
    if (D.rl !== null) {{
      const rl = D.rl.slice(start, end);
      traces.push(trace("rl-left", x, rl.map(row => row[left]), D.colors.rl_left, "solid", axisIndex, joint === 0));
      traces.push(trace("rl-right", x, rl.map(row => row[right]), D.colors.rl_right, "solid", axisIndex, joint === 0));
    }}
  }}
  return traces;
}}

function buildLayout() {{
  const layout = {{
    height: 920,
    template: "plotly_white",
    grid: {{rows: 3, columns: 2, pattern: "independent", roworder: "top to bottom"}},
    margin: {{l: 60, r: 20, t: 30, b: 60}},
    legend: {{orientation: "h", y: -0.08, x: 0.5, xanchor: "center"}},
    annotations: [],
  }};
  for (let joint = 0; joint < 6; joint++) {{
    const axisIndex = joint + 1;
    const suffix = axisIndex === 1 ? "" : String(axisIndex);
    layout["xaxis" + suffix] = {{title: "time (s)"}};
    layout["yaxis" + suffix] = {{title: D.joint_types[joint]}};
    const col = joint % 2;
    const row = Math.floor(joint / 2);
    layout.annotations.push({{
      text: D.joint_types[joint],
      xref: "paper",
      yref: "paper",
      x: col * 0.5 + 0.22,
      y: 1.0 - row * 0.34,
      showarrow: false,
      font: {{size: 13}},
    }});
  }}
  return layout;
}}

function updateFrameInfo(start) {{
  const end = Math.min(start + D.window, D.num_frames) - 1;
  const t0 = D.time[start].toFixed(2);
  const t1 = D.time[end].toFixed(2);
  info.textContent =
    "frame " + start + " - " + end + " / " + (D.num_frames - 1) +
    " (" + t0 + "s - " + t1 + "s)";
}}

function render(start) {{
  updateFrameInfo(start);
  Plotly.react(plotDiv, buildTraces(start), buildLayout(), {{responsive: true}});
}}

slider.addEventListener("input", () => render(parseInt(slider.value, 10)));
render(0);
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rl_paths = None if args.no_rl else resolve_rl_model_paths(args)
    config_path = rl_paths.config_path if rl_paths is not None else args.config or None
    policy, config = build_policy(args, config_path, rl_paths)

    dataset = load_demo_dataset(args.dataset_path, args.repo_id, config.vla_horizon)
    episode_index, frame_indices = select_episode(dataset, args.episode_index)
    q01, q99 = load_action_quantiles(dataset)
    fps = dataset._dataset.meta.fps

    log.info("Using episode %d", episode_index)
    gt, vla, rl, chunk_ref_mse, step0_ref_mse = run_episode_inference(
        policy=policy,
        dataset=dataset,
        frame_indices=frame_indices,
        chunk_length=config.chunk_length,
        stride=args.stride,
        device=args.device,
        include_rl=rl_paths is not None,
    )

    gt_plot = convert_action_space(gt, args.action_space, q01, q99)
    vla_plot = convert_action_space(vla, args.action_space, q01, q99)
    rl_plot = None if rl is None else convert_action_space(rl, args.action_space, q01, q99)

    stats_html = build_stats_html(
        gt=gt_plot,
        vla=vla_plot,
        rl=rl_plot,
        chunk_ref_mse=chunk_ref_mse,
        step0_ref_mse=step0_ref_mse,
        episode_index=episode_index,
        total_frames=len(frame_indices),
        sampled_frames=len(gt_plot),
        action_space=args.action_space,
        vla_model=args.vla_model,
        ac_ckpt=None if rl_paths is None else rl_paths.ac_ckpt,
    )
    html = build_html(
        gt=gt_plot,
        vla=vla_plot,
        rl=rl_plot,
        fps=fps,
        stride=args.stride,
        window_size=args.window_size,
        stats_html=stats_html,
        title=f"Episode {episode_index}: GT vs VLA vs RL ({args.action_space})",
    )
    Path(args.output).write_text(html, encoding="utf-8")
    log.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
