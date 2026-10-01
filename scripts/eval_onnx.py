"""Deterministic evaluation of exported ONNX policies on a Microduck mjlab task.

Training-time returns are not comparable across algorithms: PPO logs rewards while sampling
from its exploration Gaussian, an off-policy learner late in training is near-deterministic.
This runs each policy's exported mean action (the thing the robot would run) in the same env.

--curriculum-step sets the env's step counter before evaluation, so step-based curricula
(push strength, CoM range, ...) sit at the stage they reach at that many env steps
(iteration x 24). 0 = start of training (no pushes), 36000 = end of a 1500-iteration run.

  uv run scripts/eval_onnx.py --onnx PPO=path/a.onnx --onnx FastSAC=path/b.onnx
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import onnxruntime as ort
import torch


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default="Mjlab-BallKick-Flat-MicroDuck")
  p.add_argument("--onnx", action="append", required=True, help="LABEL=path.onnx")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--episodes-per-env", type=int, default=2)
  p.add_argument("--curriculum-step", type=int, action="append", default=None)
  p.add_argument("--seed", type=int, default=123)
  p.add_argument("--video-dir", default=None, help="also record one episode per policy (env 0): mp4 + contact sheet")
  p.add_argument("--video-episodes", type=int, default=2)
  p.add_argument("--video-width", type=int, default=1920)
  p.add_argument("--video-height", type=int, default=1080)
  p.add_argument("--video-distance", type=float, default=0.8, help="camera distance to the trunk (m)")
  p.add_argument("--video-only", action="store_true", help="skip the metric batteries, just record")
  args = p.parse_args()
  curriculum_steps = args.curriculum_step or [0, 36000]

  import mjlab.tasks  # noqa: F401
  import mjlab_microduck.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  if args.video_dir:
    v = env_cfg.viewer
    v.distance, v.elevation, v.azimuth = args.video_distance, -12.0, 150.0
    v.width, v.height = args.video_width, args.video_height  # mjlab's default is 320x240
    v.max_extra_envs = 0  # only the tracked duck, no neighbours in the background
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array" if args.video_dir else None)
  steps_per_episode = env.max_episode_length

  from mjlab_microduck.tasks.mdp import _ball_kick_dir

  robot = env.scene["robot"]

  def trunk_tilt_deg() -> torch.Tensor:
    # Same quantity as the task's fell_over termination (which only fires past 70 deg).
    return torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)))

  def trunk_height() -> torch.Tensor:
    return robot.data.root_link_pos_w[:, 2]

  def ball_fwd_speed() -> torch.Tensor:
    vel_xy = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
    return torch.nan_to_num((vel_xy * _ball_kick_dir(env)).sum(dim=1), nan=0.0)

  print(f"{'policy':<10}{'curric.':>8}{'eps':>5}{'return':>8}{'kick':>6}{'fell>70%':>9}{'ball m/s':>9}"
        f"{'end tilt':>9}{'end h/h0':>9}{'upright end %':>14}{'time >30deg %':>14}{'max tilt p50/p95':>18}")
  for label, path in (o.split("=", 1) for o in args.onnx):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    batched = sess.get_inputs()[0].shape[0] != 1

    def act(obs: torch.Tensor) -> torch.Tensor:
      o = obs.detach().cpu().numpy().astype(np.float32)
      out = sess.run(None, {in_name: o})[0] if batched else np.concatenate(
        [sess.run(None, {in_name: o[i:i + 1]})[0] for i in range(o.shape[0])])
      return torch.from_numpy(out).to(obs.device)

    for cstep in ([] if args.video_only else curriculum_steps):
      torch.manual_seed(args.seed)
      env.common_step_counter = cstep
      obs = env.reset()[0]["actor"]
      ret = torch.zeros(args.num_envs, device=obs.device)
      done_count = torch.zeros(args.num_envs, device=obs.device)
      returns, fell, logs = [], 0, defaultdict(list)
      peak = torch.zeros(args.num_envs, device=obs.device)
      peaks = []
      h0 = trunk_height().clone()  # standing height at spawn, per env
      prev_tilt, prev_h = trunk_tilt_deg(), trunk_height()
      tilted_steps = torch.zeros(args.num_envs, device=obs.device)
      ep_steps = torch.zeros(args.num_envs, device=obs.device)
      end_tilt, end_hrel, tilted_frac = [], [], []
      max_tilt = torch.zeros(args.num_envs, device=obs.device)  # peak lean within the episode
      max_tilts = []
      for _ in range(steps_per_episode * args.episodes_per_env):
        obs_d, rew, term, trunc, extras = env.step(act(obs))
        obs = obs_d["actor"]
        env.common_step_counter = cstep  # hold the curriculum stage fixed
        ret += rew
        peak = torch.maximum(peak, ball_fwd_speed())
        done = term | trunc
        # Done envs are already reset, so their end-of-episode pose is last step's.
        tilt, h = trunk_tilt_deg(), trunk_height()
        tilted_steps += (torch.where(done, prev_tilt, tilt) > 30).float()
        max_tilt = torch.maximum(max_tilt, torch.where(done, prev_tilt, tilt))
        ep_steps += 1
        d = done & (done_count < args.episodes_per_env)
        if d.any():
          returns += ret[d].tolist()
          peaks += peak[d].tolist()
          end_tilt += prev_tilt[d].tolist()
          end_hrel += (prev_h[d] / h0[d]).tolist()
          tilted_frac += (tilted_steps[d] / ep_steps[d]).tolist()
          max_tilts += max_tilt[d].tolist()
          fell += int((term & d).sum())
          for k, v in extras.get("log", {}).items():
            logs[k].append(float(v))
        ret[term | trunc] = 0
        peak[term | trunc] = 0
        tilted_steps[done] = 0
        max_tilt[done] = 0
        ep_steps[done] = 0
        h0 = torch.where(done, h, h0)  # new spawn height for reset envs
        prev_tilt, prev_h = tilt, h
        done_count += (term | trunc).float()
        if bool((done_count >= args.episodes_per_env).all()):
          break
      kick = logs.get("Episode_Reward/ball_forward_velocity", [float("nan")])
      et, eh = np.array(end_tilt), np.array(end_hrel)
      upright_end = 100 * np.mean((et < 30) & (eh > 0.8))
      print(f"{label:<10}{cstep:>8}{len(returns):>5}{np.mean(returns):>8.1f}{np.mean(kick):>6.1f}"
            f"{100 * fell / max(len(returns), 1):>9.1f}{np.mean(peaks):>9.2f}{np.mean(et):>8.1f}\u00b0"
            f"{np.mean(eh):>9.2f}{upright_end:>14.1f}{100 * np.mean(tilted_frac):>14.1f}"
            f"{np.percentile(max_tilts, 50):>11.1f}/{np.percentile(max_tilts, 95):.1f}\u00b0", flush=True)
    if args.video_dir:
      record(env, act, label, args, steps_per_episode)
  env.close()


def record(env, act, label: str, args, steps_per_episode: int) -> None:
  """Roll out env 0 for a few episodes at curriculum step 0 and save an mp4 (every control step,
  50 fps, streamed to disk) plus a 4x4 frame grid of the first episode."""
  import imageio.v2 as imageio
  from pathlib import Path

  out = Path(args.video_dir)
  out.mkdir(parents=True, exist_ok=True)
  torch.manual_seed(args.seed)
  env.common_step_counter = 0
  obs = env.reset()[0]["actor"]
  grid_at = set(np.linspace(0, steps_per_episode - 1, 16).astype(int).tolist())
  grid = []
  writer = imageio.get_writer(out / f"{label}.mp4", fps=50, codec="libx264", quality=9,
                              pixelformat="yuv420p", macro_block_size=8)
  n = steps_per_episode * args.video_episodes
  for t in range(n):
    obs = env.step(act(obs))[0]["actor"]
    env.common_step_counter = 0
    frame = env.render()
    writer.append_data(frame)
    if t in grid_at:
      grid.append(frame[::4, ::4])  # grid tiles at quarter resolution
  writer.close()
  rows = [np.concatenate(grid[r * 4:(r + 1) * 4], axis=1) for r in range(4)]
  imageio.imwrite(out / f"{label}_grid.png", np.concatenate(rows, axis=0))
  print(f"[video] {out / (label + '.mp4')}  ({n} frames, {frame.shape[1]}x{frame.shape[0]})", flush=True)


if __name__ == "__main__":
  main()
