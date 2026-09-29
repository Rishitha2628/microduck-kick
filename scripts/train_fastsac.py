"""FastSAC on a Microduck mjlab task — the entropy-regularized sibling of train_fasttd3.py.

Same massively-parallel off-policy recipe as FastTD3 (n-step returns, C51 clipped-double-Q
critic, asymmetric critic obs, AdamW, empirical obs normalization, large batches) and the
same replay/critic/export code, imported from train_fasttd3.py. What changes is the actor:

  - a squashed-Gaussian policy (tanh(u) * bound) sampled for exploration, instead of a
    deterministic actor plus hand-set per-env noise;
  - an entropy bonus alpha * H(pi) in both the critic target and the actor loss;
  - alpha tuned automatically toward a target entropy.

The ONNX export is the deterministic mean action, so it keeps the [1,61] -> [1,14] contract.

Usage:
  uv run scripts/train_fastsac.py --task Mjlab-BallKick-Flat-MicroDuck --num-envs 1024 \
      --total-env-steps 36000 --max-minutes 235
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train_fasttd3 import DistributionalCritic, EmpiricalNormalizer, NStepReplay, export_onnx, mlp


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", default="Mjlab-BallKick-Flat-MicroDuck")
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--total-env-steps", type=int, default=36_000, help="steps per env (PPO: iters x 24)")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--buffer-per-env", type=int, default=1000)
  p.add_argument("--batch-size", type=int, default=32768)
  p.add_argument("--learning-starts", type=int, default=10, help="env steps before updates begin")
  p.add_argument("--updates-per-step", type=int, default=2)
  p.add_argument("--policy-frequency", type=int, default=2)
  p.add_argument("--n-step", type=int, default=3)
  p.add_argument("--gamma", type=float, default=0.99)
  p.add_argument("--tau", type=float, default=0.1)
  p.add_argument("--actor-lr", type=float, default=3e-4)
  p.add_argument("--critic-lr", type=float, default=3e-4)
  p.add_argument("--alpha-lr", type=float, default=3e-4)
  p.add_argument("--weight-decay", type=float, default=0.1)
  p.add_argument("--alpha-init", type=float, default=0.01,
                 help="small because mjlab rewards are dt-scaled (~0.1-0.3 per step)")
  p.add_argument("--target-entropy-per-dim", type=float, default=-1.0, help="target H = this * act_dim")
  p.add_argument("--log-std-min", type=float, default=-5.0)
  p.add_argument("--log-std-max", type=float, default=0.5)
  p.add_argument("--init-log-std", type=float, default=0.0, help="initial pre-tanh log std (0 -> std 1, as PPO)")
  p.add_argument("--v-min", type=float, default=-20.0)
  p.add_argument("--v-max", type=float, default=50.0)
  p.add_argument("--num-atoms", type=int, default=101)
  p.add_argument("--action-bound", type=float, default=1.0, help="tanh bound, in the env's action units (rad)")
  p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
  p.add_argument("--log-interval", type=int, default=24, help="env steps between log lines (= one PPO iteration)")
  p.add_argument("--save-interval", type=int, default=2400)
  p.add_argument("--max-minutes", type=float, default=None, help="stop (and export) after this much wall time")
  p.add_argument("--run-name", default="fastsac")
  return p.parse_args()


class GaussianActor(nn.Module):
  """tanh-squashed Gaussian. forward() is the deterministic mean action (used for export)."""

  def __init__(self, obs_dim: int, act_dim: int, bound: float, log_std_min: float, log_std_max: float,
               init_log_std: float = 0.0):
    super().__init__()
    self.trunk = mlp(obs_dim, (512, 256), 128)
    self.mu = nn.Linear(128, act_dim)
    self.log_std = nn.Linear(128, act_dim)
    self.bound, self.log_std_min, self.log_std_max = bound, log_std_min, log_std_max
    # Start at std = exp(init_log_std) like PPO's init_std, not at the range midpoint
    # (~0.1 for [-5, 0.5]), which explores too little to find the kick.
    x = 2 * (init_log_std - log_std_min) / (log_std_max - log_std_min) - 1
    with torch.no_grad():
      self.log_std.weight.mul_(0.01)
      self.log_std.bias.fill_(math.atanh(x))

  def _dist_params(self, obs_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = F.relu(self.trunk(obs_n))
    log_std = torch.tanh(self.log_std(h))  # smooth squash into [min, max]
    log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
    return self.mu(h), log_std

  def forward(self, obs_n: torch.Tensor) -> torch.Tensor:
    mu, _ = self._dist_params(obs_n)
    return torch.tanh(mu) * self.bound

  def sample(self, obs_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterized action and its log-prob (tanh change-of-variables included)."""
    mu, log_std = self._dist_params(obs_n)
    mu, log_std = mu.float(), log_std.float()
    std = log_std.exp()
    u = mu + std * torch.randn_like(mu)
    a = torch.tanh(u)
    log_prob = (-0.5 * ((u - mu) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_prob -= (torch.log(self.bound * (1 - a.pow(2)) + 1e-6)).sum(-1)
    return a * self.bound, log_prob


def main() -> None:
  args = parse_args()
  torch.manual_seed(args.seed)
  device = torch.device("cuda:0")

  import mjlab.tasks  # noqa: F401  (registers tasks / plugins)
  import mjlab_microduck.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))
  obs_dict, _ = env.reset()
  obs, cobs = obs_dict["actor"], obs_dict["critic"]
  n_envs, obs_dim, cobs_dim = obs.shape[0], obs.shape[1], cobs.shape[1]
  act_dim = env.action_manager.total_action_dim
  print(f"[fastsac] envs={n_envs} actor_obs={obs_dim} critic_obs={cobs_dim} act={act_dim}")

  actor = GaussianActor(obs_dim, act_dim, args.action_bound, args.log_std_min, args.log_std_max,
                        args.init_log_std).to(device)
  critic = DistributionalCritic(cobs_dim, act_dim, args.num_atoms, args.v_min, args.v_max).to(device)
  critic_target = DistributionalCritic(cobs_dim, act_dim, args.num_atoms, args.v_min, args.v_max).to(device)
  critic_target.load_state_dict(critic.state_dict())
  obs_norm, cobs_norm = EmpiricalNormalizer(obs_dim).to(device), EmpiricalNormalizer(cobs_dim).to(device)
  log_alpha = torch.tensor(math.log(args.alpha_init), device=device, requires_grad=True)
  target_entropy = args.target_entropy_per_dim * act_dim
  actor_opt = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay)
  critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
  alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
  amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp)

  replay = NStepReplay(args.buffer_per_env, n_envs, obs_dim, cobs_dim, act_dim, device)

  log_dir = Path("logs/fastsac") / args.task / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.run_name}")
  log_dir.mkdir(parents=True, exist_ok=True)
  (log_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
  writer = SummaryWriter(str(log_dir))
  print(f"[fastsac] logging to {log_dir}")

  ep_ret = torch.zeros(n_envs, device=device)
  ep_len = torch.zeros(n_envs, device=device)
  finished_ret, finished_len = [], []
  env_logs: dict[str, list[float]] = defaultdict(list)
  losses: dict[str, list[float]] = defaultdict(list)
  update_count = 0
  t_start = time.time()
  # SIGINT (Ctrl-C / the watchdog's graceful stop) ends the loop so the run still saves + exports.
  stop_requested = {"flag": False}
  signal.signal(signal.SIGINT, lambda *_: stop_requested.__setitem__("flag", True))

  def save(tag: str) -> None:
    torch.save({
      "actor": actor.state_dict(), "critic": critic.state_dict(),
      "critic_target": critic_target.state_dict(), "log_alpha": log_alpha.detach(),
      "obs_norm": obs_norm.state_dict(), "cobs_norm": cobs_norm.state_dict(),
      "actor_opt": actor_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
      "alpha_opt": alpha_opt.state_dict(), "env_step": step, "args": vars(args),
    }, log_dir / f"model_{tag}.pt")

  for step in range(1, args.total_env_steps + 1):
    # ── act: sample from the policy (its own std is the exploration) ──
    with torch.no_grad():
      obs_norm.update(obs)
      cobs_norm.update(cobs)
      if step <= args.learning_starts:
        act = (torch.rand(n_envs, act_dim, device=device) * 2 - 1) * args.action_bound
      else:
        with amp:
          act, _ = actor.sample(obs_norm(obs))

    next_obs_dict, rew, term, trunc, extras = env.step(act)
    replay.add(obs, cobs, act, rew, term, trunc)
    obs, cobs = next_obs_dict["actor"], next_obs_dict["critic"]

    ep_ret += rew
    ep_len += 1
    done = term | trunc
    if done.any():
      d = done.nonzero().squeeze(-1)
      finished_ret += ep_ret[d].tolist()
      finished_len += ep_len[d].tolist()
      ep_ret[d] = 0
      ep_len[d] = 0
    for k, v in extras.get("log", {}).items():
      env_logs[k].append(float(v))

    # ── learn ──
    if step > args.learning_starts and replay.size > args.n_step + 1:
      for _ in range(args.updates_per_step):
        update_count += 1
        alpha = log_alpha.exp().detach()
        b = replay.sample(args.batch_size, args.n_step, args.gamma)
        with torch.no_grad():
          o_n, co_n = obs_norm(b["obs"]), cobs_norm(b["cobs"])
          no_n, nco_n = obs_norm(b["next_obs"]), cobs_norm(b["next_cobs"])
          with amp:
            na, nlogp = actor.sample(no_n)
            tq1, tq2 = critic_target(nco_n, na)
          # Soft target: returns + discount * (Z - alpha * log pi) == (returns - discount*alpha*logp) + discount*Z
          soft_returns = b["returns"] - b["discount"] * alpha * nlogp
          p1 = critic_target.project(tq1, soft_returns, b["discount"])
          p2 = critic_target.project(tq2, soft_returns, b["discount"])
          use1 = (critic_target.value(tq1) < critic_target.value(tq2)).unsqueeze(1)
          target = torch.where(use1, p1, p2)
        with amp:
          q1, q2 = critic(co_n, b["act"])
        ce = lambda logits: -(target * F.log_softmax(logits.float(), -1)).sum(-1)
        critic_loss = ((ce(q1) + ce(q2)) * b["mask"]).sum() / b["mask"].sum().clamp(min=1)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()
        losses["critic_loss"].append(critic_loss.item())
        losses["q_mean"].append(critic.value(q1.detach()).mean().item())

        if update_count % args.policy_frequency == 0:
          with amp:
            a, logp = actor.sample(o_n)
            aq1, aq2 = critic(co_n, a)
          actor_loss = (alpha * logp - torch.minimum(critic.value(aq1), critic.value(aq2))).mean()
          actor_opt.zero_grad(set_to_none=True)
          actor_loss.backward()
          actor_opt.step()
          alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
          alpha_opt.zero_grad(set_to_none=True)
          alpha_loss.backward()
          alpha_opt.step()
          losses["actor_loss"].append(actor_loss.item())
          losses["entropy"].append(-logp.detach().mean().item())
          losses["alpha"].append(log_alpha.exp().item())

        with torch.no_grad():
          for p, tp in zip(critic.parameters(), critic_target.parameters()):
            tp.lerp_(p, args.tau)

    # ── log ──
    if step % args.log_interval == 0:
      it = step // args.log_interval
      elapsed = time.time() - t_start
      if finished_ret:
        writer.add_scalar("Train/mean_reward", sum(finished_ret) / len(finished_ret), it)
        writer.add_scalar("Train/mean_episode_length", sum(finished_len) / len(finished_len), it)
      for k, v in env_logs.items():
        writer.add_scalar(k, sum(v) / len(v), it)
      for k, v in losses.items():
        writer.add_scalar(f"Loss/{k}", sum(v) / len(v), it)
      writer.add_scalar("Perf/wall_time_s", elapsed, it)
      writer.add_scalar("Perf/env_steps_total", step * n_envs, it)
      mean = lambda v: sum(v) / len(v) if v else float("nan")
      eta = elapsed / step * (args.total_env_steps - step)
      print(f"Learning iteration {it}/{args.total_env_steps // args.log_interval} | "
            f"Mean reward: {mean(finished_ret):.2f} | Mean episode length: {mean(finished_len):.1f} | "
            f"ball_forward_velocity: {mean(env_logs.get('Episode_Reward/ball_forward_velocity', [])):.4f} | "
            f"fell_over: {mean(env_logs.get('Episode_Termination/fell_over', [])):.4f} | "
            f"critic_loss: {mean(losses['critic_loss']):.4f} | entropy: {mean(losses['entropy']):.2f} | "
            f"alpha: {mean(losses['alpha']):.5f} | elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min",
            flush=True)
      finished_ret.clear(); finished_len.clear(); env_logs.clear(); losses.clear()

    if step % args.save_interval == 0:
      save(str(step // args.log_interval))

    if stop_requested["flag"]:
      print(f"[fastsac] interrupted at env step {step}", flush=True)
      break
    if args.max_minutes is not None and time.time() - t_start > args.max_minutes * 60:
      print(f"[fastsac] wall-time budget of {args.max_minutes} min reached at env step {step}", flush=True)
      break

  save("final")
  export_onnx(actor, obs_norm, obs_dim, log_dir / f"{args.run_name}.onnx", device)
  writer.close()
  env.close()
  print(f"[fastsac] done in {(time.time() - t_start) / 60:.1f} min -> {log_dir}")


if __name__ == "__main__":
  main()
