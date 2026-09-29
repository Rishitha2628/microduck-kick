"""FastTD3 on a Microduck mjlab task — an off-policy baseline to compare against rsl_rl PPO.

Follows FastTD3 (Seo et al., 2025): many parallel envs, large batches, n-step returns,
a distributional (C51) clipped-double-Q critic, per-env exploration noise, AdamW with
weight decay, and empirical observation normalization. The critic is asymmetric: it sees
the env's "critic" observation group, the actor only the 61D "actor" group, exactly like
the PPO setup, so the exported ONNX has the same [1,61] -> [1,14] contract.

The env is driven with auto_reset on (mjlab's default), so the next-observation of a
done transition is the post-reset one. Terminated transitions do not bootstrap, so that
is harmless; n-step windows that end in a time-out are masked out of the critic loss
(~n / episode_length of samples).

Usage:
  uv run scripts/train_fasttd3.py --task Mjlab-BallKick-Flat-MicroDuck --num-envs 1024 \
      --total-env-steps 36000
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
  p.add_argument("--weight-decay", type=float, default=0.1)
  p.add_argument("--v-min", type=float, default=-20.0)
  p.add_argument("--v-max", type=float, default=50.0)
  p.add_argument("--num-atoms", type=int, default=101)
  p.add_argument("--std-min", type=float, default=0.001)
  p.add_argument("--std-max", type=float, default=0.4)
  p.add_argument("--policy-noise", type=float, default=0.001)
  p.add_argument("--noise-clip", type=float, default=0.5)
  p.add_argument("--action-bound", type=float, default=1.0, help="tanh bound, in the env's action units (rad)")
  p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
  p.add_argument("--log-interval", type=int, default=24, help="env steps between log lines (= one PPO iteration)")
  p.add_argument("--save-interval", type=int, default=2400)
  p.add_argument("--max-minutes", type=float, default=None, help="stop (and export) after this much wall time")
  p.add_argument("--run-name", default="fasttd3")
  return p.parse_args()


# ── Networks ────────────────────────────────────────────────────────────────


class EmpiricalNormalizer(nn.Module):
  """Running mean/std over every observation seen in rollouts (not replay samples)."""

  def __init__(self, dim: int, eps: float = 1e-2, clip: float = 10.0):
    super().__init__()
    self.eps, self.clip = eps, clip
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros(()))

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    n = x.shape[0]
    batch_mean, batch_var = x.mean(0), x.var(0, unbiased=False)
    total = self.count + n
    delta = batch_mean - self.mean
    self.mean += delta * n / total
    self.var = (self.var * self.count + batch_var * n + delta.pow(2) * self.count * n / total) / total
    self.count = total

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  for h in hidden:
    layers += [nn.Linear(in_dim, h), nn.ReLU()]
    in_dim = h
  layers.append(nn.Linear(in_dim, out_dim))
  return nn.Sequential(*layers)


class Actor(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int, bound: float):
    super().__init__()
    self.net = mlp(obs_dim, (512, 256, 128), act_dim)
    self.bound = bound

  def forward(self, obs_n: torch.Tensor) -> torch.Tensor:
    return torch.tanh(self.net(obs_n)) * self.bound


class DistributionalCritic(nn.Module):
  """Two C51 Q-heads over a fixed support."""

  def __init__(self, obs_dim: int, act_dim: int, num_atoms: int, v_min: float, v_max: float):
    super().__init__()
    self.q1 = mlp(obs_dim + act_dim, (1024, 512, 256), num_atoms)
    self.q2 = mlp(obs_dim + act_dim, (1024, 512, 256), num_atoms)
    self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
    self.v_min, self.v_max, self.num_atoms = v_min, v_max, num_atoms
    self.delta_z = (v_max - v_min) / (num_atoms - 1)

  def forward(self, obs_n: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([obs_n, act], -1)
    return self.q1(x), self.q2(x)

  def value(self, logits: torch.Tensor) -> torch.Tensor:
    return (F.softmax(logits.float(), -1) * self.support).sum(-1)

  @torch.no_grad()
  def project(self, logits: torch.Tensor, returns: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
    """Project the distribution of `returns + discount * Z` back onto the support."""
    probs = F.softmax(logits.float(), -1)
    tz = (returns.unsqueeze(1) + discount.unsqueeze(1) * self.support).clamp(self.v_min, self.v_max)
    b = (tz - self.v_min) / self.delta_z
    lo, hi = b.floor().long(), b.ceil().long()
    # When b lands exactly on an atom, lo == hi and both weights below are 0: nudge apart.
    lo[(hi > 0) & (lo == hi)] -= 1
    hi[(lo < self.num_atoms - 1) & (lo == hi)] += 1
    out = torch.zeros_like(probs)
    out.scatter_add_(1, lo, probs * (hi.float() - b))
    out.scatter_add_(1, hi, probs * (b - lo.float()))
    return out


# ── Replay ──────────────────────────────────────────────────────────────────


class NStepReplay:
  """Per-env circular buffer on the GPU; n-step targets are assembled at sample time."""

  def __init__(self, cap: int, n_envs: int, obs_dim: int, cobs_dim: int, act_dim: int, device):
    self.cap, self.n_envs, self.device = cap, n_envs, device
    z = lambda *s: torch.zeros(cap, n_envs, *s, device=device)
    self.obs, self.cobs, self.act = z(obs_dim), z(cobs_dim), z(act_dim)
    self.rew, self.term, self.trunc = z(), z(), z()
    self.ptr, self.size = 0, 0

  def add(self, obs, cobs, act, rew, term, trunc) -> None:
    i = self.ptr
    self.obs[i], self.cobs[i], self.act[i] = obs, cobs, act
    self.rew[i], self.term[i], self.trunc[i] = rew, term.float(), trunc.float()
    self.ptr = (self.ptr + 1) % self.cap
    self.size = min(self.size + 1, self.cap)

  def sample(self, batch: int, n: int, gamma: float):
    # Rows t whose t+1..t+n are already written: skip the newest n rows.
    oldest = (self.ptr - self.size) % self.cap
    k = torch.randint(0, self.size - n, (batch,), device=self.device)
    t = (oldest + k) % self.cap
    e = torch.randint(0, self.n_envs, (batch,), device=self.device)
    steps = (t.unsqueeze(1) + torch.arange(n, device=self.device)) % self.cap  # [B, n]
    ee = e.unsqueeze(1).expand_as(steps)
    r, term, trunc = self.rew[steps, ee], self.term[steps, ee], self.trunc[steps, ee]
    done = torch.maximum(term, trunc)
    # alive[:, i] = step i of the window still belongs to the sampled episode.
    alive = torch.cat([torch.ones_like(done[:, :1]), torch.cumprod(1 - done, 1)[:, :-1]], 1)
    disc = gamma ** torch.arange(n, device=self.device, dtype=torch.float32)
    returns = (alive * disc * r).sum(1)
    m = alive.sum(1)  # steps included
    ended_term = (alive * term).amax(1)
    ended_trunc = (alive * trunc * (1 - term)).amax(1)
    boot_t = (t + m.long()) % self.cap
    return dict(
      obs=self.obs[t, e], cobs=self.cobs[t, e], act=self.act[t, e],
      next_obs=self.obs[boot_t, e], next_cobs=self.cobs[boot_t, e],
      returns=returns, discount=(gamma ** m) * (1 - ended_term), mask=1 - ended_trunc,
    )


# ── Training ────────────────────────────────────────────────────────────────


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
  print(f"[fasttd3] envs={n_envs} actor_obs={obs_dim} critic_obs={cobs_dim} act={act_dim}")

  actor = Actor(obs_dim, act_dim, args.action_bound).to(device)
  actor_target = Actor(obs_dim, act_dim, args.action_bound).to(device)
  actor_target.load_state_dict(actor.state_dict())
  critic = DistributionalCritic(cobs_dim, act_dim, args.num_atoms, args.v_min, args.v_max).to(device)
  critic_target = DistributionalCritic(cobs_dim, act_dim, args.num_atoms, args.v_min, args.v_max).to(device)
  critic_target.load_state_dict(critic.state_dict())
  obs_norm, cobs_norm = EmpiricalNormalizer(obs_dim).to(device), EmpiricalNormalizer(cobs_dim).to(device)
  actor_opt = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay)
  critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
  amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp)

  replay = NStepReplay(args.buffer_per_env, n_envs, obs_dim, cobs_dim, act_dim, device)
  noise_std = torch.empty(n_envs, 1, device=device).uniform_(args.std_min, args.std_max)

  log_dir = Path("logs/fasttd3") / args.task / (
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.run_name}")
  log_dir.mkdir(parents=True, exist_ok=True)
  (log_dir / "args.json").write_text(json.dumps(vars(args) | {"task": args.task}, indent=2))
  writer = SummaryWriter(str(log_dir))
  print(f"[fasttd3] logging to {log_dir}")

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
      "actor_target": actor_target.state_dict(), "critic_target": critic_target.state_dict(),
      "obs_norm": obs_norm.state_dict(), "cobs_norm": cobs_norm.state_dict(),
      "actor_opt": actor_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
      "env_step": step, "args": vars(args),
    }, log_dir / f"model_{tag}.pt")

  for step in range(1, args.total_env_steps + 1):
    # ── act ──
    with torch.no_grad():
      obs_norm.update(obs)
      cobs_norm.update(cobs)
      if step <= args.learning_starts:
        act = (torch.rand(n_envs, act_dim, device=device) * 2 - 1) * args.action_bound
      else:
        with amp:
          act = actor(obs_norm(obs)).float()
        act = (act + torch.randn_like(act) * noise_std * args.action_bound).clamp(-args.action_bound, args.action_bound)

    next_obs_dict, rew, term, trunc, extras = env.step(act)
    replay.add(obs, cobs, act, rew, term, trunc)
    obs, cobs = next_obs_dict["actor"], next_obs_dict["critic"]

    # ── episode bookkeeping ──
    ep_ret += rew
    ep_len += 1
    done = term | trunc
    if done.any():
      d = done.nonzero().squeeze(-1)
      finished_ret += ep_ret[d].tolist()
      finished_len += ep_len[d].tolist()
      ep_ret[d] = 0
      ep_len[d] = 0
      noise_std[d] = torch.empty(len(d), 1, device=device).uniform_(args.std_min, args.std_max)
    for k, v in extras.get("log", {}).items():
      env_logs[k].append(float(v))

    # ── learn ──
    if step > args.learning_starts and replay.size > args.n_step + 1:
      for _ in range(args.updates_per_step):
        update_count += 1
        b = replay.sample(args.batch_size, args.n_step, args.gamma)
        with torch.no_grad():
          o_n, co_n = obs_norm(b["obs"]), cobs_norm(b["cobs"])
          no_n, nco_n = obs_norm(b["next_obs"]), cobs_norm(b["next_cobs"])
          with amp:
            na = actor_target(no_n).float()
            na = (na + (torch.randn_like(na) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
                  * args.action_bound).clamp(-args.action_bound, args.action_bound)
            tq1, tq2 = critic_target(nco_n, na)
          p1 = critic_target.project(tq1, b["returns"], b["discount"])
          p2 = critic_target.project(tq2, b["returns"], b["discount"])
          # Clipped double Q: take the target distribution with the lower mean.
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
            a = actor(o_n)
            aq1, aq2 = critic(co_n, a)
          actor_loss = -torch.minimum(critic.value(aq1), critic.value(aq2)).mean()
          actor_opt.zero_grad(set_to_none=True)
          actor_loss.backward()
          actor_opt.step()
          losses["actor_loss"].append(actor_loss.item())

        with torch.no_grad():
          for net, tgt in ((critic, critic_target), (actor, actor_target)):
            for p, tp in zip(net.parameters(), tgt.parameters()):
              tp.lerp_(p, args.tau)

    # ── log ──
    if step % args.log_interval == 0:
      it = step // args.log_interval  # PPO-iteration equivalent (24 env steps)
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
      mr = sum(finished_ret) / len(finished_ret) if finished_ret else float("nan")
      ml = sum(finished_len) / len(finished_len) if finished_len else float("nan")
      kick = env_logs.get("Episode_Reward/ball_forward_velocity", [float("nan")])
      fell = env_logs.get("Episode_Termination/fell_over", [float("nan")])
      eta = elapsed / step * (args.total_env_steps - step)
      print(f"Learning iteration {it}/{args.total_env_steps // args.log_interval} | "
            f"Mean reward: {mr:.2f} | Mean episode length: {ml:.1f} | "
            f"ball_forward_velocity: {sum(kick) / len(kick):.4f} | fell_over: {sum(fell) / len(fell):.4f} | "
            f"critic_loss: {sum(losses['critic_loss'] or [0]) / max(len(losses['critic_loss']), 1):.4f} | "
            f"elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)
      finished_ret.clear(); finished_len.clear(); env_logs.clear(); losses.clear()

    if step % args.save_interval == 0:
      save(str(step // args.log_interval))

    if stop_requested["flag"]:
      print(f"[fasttd3] interrupted at env step {step}", flush=True)
      break
    if args.max_minutes is not None and time.time() - t_start > args.max_minutes * 60:
      print(f"[fasttd3] wall-time budget of {args.max_minutes} min reached at env step {step}", flush=True)
      break

  save("final")
  export_onnx(actor, obs_norm, obs_dim, log_dir / f"{args.run_name}.onnx", device)
  writer.close()
  env.close()
  print(f"[fasttd3] done in {(time.time() - t_start) / 60:.1f} min -> {log_dir}")


def export_onnx(actor: Actor, obs_norm: EmpiricalNormalizer, obs_dim: int, path: Path, device) -> None:
  """actor(normalizer(obs)) as [1, obs_dim] -> [1, act_dim], matching the PPO export contract."""

  class Policy(nn.Module):
    def __init__(self):
      super().__init__()
      self.norm, self.actor = obs_norm, actor

    def forward(self, obs):
      return self.actor(self.norm(obs))

  policy = Policy().eval().float()
  try:
    torch.onnx.export(policy, torch.zeros(1, obs_dim, device=device), str(path),
                      input_names=["obs"], output_names=["actions"], dynamo=False)
    print(f"[export] wrote {path}")
  except Exception as e:  # export is a convenience; the .pt checkpoint is the source of truth
    print(f"[export] ONNX export failed: {e}")


if __name__ == "__main__":
  main()
