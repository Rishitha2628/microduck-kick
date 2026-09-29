"""Compare training runs (rsl_rl PPO / FastTD3) from their TensorBoard logs.

Both trainers log the same tags (Train/mean_reward, Episode_Reward/*, Episode_Termination/*)
against a PPO-iteration step (24 env steps), so curves line up by experience; wall-clock
comes from the event timestamps.

  uv run scripts/compare_runs.py --run PPO=logs/rsl_rl/ball_kick_right/<run> \
      --run FastTD3=logs/fasttd3/Mjlab-BallKick-Flat-MicroDuck/<run> --out compare.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

PANELS = [
  ("Train/mean_reward", "Episode return", "iter"),
  ("Train/mean_reward", "Episode return vs wall-clock", "time"),
  ("Episode_Reward/ball_forward_velocity", "Kick reward (ball forward speed)", "iter"),
  ("Episode_Termination/fell_over", "Falls per log window", "iter"),
]


def load(run_dir: Path) -> dict[str, tuple[list[float], list[float], list[float]]]:
  ev = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  ev.Reload()
  out = {}
  for tag in ev.Tags()["scalars"]:
    s = ev.Scalars(tag)
    t0 = s[0].wall_time
    out[tag] = ([x.step for x in s], [(x.wall_time - t0) / 60 for x in s], [x.value for x in s])
  return out


def smooth(v: list[float], k: int = 15) -> list[float]:
  out, acc = [], []
  for x in v:
    acc.append(x)
    acc = acc[-k:]
    out.append(sum(acc) / len(acc))
  return out


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run", action="append", required=True, help="LABEL=path/to/run_dir")
  p.add_argument("--out", default="compare.png")
  args = p.parse_args()
  runs = [(r.split("=", 1)[0], load(Path(r.split("=", 1)[1]))) for r in args.run]

  plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                       "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.edgecolor": GRID})
  fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=SURFACE)
  for ax, (tag, title, xkind) in zip(axes.flat, PANELS):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel("PPO-equivalent iteration (24 env steps)" if xkind == "iter" else "wall-clock (min)")
    for (label, data), color in zip(runs, SERIES_COLORS):
      if tag not in data:
        continue
      steps, mins, vals = data[tag]
      x = steps if xkind == "iter" else mins
      y = smooth(vals)
      ax.plot(x, y, color=color, linewidth=2, label=label)
      ax.annotate(f"{label} {y[-1]:.1f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                  va="center", fontsize=9, color=TEXT)
  handles, labels = axes.flat[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper right", frameon=False)
  fig.suptitle("Microduck BallKick (right foot), 1024 envs, RTX 3060 laptop: PPO & FastSAC ~4 h, FastTD3 stopped at ~1 h", x=0.01, ha="left",
               fontsize=13, color=TEXT)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.savefig(args.out, dpi=130, facecolor=SURFACE)
  print(f"saved {args.out}")

  print(f"\n{'run':<10}{'iters':>7}{'minutes':>9}{'return':>9}{'kick':>8}{'falls':>8}")
  for label, d in runs:
    last = lambda tag: sum(d[tag][2][-15:]) / len(d[tag][2][-15:]) if tag in d else float("nan")
    steps, mins, _ = d["Train/mean_reward"]
    print(f"{label:<10}{steps[-1]:>7}{mins[-1]:>9.0f}{last('Train/mean_reward'):>9.2f}"
          f"{last('Episode_Reward/ball_forward_velocity'):>8.2f}{last('Episode_Termination/fell_over'):>8.2f}")


if __name__ == "__main__":
  main()
