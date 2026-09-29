# Training on a laptop GPU

Everything here was trained on a gaming laptop: Ryzen 9 5900HX, RTX 3060 Laptop (6 GB),
16 GB RAM, Ubuntu 22.04, NVIDIA driver 535. It works, but heat is the real limit, not
memory.

## What to expect

- `uv sync` needs about 8 GB of disk (the `.venv` alone is 7.8 GB: torch + CUDA wheels, warp).
- GPU memory is not a problem at 1024 envs: ~1.3 GB for PPO, ~2-2.7 GB for FastTD3/FastSAC
  (the replay buffer lives on the GPU).
- Temperature is. At 1024 envs the GPU goes from ~55 °C to 83 °C in about a minute and
  keeps climbing if nothing stops it.
- Speeds I measured at 1024 envs, GPU cool: PPO ~3.8 s/iteration on the walking task and
  ~4.6 s on the kick task (the ball adds physics); FastSAC/FastTD3 ~6 s. With the thermal
  pauses included the effective time roughly doubles.
- Driver 535 is older than what torch 2.9's CUDA 12.8 wheels officially want, but it worked
  fine for everything here.

## The watchdog

`scripts/run_guarded.sh` runs any `uv run` command in its own process group and checks the
GPU temperature every 5 seconds:

- 83 °C or more: pause the job (`SIGSTOP`)
- back down to 72 °C: resume it (`SIGCONT`)
- 88 °C: stop it for good (`SIGINT` first, so the FastTD3/FastSAC scripts save a checkpoint
  and export ONNX before exiting)

```bash
scripts/run_guarded.sh train Mjlab-BallKick-Flat-MicroDuck --env.scene.num-envs 1024 \
    --agent.max-iterations 1500 --agent.save-interval 100 --agent.logger tensorboard

OUT_DIR=runs/sac scripts/run_guarded.sh scripts/train_fastsac.py --max-minutes 235
```

Logs land in `$OUT_DIR` (default `guarded_run/`): `train.log`, `watchdog.log` (every pause
and resume) and `gpu.csv` (temperature, utilization, memory, power every 5 s). To stop a run
early and still get a checkpoint: `touch $OUT_DIR/STOP_REQUEST`.

Thresholds can be changed with `PAUSE_AT`, `RESUME_AT` and `STOP_AT`.

In practice the job ends up running about half the time. Over the 4 hour kick runs the
watchdog paused the job 250-290 times, the peak was 85-86 °C and the hard stop never fired.

## Things that help

- Fewer envs (1024 or even 512 instead of the upstream 4096).
- A power cap, if your laptop's GPU accepts one: `sudo nvidia-smi -pl 45`
  (resets on reboot). Mine reports a settable range of 1-95 W.
- Airflow: raise the back of the laptop or use a cooling pad, keep it plugged in.
- Local TensorBoard logging (`--agent.logger tensorboard`) instead of W&B if you don't want
  to log in anywhere.
