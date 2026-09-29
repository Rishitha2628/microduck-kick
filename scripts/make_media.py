"""Build the side-by-side comparison video + README GIF from two eval_onnx.py recordings.

  uv run scripts/make_media.py --left media/PPO.mp4 --left-label "PPO: kicks, then falls" \
      --right media/FastSAC.mp4 --right-label "FastSAC: kicks, stays up"

Each 1920x1080 input is cropped to a 960x1080 window around the duck, so the output is a
1920x1080 side-by-side at the input frame rate, plus a smaller looping GIF of the first episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--left", required=True)
  p.add_argument("--right", required=True)
  p.add_argument("--left-label", required=True)
  p.add_argument("--right-label", required=True)
  p.add_argument("--crop-x", type=int, default=560, help="left edge of the 960 px crop in each input")
  p.add_argument("--out", default="media/side_by_side.mp4")
  p.add_argument("--gif", default="media/side_by_side.gif")
  p.add_argument("--gif-width", type=int, default=640)
  p.add_argument("--gif-fps", type=int, default=15)
  p.add_argument("--gif-seconds", type=float, default=4.5)
  args = p.parse_args()

  font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
  font = ImageFont.truetype(str(font_path), 44)

  left, right = imageio.get_reader(args.left), imageio.get_reader(args.right)
  fps = left.get_meta_data()["fps"]
  writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=9,
                              pixelformat="yuv420p", macro_block_size=8)  # keeps 1080, not 1088
  gif_every = max(1, round(fps / args.gif_fps))
  gif_frames, n = [], 0
  for fl, fr in zip(left, right):
    x = args.crop_x
    img = Image.fromarray(np.concatenate([fl[:, x:x + 960], fr[:, x:x + 960]], axis=1))
    draw = ImageDraw.Draw(img)
    draw.rectangle([958, 0, 961, img.height], fill=(255, 255, 255))
    for x0, text in ((0, args.left_label), (960, args.right_label)):
      w = draw.textlength(text, font=font)
      draw.text((x0 + (960 - w) / 2, 40), text, font=font, fill=(255, 255, 255))
    frame = np.asarray(img)
    writer.append_data(frame)
    if n % gif_every == 0 and n < args.gif_seconds * fps:
      h = round(img.height * args.gif_width / img.width)
      gif_frames.append(np.asarray(img.resize((args.gif_width, h), Image.LANCZOS)))
    n += 1
  writer.close()
  imageio.mimsave(args.gif, gif_frames, duration=1000 / args.gif_fps, loop=0)
  print(f"{args.out}: {n} frames at {fps} fps | {args.gif}: {len(gif_frames)} frames, "
        f"{Path(args.gif).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
  main()
