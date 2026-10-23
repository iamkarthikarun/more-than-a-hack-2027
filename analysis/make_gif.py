"""Animate a stretch of the tactical tracking as a flat 2D GIF.

    uv run python analysis/make_gif.py                       # default 24 s clip
    uv run python analysis/make_gif.py --start 13700 --seconds 15 --no-labels

The banner shows the public ground-truth label (H/A/D) where one exists. That is for looking at
the data only -- RULES.md forbids tuning anything on it. Output goes to analysis/figures/, which
is git-ignored: frame-level tracking and clipped video must not be published.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.tracking import load, FPS  # noqa: E402
from starter.utils import draw_pitch, HOME_COLOUR, AWAY_COLOUR  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

GT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BANNER = {"H": ("Barcelona in possession", HOME_COLOUR), "A": ("Athletic in possession", AWAY_COLOUR),
          "D": ("Dead ball (play stopped)", "#555555")}
TRAIL = FPS  # one second of ball history


def make(start, seconds, step, out, labels):
    t = load()
    frames = np.arange(start, min(start + seconds * FPS, t.n), step)
    gt = pd.read_csv(GT).label.values if labels and os.path.exists(GT) else None
    home, away = t.team == "H", t.team == "A"

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    draw_pitch(ax)
    sc_h = ax.scatter([], [], s=150, c=HOME_COLOUR, edgecolor="w", zorder=3)
    sc_a = ax.scatter([], [], s=150, c=AWAY_COLOUR, edgecolor="w", zorder=3)
    sc_b = ax.scatter([], [], s=110, c="yellow", edgecolor="k", zorder=5)
    (trail,) = ax.plot([], [], color="yellow", lw=2, alpha=.7, zorder=4)
    title = ax.set_title("00.0s   Barcelona in possession   [ball not tracked]", fontsize=13,
                         fontweight="bold")
    ax.set_xlabel("metres  |  red = Barcelona, blue = Athletic, yellow = ball  |  "
                  "1st half: Barcelona attack left to right")
    fig.tight_layout()

    def pos(f, mask):
        xy = t.metres(t.xy[f][mask])
        return xy[~np.isnan(xy[:, 0])]

    def update(i):
        f = frames[i]
        sc_h.set_offsets(pos(f, home)); sc_a.set_offsets(pos(f, away))
        ball = t.metres(t.ball[f, :2])
        has_ball = not np.isnan(ball[0])
        sc_b.set_offsets(ball[None] if has_ball else np.empty((0, 2)))
        hist = t.metres(t.ball[max(0, f - TRAIL):f + 1, :2])
        hist = hist[~np.isnan(hist[:, 0])]
        trail.set_data(hist[:, 0], hist[:, 1])
        clock = f"{(f - start) / FPS:4.1f}s"
        text, colour = BANNER.get(gt[f], ("", "k")) if gt is not None else ("", "k")
        note = "" if has_ball else "   [ball not tracked]"
        title.set_text(f"{clock}   {text}{note}"); title.set_color(colour)
        return sc_h, sc_a, sc_b, trail, title

    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(out, writer=PillowWriter(fps=FPS / step), dpi=60)
    print(f"wrote {out}  ({len(frames)} frames, {os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=15380, help="first video frame")
    p.add_argument("--seconds", type=int, default=24)
    p.add_argument("--step", type=int, default=2, help="use every Nth frame (2 = 12.5 fps)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "figures", "clip.gif"))
    p.add_argument("--no-labels", action="store_true", help="hide the ground-truth banner")
    a = p.parse_args()
    make(a.start, a.seconds, a.step, a.out, not a.no_labels)
