"""One-page overview of the tactical tracking: ball coverage, tracked-object counts, ball gaps, a frame.

    uv run python analysis/overview_plot.py          # writes analysis/figures/overview.png
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.tracking import load, graded_mask, segments, FPS  # noqa: E402
from games import half_video_blocks  # noqa: E402
from starter.utils import draw_pitch  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "overview.png")
WIN = 60 * FPS


def windowed(x, win=WIN):
    n = len(x) // win
    return x[: n * win].reshape(n, win).mean(1), (np.arange(n) * win + win / 2) / FPS / 60


def runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(int))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def main(sample_frame=40000):
    t = load()
    G = graded_mask()
    H = t.team == "H"
    vis = ~np.isnan(t.ball[:, 0])
    seen = ~np.isnan(t.xy[..., 0])

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    ax1, ax2, ax3 = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :])

    v, m = windowed(vis.astype(float))
    ax1.plot(m, v * 100, color="k", lw=1, label="ball visible %")
    for lab, ls in (("H", "#d62728"), ("A", "#1f77b4")):
        c, m2 = windowed(seen[:, t.team == lab].sum(1).astype(float))
        ax1.plot(m2, c / 11 * 100, color=ls, lw=1, alpha=.8, label=f"{lab} objects tracked (% of 11)")
    for name, a, b in segments():
        if "half" in name and name != "halftime":
            ax1.axvspan(a / FPS / 60, b / FPS / 60, color="green", alpha=.06)
    ax1.axvspan(11185 / FPS / 60, 18684 / FPS / 60, color="orange", alpha=.35, label="public GT window")
    ax1.set_xlabel("minutes of video"); ax1.set_ylabel("%"); ax1.set_ylim(0, 105)
    ax1.set_title("Coverage per minute (shaded green = graded halves)"); ax1.legend(loc="lower right", fontsize=8)

    gaps = np.concatenate([runs(~vis[a:b + 1]) for _, a, b in half_video_blocks("BAR-ATH")]) / FPS
    ax2.hist(gaps, bins=np.logspace(-1, 2.5, 30), color="gray")
    ax2.set_xscale("log"); ax2.set_xlabel("ball-missing gap (s)"); ax2.set_ylabel("count")
    ax2.set_title(f"Ball gaps in play: {len(gaps)}, median {np.median(gaps):.1f}s")

    draw_pitch(ax3)
    for lab, col in (("H", "#d62728"), ("A", "#1f77b4")):
        idx = np.where((t.team == lab) & seen[sample_frame])[0]
        xy = t.metres(t.xy[sample_frame, idx])
        ax3.scatter(xy[:, 0], xy[:, 1], s=130, c=col, edgecolor="w", zorder=3)
        for k, (x, y) in zip(idx, xy):
            ax3.text(x, y, str(k + 1), ha="center", va="center", fontsize=7, color="w", zorder=4)
    if vis[sample_frame]:
        bx, by = t.metres(t.ball[sample_frame, :2])
        ax3.scatter([bx], [by], s=80, c="yellow", edgecolor="k", zorder=5, label="ball")
    ax3.set_title(f"Video frame {sample_frame}: red = Barcelona (H), blue = Athletic (A), numbers = track slots")
    ax3.legend(loc="upper right")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=110)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
