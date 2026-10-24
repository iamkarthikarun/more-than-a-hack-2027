"""Does Official flip possession on a deliberate touch (even a header or clearance), or only on control?

    uv run python analysis/touch_vs_control.py

Analysis only, on the 5-minute public ground truth window (video frames 11185-18684). Nothing printed here
changes a rule. Two views of the same question:

  1. Kaggle event -> Official. For every Kaggle event by a team in the window, when Official was showing the
     other team (or a dead ball) just before it, how often does Official show the event's team within 2 s,
     and how long after the event does it start doing so? Grouped by event type.
  2. Official team change -> Kaggle. For every H<->A change in Official, which Kaggle events lie within +-1.5 s.
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import events as ev_mod  # noqa: E402
from analysis import pseudo_gt  # noqa: E402
from analysis.video_time import frame_to_timecode  # noqa: E402

FPS = 25
A, B = 11185, 18684
GT_CSV = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BEFORE_S, AFTER_S = 0.5, 2.0

CONTESTED = {"Header", "Flick-on", "Aerial", "Clearance", "Interception", "Blocked Pass", "Tackle", "Challenge",
             "Dispossessed", "Ball touch", "Unsuccessful touch"}
RESTARTS = {"Throw in", "Goal kick", "Drop of ball", "Corner", "Free kick", "Start delay", "Out", "Offside Pass"}


def event_frames(ev):
    ev = ev[ev.period.isin([1, 2])]
    off = pseudo_gt.offsets(ev)
    return ev.assign(frame=[int(round((w + off[p]) * FPS)) for w, p in zip(ev.wall, ev.period)])


def kind(name):
    if name in CONTESTED:
        return "contested"
    if name in RESTARTS:
        return "restart/stoppage"
    return "control-type"


def main():
    gt = pd.read_csv(GT_CSV).label.values
    ev = event_frames(ev_mod.load_events())
    ev = ev[(ev.frame >= A + 2 * FPS) & (ev.frame <= B - 2 * FPS) & ev.team.isin(["H", "A"])].sort_values("frame")
    print(f"{len(ev)} Kaggle events by H or A inside the window (2 s trimmed at each end)\n")

    print("1. Kaggle event -> Official. Only events where Official showed the other team just before it "
          f"(label at -{BEFORE_S} s was the other team)")
    print(f"   'flip' = Official shows the event's team at some frame in [-{BEFORE_S} s, +{AFTER_S} s]; lag = first such frame - event\n")
    rows = []
    for _, r in ev.iterrows():
        f, team = r.frame, r.team
        other = "A" if team == "H" else "H"
        if gt[f - int(BEFORE_S * FPS)] != other:
            continue
        win = gt[f - int(BEFORE_S * FPS): f + int(AFTER_S * FPS) + 1]
        hit = np.flatnonzero(win == team)
        lag = (hit[0] - int(BEFORE_S * FPS)) / FPS if len(hit) else None
        rows.append((r["name"], kind(r["name"]), lag is not None, lag))
    df = pd.DataFrame(rows, columns=["event", "kind", "flip", "lag"])
    print(f"   {'event':<20}{'n':>4}{'flips':>7}{'share':>7}{'median lag s':>14}")
    for name, g in sorted(df.groupby("event"), key=lambda kv: -len(kv[1])):
        print(f"   {name:<20}{len(g):>4}{g.flip.sum():>7}{g.flip.mean():>7.0%}{g.lag.median() if g.flip.any() else float('nan'):>14.2f}")
    print()
    for k, g in df.groupby("kind"):
        print(f"   {k:<18} n={len(g):>3}  flips {g.flip.sum():>3} ({g.flip.mean():.0%})  median lag {g.lag.median():.2f} s")

    print("\n   Discriminating cut: contested events with NO control-type event by the same team in the next 3 s "
          "(a lone touch; a flip here cannot be explained by a later control)")
    lone = []
    for _, r in ev[ev["name"].isin(CONTESTED)].iterrows():
        f, team = r.frame, r.team
        other = "A" if team == "H" else "H"
        if gt[f - int(BEFORE_S * FPS)] != other:
            continue
        later = ev[(ev.team == team) & (ev.frame > f) & (ev.frame <= f + 3 * FPS) & (ev["name"].map(kind) == "control-type")]
        win = gt[f - int(BEFORE_S * FPS): f + int(AFTER_S * FPS) + 1]
        lone.append((frame_to_timecode(f)[3:], r["name"], team, len(later) == 0, "yes" if (win == team).any() else "no"))
    print(f"   {'time':>9}  {'event':<20}{'team':>5}  {'lone':>5}  Official flips to team")
    for t, n, tm, ln, fl in lone:
        print(f"   {t:>9}  {n:<20}{tm:>5}  {str(ln):>5}  {fl}")

    print("\n2. Official H<->A changes in the window and the Kaggle events within +-1.5 s (event name, team, offset s)")
    ch = np.flatnonzero((gt[A + 1:B + 1] != gt[A:B]) & np.isin(gt[A + 1:B + 1], ["H", "A"]) & np.isin(gt[A:B], ["H", "A"])) + A + 1
    nearest = Counter()
    for f in ch:
        near = ev[(ev.frame - f).abs() <= 1.5 * FPS]
        desc = ", ".join(f"{r['name']}({r.team}){(r.frame - f) / FPS:+.1f}" for _, r in near.iterrows())
        print(f"   {frame_to_timecode(f)[3:]:>9} {gt[f - 1]}>{gt[f]}   {desc or '(no Kaggle event within 1.5 s)'}")
        first = near[near.team == gt[f]]
        nearest.update([kind(first.iloc[(first.frame - f).abs().argsort()].iloc[0]['name'])] if len(first) else ["none for new team"])
    print(f"\n   nearest same-team Kaggle event by kind: {dict(nearest)}")


if __name__ == "__main__":
    main()
