"""Run the shipped nearest-player baseline on an OTHER match (open DFL data) with the repo's scorer.

Why: the public ground truth is 17 possessions and unrepresentative, so it cannot compare
methods. One DFL match has a few hundred possessions and per-frame labels, and we can hide the
ball in it with the gap pattern of the provided match so it resembles our situation.

    uv run python analysis/dfl_baseline.py

Two caveats, both reported in the output rather than hidden:
  - DFL's possession labels are a different, more fine-grained definition than the provided GT.
  - The hidden-ball pattern is copied from where the provided match lost its ball, so it is
    independent of the action in the DFL match, whereas real losses cluster around crowded play.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from analysis import dfl, tracking  # noqa: E402
from baseline import smooth, HOME, AWAY, NOBODY  # noqa: E402
import possession_metrics as pm  # noqa: E402

GAP = 100  # frames of X between the halves, so the scorer sees two separate runs
CODE = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}

# The seven open matches. Each match's hidden-ball gap pattern is the provided match's pattern rolled
# by ROLL frames times the match's index here, so a match scores the same whichever subset is run.
ALL_MATCHES = ("DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY",
               "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9")
ROLL = 20000


def downloaded(match_id):
    return any(f.startswith("DFL_04_03_positions_raw_observed") and match_id in f
               for f in os.listdir(dfl.DFL_DIR))


def nearest_team(half, ball_visible):
    """Team code (HOME/AWAY/NOBODY) of the player nearest the ball on each frame."""
    d = np.linalg.norm(half.xy - half.ball[:, None, :2], axis=2)        # (T, P), NaN if untracked
    d[:, np.all(np.isnan(d), axis=0)] = np.inf
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.argmin(1)
    team = np.where(half.team[nearest] == "H", HOME, AWAY).astype(np.int8)
    team[~ball_visible | ~np.isfinite(d.min(1))] = NOBODY
    return team


def provided_visibility():
    """Per-half boolean ball-visible masks from the provided match, as the gap pattern to copy."""
    t = tracking.load()
    vis = ~np.isnan(t.ball[:, 0])
    (_, a1, b1), (_, a2, b2) = tracking.half_video_blocks("BAR-ATH")
    return [vis[a1:b1 + 1], vis[a2:b2 + 1]]


def assemble(parts):
    gap = [np.full(GAP, "X", "<U1")]
    return np.concatenate([parts[0]] + gap + [parts[1]])


def encode(labels):
    return np.array([CODE[c] for c in labels], dtype=np.int8)


def run(matches=None):
    """Score each match separately. Default: every open match that is downloaded."""
    if matches is None:
        matches = [m for m in ALL_MATCHES if downloaded(m)]
    data = []
    for mid in matches:
        halves = list(dfl.load(mid).values())
        data.append((mid, halves, encode(assemble([h.labels() for h in halves])), ALL_MATCHES.index(mid)))
    patterns = provided_visibility()
    windows = (0, 25, 50, 125, 250)

    for scenario in ("ball always visible (as DFL delivers it)", "ball hidden with our gap pattern"):
        masked = scenario.startswith("ball hidden")
        print(f"\n=== {scenario}")
        scores = {w: [] for w in windows}
        spells = {w: [] for w in windows}
        for mid, halves, gt, k in data:
            vis = [np.roll(p, ROLL * k)[:h.n] if masked else np.ones(h.n, bool)
                   for h, p in zip(halves, patterns)]
            codes = [nearest_team(h, v) for h, v in zip(halves, vis)]
            for w in windows:
                pred = encode(assemble([smooth(c, w) for c in codes]))
                r = pm.evaluate(gt, pred)
                scores[w].append(r["spell_f1_mean"]); spells[w].append((r["n_pred_spells"], r["n_gt_spells"]))
        names = "  ".join(f"{m[-6:]:>7}" for m, *_ in data)
        print(f"{'window':>8}  spellF1 per match: {names}   {'mean':>6}   pred/true spells per match")
        for w in windows:
            cells = "  ".join(f"{s:7.3f}" for s in scores[w])
            print(f"{'+-' + str(w):>8}  {'':>18}{cells}   {np.mean(scores[w]):6.3f}   "
                  + "  ".join(f"{p}/{t}" for p, t in spells[w]))


if __name__ == "__main__":
    run()
