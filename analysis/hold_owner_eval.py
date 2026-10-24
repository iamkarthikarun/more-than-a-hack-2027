"""Hold-the-last-owner against the nearest-player baseline, on degraded DFL, tuned leave-one-match-out.

    uv run python analysis/hold_owner_eval.py                       # every downloaded match
    uv run python analysis/hold_owner_eval.py --matches DFL-MAT-J03WOH DFL-MAT-J03WN1

Each match is degraded to look like the provided tracking (analysis/degrade.py). Settings are chosen on
the other matches only and scored on the held-out one, so no number is tuned on the match it is
reported on. No provided-match labels are used anywhere.

Two label definitions are scored, because the organisers' ground truth says a touch without control
is not a possession, while DFL's labels are finer:
  raw      DFL's possession flag as delivered
  merged   the same, with every H/A spell shorter than 3 s merged into its neighbour, a proxy for
           "controlled possession" (CHALLENGE.md: keeps the ball a few seconds). A sensitivity check,
           not a claim about what the real ground truth is.
"""

import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import dfl, degrade, hold_owner as ho  # noqa: E402
from analysis.degrade_eval import CALIBRATION_MATCH  # noqa: E402
from analysis.dfl_baseline import ALL_MATCHES, assemble, downloaded, encode, nearest_team  # noqa: E402
from baseline import smooth as vote  # noqa: E402
import possession_metrics as pm  # noqa: E402

SEEDS = (0, 1)
CONDITIONS = ("clean ball", "constant hazard", "learned hazard")
TUNE_ON = ("constant hazard", "learned hazard")
GT_VARIANTS = ("raw", "merged")
MERGE_FRAMES = 75
GRID = [ho.Params(r_control=r, d_still=d, dead_hold_max=h, min_spell=s, min_dead=m)
        for r, d, h, (s, m) in itertools.product((0.5, 0.75, 1.0, 1.5, 2.0, 3.0), (1.0, 2.0), (0, 250),
                                                  ((1, 1), (10, 10), (25, 25), (50, 25), (75, 50), (125, 75)))]
DEFAULT = ho.Params()
KEYS = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive", "n_pred_spells", "n_gt_spells")


def variants(model, halves, cond):
    if cond == "clean ball":
        return [[degrade.apply(model, h, ball=False, players=False, smooth_positions=False) for h in halves]]
    hz = "constant" if cond == "constant hazard" else "learned"
    return [[degrade.apply(model, h, seed=s, ball_hazard=hz) for h in halves] for s in SEEDS]


def build(model, matches):
    data = {}
    for m in matches:
        halves = list(dfl.load(m).values())
        labs = [h.labels() for h in halves]
        gt = {"raw": encode(assemble(labs)),
              "merged": encode(assemble([ho.merge_short(l, MERGE_FRAMES, 0) for l in labs]))}
        entry = {"gt": gt, "cond": {}}
        for cond in CONDITIONS:
            entry["cond"][cond] = [
                {"prep": [ho.prepare(h) for h in v],
                 "base": [vote(nearest_team(h, ~np.isnan(h.ball[:, 0])), 50) for h in v]}
                for v in variants(model, halves, cond)]
        data[m] = entry
    return data


def _mean(acc):
    return {g: np.mean(a, 0) for g, a in acc.items()}


def score(entry, cond, params):
    """{gt variant: mean over seeds of KEYS} for the hold-the-last-owner labels."""
    acc = {g: [] for g in GT_VARIANTS}
    for v in entry["cond"][cond]:
        pred = encode(assemble([ho.labels(p, params) for p in v["prep"]]))
        for g in GT_VARIANTS:
            r = pm.evaluate(entry["gt"][g], pred)
            acc[g].append([r[k] for k in KEYS])
    return _mean(acc)


def baseline(entry, cond):
    acc = {g: [] for g in GT_VARIANTS}
    for v in entry["cond"][cond]:
        pred = encode(assemble(v["base"]))
        for g in GT_VARIANTS:
            acc[g].append(pm.evaluate(entry["gt"][g], pred)["spell_f1_mean"])
    return {g: float(np.mean(a)) for g, a in acc.items()}


def main(matches=None):
    matches = matches or [m for m in ALL_MATCHES if downloaded(m)]
    if len(matches) < 2:
        sys.exit("need at least two matches for leave-one-match-out")
    print("fitting the degradation on the provided match ...")
    model = degrade.fit(list(dfl.load(CALIBRATION_MATCH).values())[0], verbose=False)
    data = build(model, matches)
    print(f"matches: {', '.join(m[-6:] for m in matches)}; {len(GRID)} settings in the grid\n")

    tune = {m: {c: [score(data[m], c, g) for g in GRID] for c in TUNE_ON} for m in matches}
    summary = {(g, c): {"base": [], "default": [], "tuned": []} for g in GT_VARIANTS for c in CONDITIONS}
    chosen = {}
    for gv in GT_VARIANTS:
        print(f"=== ground truth: {gv}")
        print(f"{'held-out':9}{'condition':18}{'nearest+vote50':>15}{'hold, default':>15}{'hold, tuned':>13}"
              f"{'macroF1':>9}{'alive/dead':>11}{'team acc':>9}{'spells pred/true':>18}")
        for test in matches:
            train = [m for m in matches if m != test]
            totals = [np.mean([tune[m][c][i][gv][0] for m in train for c in TUNE_ON]) for i in range(len(GRID))]
            best = GRID[int(np.argmax(totals))]
            chosen[(gv, test)] = best
            for c in CONDITIONS:
                b = baseline(data[test], c)[gv]
                d = score(data[test], c, DEFAULT)[gv][0]
                t = score(data[test], c, best)[gv]
                s = summary[(gv, c)]
                s["base"].append(b); s["default"].append(d); s["tuned"].append(t[0])
                print(f"{test[-6:]:9}{c:18}{b:15.3f}{d:15.3f}{t[0]:13.3f}{t[1]:9.3f}{t[2]:11.3f}{t[3]:9.3f}"
                      f"{t[4]:12.0f} /{t[5]:4.0f}")
        print()

    print("mean over held-out matches")
    for gv in GT_VARIANTS:
        for c in CONDITIONS:
            s = summary[(gv, c)]
            print(f"  gt {gv:7} {c:18} nearest+vote50 {np.mean(s['base']):.3f} | hold default "
                  f"{np.mean(s['default']):.3f} | hold tuned {np.mean(s['tuned']):.3f}")
    print("\nsettings chosen on the other matches (r_control m, d_still m, dead_hold_max, min_spell, min_dead):")
    for (gv, m), p in chosen.items():
        print(f"  gt {gv:7} held out {m[-6:]}: {p.r_control}, {p.d_still}, {p.dead_hold_max}, {p.min_spell}, {p.min_dead}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", choices=ALL_MATCHES)
    main(ap.parse_args().matches)
