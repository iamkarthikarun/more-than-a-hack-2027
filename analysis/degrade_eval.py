"""Does degraded DFL look like the provided match, and how much harder is it for the baseline?

    uv run python analysis/degrade_eval.py

Fits the degradation on the provided match's tracking (no labels), applies it to every downloaded DFL
match, and compares: tracking statistics against the provided match, and the nearest-player baseline's
spellF1 under four conditions. Also reports how often a real possession change falls in a ball gap.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import dfl, degrade  # noqa: E402
from analysis.dfl_baseline import (ALL_MATCHES, ROLL, assemble, downloaded, encode, nearest_team,  # noqa: E402
                                   provided_visibility)
from baseline import smooth as vote  # noqa: E402
import possession_metrics as pm  # noqa: E402

WINDOWS = (0, 25, 50, 125, 250)
SEEDS = (0, 1, 2)
CALIBRATION_MATCH = "DFL-MAT-J03WOH"   # fixed, so every machine fits the same model: full 11 v 11 in the 1st half


def runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(int))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def run_stats(vis_halves):
    miss = np.concatenate([runs(~v) for v in vis_halves]) / degrade.FPS
    seen = np.concatenate([runs(v) for v in vis_halves]) / degrade.FPS
    share = np.mean(np.concatenate(vis_halves))
    return share, np.median(seen), seen.mean(), np.median(miss), miss.mean(), len(miss)


def deficit_table(items):
    """items: (tracked xy, on-pitch xy or None for 11, team). Share of frames a team has 0/1/2/>=3
    of its on-pitch players untracked, and (provided match only) the share with a phantom 12th+."""
    out = {}
    for lab in "HA":
        pmfs, ph = [], []
        for xy, clean, team in items:
            tr = (~np.isnan(xy[..., 0]))[:, team == lab].sum(1)
            cl = (~np.isnan(clean[..., 0]))[:, team == lab].sum(1) if clean is not None else 11
            pmfs.append(degrade.deficit_pmf(tr, cl) * len(tr))
            ph.append(np.sum(tr > cl))
        n = sum(len(xy) for xy, _, _ in items)
        out[lab] = (np.sum(pmfs, 0) / n, np.sum(ph) / n)
    return out


def transitions_hidden(labels, hidden):
    """Share of real H<->A possession changes whose frame has the ball hidden."""
    idx = [i for i in range(1, len(labels)) if labels[i] in "HA" and labels[i - 1] in "HA" and labels[i] != labels[i - 1]]
    return (np.mean(hidden[idx]) if idx else np.nan), len(idx)


def main(matches=None):
    matches = matches or [m for m in ALL_MATCHES if downloaded(m)]
    data = {m: list(dfl.load(m).values()) for m in matches}
    print(f"DFL matches used: {', '.join(m[-6:] for m in matches)}\n")

    print(f"Fitting on the provided match (clean half for calibration: {CALIBRATION_MATCH[-6:]} 1st half):")
    calibration = (data[CALIBRATION_MATCH] if CALIBRATION_MATCH in data else list(dfl.load(CALIBRATION_MATCH).values()))[0]
    model = degrade.fit(calibration)

    t, ours_xy, ours_ball, blocks = degrade._ours()
    vis = ~np.isnan(ours_ball[:, 0])
    seen = ~np.isnan(ours_xy[..., 0])
    ours_vis = [vis[a:b + 1] for _, a, b in blocks]

    # ---- realism
    deg = {m: [degrade.apply(model, h, seed=0) for h in data[m]] for m in matches}
    print("\nRealism: provided match vs DFL (clean) vs DFL (degraded), in play")
    deg_vis = [~np.isnan(h.ball[:, 0]) for m in matches for h in deg[m]]
    clean_vis = [np.ones(h.n, bool) for m in matches for h in data[m]]
    print(f"{'':22}{'ball seen':>10}{'seen run med/mean s':>22}{'gap med/mean s':>18}{'gaps':>6}")
    for name, v in (("provided", ours_vis), ("DFL clean", clean_vis), ("DFL degraded", deg_vis)):
        if name == "DFL clean":
            print(f"{name:22}{'100.0%':>10}")
            continue
        s, sm, sa, gm, ga, n = run_stats(v)
        print(f"{name:22}{s:10.1%}{sm:11.1f} /{sa:6.1f}{gm:11.1f} /{ga:5.1f}{n:6d}")

    print("\n  untracked players of a team (of its 11)      none     1       2      3+    phantom 12th+")
    rows = (("provided", [(ours_xy[a:b + 1], None, t.team) for _, a, b in blocks]),
            ("DFL degraded", [(d.xy, c.xy, c.team) for m in matches for d, c in zip(deg[m], data[m])]))
    for name, items in rows:
        tab = deficit_table(items)
        for lab in "HA":
            pmf, ph = tab[lab]
            print(f"   {name:14} {lab}                   " + "  ".join(f"{v:6.1%}" for v in pmf) + f"   {ph:6.1%}")
    print("   (DFL clean has none missing by construction; the 12th+ phantom tracks are not simulated)")

    print("\n  position jitter, median / p90 / p99 of |second difference| in mm")
    pj = np.mean([degrade.second_diff_stats(ours_xy[a:b + 1]) for _, a, b in blocks], 0)
    cj = np.mean([degrade.second_diff_stats(h.xy + degrade.CENTRE) for m in matches for h in data[m]], 0)
    dj = np.mean([degrade.second_diff_stats(h.xy) for m in matches for h in deg[m]], 0)
    for name, j in (("provided", pj), ("DFL clean", cj), ("DFL degraded", dj)):
        print(f"   {name:14} {j[0]:6.1f} / {j[1]:6.1f} / {j[2]:7.1f}")

    # ---- baseline difficulty
    patterns = provided_visibility()
    CONDS = ("clean", "copied", "ball const", "ball", "ball+players")
    results = {c: {w: [] for w in WINDOWS} for c in CONDS}
    hard = {c: [] for c in CONDS[1:]}
    base_hidden = {c: [] for c in CONDS[1:]}
    for m in matches:
        halves = data[m]
        gt = encode(assemble([h.labels() for h in halves]))
        k = ALL_MATCHES.index(m)

        def score(codes_per_half, cond):
            for w in WINDOWS:
                pred = encode(assemble([vote(c, w) for c in codes_per_half]))
                results[cond][w].append(pm.evaluate(gt, pred)["spell_f1_mean"])

        score([nearest_team(h, np.ones(h.n, bool)) for h in halves], "clean")
        copied = [np.roll(p, ROLL * k)[:h.n] for h, p in zip(halves, patterns)]
        score([nearest_team(h, v) for h, v in zip(halves, copied)], "copied")
        lab = np.concatenate([h.labels() for h in halves])
        hard["copied"].append(transitions_hidden(lab, ~np.concatenate(copied))[0])
        base_hidden["copied"].append(np.mean(~np.concatenate(copied)))
        for cond, kw in (("ball const", dict(players=False, ball_hazard="constant")),
                         ("ball", dict(players=False)), ("ball+players", dict())):
            for seed in SEEDS:
                dh = [degrade.apply(model, h, seed=seed, **kw) for h in halves]
                score([nearest_team(h, ~np.isnan(h.ball[:, 0])) for h in dh], cond)
                hid = ~np.concatenate([~np.isnan(h.ball[:, 0]) for h in dh])
                hard[cond].append(transitions_hidden(lab, hid)[0])
                base_hidden[cond].append(np.mean(hid))

    print(f"\nNearest-player baseline, spellF1 averaged over {len(matches)} matches"
          f" (stochastic conditions also over {len(SEEDS)} seeds)")
    print(f"{'condition':52}" + "".join(f"{'+-' + str(w):>8}" for w in WINDOWS))
    names = {"clean": "ball always visible, clean players",
             "copied": "ball hidden: provided pattern copied (old)",
             "ball const": "ball hidden: constant hazard, real gap lengths",
             "ball": "ball hidden: learned hazard, real gap lengths",
             "ball+players": "learned ball hazard + players dropped + smoothed"}
    for c in results:
        print(f"{names[c]:52}" + "".join(f"{np.mean(results[c][w]):8.3f}" for w in WINDOWS))

    print("\nBall hidden at the moment of a real H<->A possession change   (share of all frames hidden)")
    for c in hard:
        print(f"   {names[c]:52} {np.nanmean(hard[c]):6.1%}   ({np.mean(base_hidden[c]):.1%})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--matches", nargs="+", choices=ALL_MATCHES, help="default: every downloaded match")
    main(p.parse_args().matches)
