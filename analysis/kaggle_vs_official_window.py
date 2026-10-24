"""How well do Kaggle-labels agree with the Official public ground truth, on its 5-minute window?

    uv run python analysis/kaggle_vs_official_window.py

A check of what the labels mean, not a tuning tool: no rule is changed by anything printed here. The public
ground truth covers video frames 11185-18684 (first half). Names: Official = organisers' data, Kaggle = the
third-party event log, Kaggle-labels = H/A/D labels derived from it (analysis/pseudo_gt.py).
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from analysis import pseudo_gt, tracking  # noqa: E402
from analysis.kaggle_first_half import gap_class, hidden_runs, load_half  # noqa: E402
from analysis.video_time import frame_to_timecode  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
A, B = 11185, 18684
GT_CSV = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
CODE = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}


def runs(lab, classes):
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    s, e = np.r_[0, ch], np.r_[ch, len(lab)]
    return [(lab[i], i, j) for i, j in zip(s, e) if lab[i] in classes]


def transitions(lab):
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    out = []
    for i in ch:
        p, n = lab[i - 1], lab[i]
        kind = "team change" if p in "HA" and n in "HA" else ("play stops" if n == "D" else "play restarts")
        out.append((i, kind, p + ">" + n))
    return out


def main():
    gt_all = pd.read_csv(GT_CSV).label.values
    gt, kg = gt_all[A:B + 1], pseudo_gt.labels()[A:B + 1]
    vis = ~np.isnan(tracking.load().ball[A:B + 1, 0])
    n = len(gt)
    print(f"Official public window: video {frame_to_timecode(A)} - {frame_to_timecode(B)} ({n} frames, {n / FPS / 60:.1f} min); "
          f"ball hidden on {1 - vis.mean():.1%} of it\n")

    print("1. Frame by frame (rows Official, columns Kaggle-labels)")
    print(f"   {'':9}{'H':>7}{'A':>7}{'D':>7}{'':>6}{'n':>6}{'recall':>8}")
    for g in "HAD":
        row = [(kg[gt == g] == k).sum() for k in "HAD"]
        print(f"   Official {g}" + "".join(f"{x:7d}" for x in row) + f"{'':6}{sum(row):6d}{row['HAD'.index(g)] / max(sum(row), 1):8.1%}")
    print("   precision " + "".join(f"{(gt[kg == k] == k).sum() / max((kg == k).sum(), 1):7.1%}" for k in "HAD"))
    alive_g, alive_k = gt != "D", kg != "D"
    both = alive_g & alive_k
    print(f"   all three classes agree on {np.mean(gt == kg):.1%} of frames; alive/dead agree on {np.mean(alive_g == alive_k):.1%}; "
          f"on frames both call alive, the team agrees on {np.mean(gt[both] == kg[both]):.1%}\n")

    gt_codes = np.full(len(gt_all), pm.X, np.int8)
    gt_codes[A:B + 1] = [CODE[c] for c in gt]
    k_codes = np.full(len(gt_all), pm.X, np.int8)
    k_codes[A:B + 1] = [CODE[c] for c in kg]
    r = pm.evaluate(gt_codes, k_codes)
    print("2. Whole possessions (the ranking metric's view)")
    print(f"   Official {r['n_gt_spells']} possessions, Kaggle-labels {r['n_pred_spells']}; spellF1 of Kaggle-labels scored against Official: "
          f"{r['spell_f1_mean']:.3f}  (the shipped baseline scores about 0.3 on this window)")
    gs, ks = runs(gt, "HA"), runs(kg, "HA")
    print(f"   {'Official spell (team, start, length)':42}{'best Kaggle match IoU':>22}{'start off':>11}{'end off':>9}")
    unmatched = 0
    for lab, i, j in gs:
        best = (0.0, None)
        for l2, i2, j2 in ks:
            if l2 != lab:
                continue
            inter = max(0, min(j, j2) - max(i, i2))
            iou = inter / (j - i + j2 - i2 - inter)
            if iou > best[0]:
                best = (iou, (i2, j2))
        if best[1] is None or best[0] < 0.3:
            unmatched += 1
        so, eo = ((best[1][0] - i) / FPS, (best[1][1] - j) / FPS) if best[1] else (np.nan, np.nan)
        print(f"   {lab} {frame_to_timecode(A + i)[3:]} {(j - i) / FPS:5.1f} s{'':20}{best[0]:22.2f}{so:+11.1f}{eo:+9.1f}")
    print(f"   {unmatched} of {len(gs)} Official possessions have no Kaggle spell with IoU of at least 0.3\n")

    print("3. When the changes happen: each Official transition against the nearest Kaggle-labels one of the same kind")
    gts, kgs = transitions(gt), transitions(kg)
    print(f"   {'kind':15}{'Official':>9}{'found within 3 s':>18}{'median offset':>15}{'10th..90th pct':>18}")
    for kind in ("play stops", "play restarts", "team change"):
        G = [(i, t) for i, k, t in gts if k == kind]
        K = [(i, t) for i, k, t in kgs if k == kind]
        offs, wrong = [], 0
        for i, t in G:
            cand = [(abs(i2 - i), i2 - i, t2) for i2, t2 in K if abs(i2 - i) <= 3 * FPS and (kind != "team change" or t2 == t)]
            if cand:
                d = min(cand)
                offs.append(d[1] / FPS)
        if offs:
            print(f"   {kind:15}{len(G):9d}{len(offs):18d}{np.median(offs):+14.1f}s{np.percentile(offs, 10):+9.1f}..{np.percentile(offs, 90):+.1f}s")
        else:
            print(f"   {kind:15}{len(G):9d}{0:18d}")
    print("   (a negative offset means Kaggle changes before Official)\n")

    print("4. Dead-ball stretches")
    gd, kd = runs(gt, "D"), runs(kg, "D")
    print(f"   Official {len(gd)} stretches ({sum(j - i for _, i, j in gd) / FPS:.0f} s), Kaggle-labels {len(kd)} ({sum(j - i for _, i, j in kd) / FPS:.0f} s)")
    for _, i, j in gd:
        ov = [(i2, j2) for _, i2, j2 in kd if min(j, j2) > max(i, i2)]
        txt = "; ".join(f"Kaggle {frame_to_timecode(A + i2)[3:]}-{frame_to_timecode(A + j2)[3:]} (start {(i2 - i) / FPS:+.1f} s, end {(j2 - j) / FPS:+.1f} s)" for i2, j2 in ov) or "NO Kaggle dead stretch"
        print(f"   Official {frame_to_timecode(A + i)[3:]}-{frame_to_timecode(A + j)[3:]} ({(j - i) / FPS:.1f} s)  <->  {txt}")
    for _, i2, j2 in kd:
        if not any(min(j2, j) > max(i2, i) for _, i, j in gd):
            print(f"   Kaggle-only dead stretch {frame_to_timecode(A + i2)[3:]}-{frame_to_timecode(A + j2)[3:]} ({(j2 - i2) / FPS:.1f} s): Official calls it alive")
    print()

    print("5. Where the team differs while both call it alive (0.5 s or longer)")
    steps = load_half()[0]
    diff = both & (gt != kg)
    ch = np.flatnonzero(np.diff(np.r_[0, diff.astype(int), 0]))
    shown = 0
    for i, j in zip(ch[::2], ch[1::2]):
        if j - i < FPS // 2:
            continue
        ev = [f"{d['team']} {d['name']}" for d in steps if abs(d["frame"] - (A + i)) <= 2 * FPS]
        print(f"   {frame_to_timecode(A + i)[3:]} {(j - i) / FPS:4.1f} s  Official {gt[i]}, Kaggle {kg[i]}   events within 2 s of the start: {', '.join(ev[:5]) or '-'}")
        shown += 1
    print(f"   {shown} stretches; the rest of the disagreement is shorter than 0.5 s\n")

    print("6. The missing-ball gaps inside this window")
    hr = hidden_runs(vis)
    print(f"   {len(hr)} gaps, {int((~vis).sum())} hidden frames. Hidden frames by label: "
          f"Official H {np.mean(gt[~vis] == 'H'):.0%} A {np.mean(gt[~vis] == 'A'):.0%} D {np.mean(gt[~vis] == 'D'):.0%} | "
          f"Kaggle-labels H {np.mean(kg[~vis] == 'H'):.0%} A {np.mean(kg[~vis] == 'A'):.0%} D {np.mean(kg[~vis] == 'D'):.0%}")
    same = 0
    print(f"   {'gap':>14}{'length':>8}  {'Official says':44}{'Kaggle-labels say':44}")
    for s, e in hr:
        cg, ck = gap_class(gt[s:e]), gap_class(kg[s:e])
        same += cg == ck
        print(f"   {frame_to_timecode(A + s)[3:]:>14}{(e - s) / FPS:7.1f}s  {cg:44}{ck:44}")
    print(f"   the two label sets put the gap in the same class for {same} of {len(hr)} gaps")


if __name__ == "__main__":
    main()
