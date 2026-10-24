"""Test methods on the REAL provided tracking against the event-derived pseudo ground truth.

    uv run python analysis/real_eval.py

The pseudo ground truth (analysis/pseudo_gt.py) is an approximate, second-level-accurate proxy built from
third-party events, so this is a sanity check of whether gains measured on degraded DFL carry over to the
real tracking, not a score. Nothing here is tuned on it: the hold-the-last-owner settings come from the
leave-one-match-out runs on DFL. The final evaluation is on other, unreleased matches.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import hold_owner as ho, pseudo_gt, tracking  # noqa: E402
from analysis.dfl_baseline import encode, nearest_team  # noqa: E402
from baseline import smooth as vote  # noqa: E402
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

SETTINGS = {
    "hold, default": ho.Params(),
    "hold, DFL-tuned (raw labels)": ho.Params(r_control=0.75, d_still=1.0, dead_hold_max=0, min_spell=10, min_dead=10),
    "hold, DFL-tuned (merged 3 s)": ho.Params(r_control=0.75, d_still=1.0, dead_hold_max=0, min_spell=75, min_dead=50),
}


def real_halves():
    t = tracking.load()
    P = np.array(tracking.PITCH, np.float32)
    xy, ball = t.xy * P, t.ball[:, :2] * P
    return [(SimpleNamespace(xy=xy[a:b + 1], ball=ball[a:b + 1], team=t.team), a, b)
            for _, a, b in games.half_video_blocks(games.GAME)]


def assemble(per_half, halves):
    out = np.full(pseudo_gt.N_FRAMES, "X", dtype="<U1")
    for lab, (_, a, b) in zip(per_half, halves):
        out[a:b + 1] = lab
    return encode(out)


def main():
    halves = real_halves()
    methods = {"nearest + vote50 (shipped baseline)":
               [vote(nearest_team(h, ~np.isnan(h.ball[:, 0])), 50) for h, _, _ in halves]}
    preps = [ho.prepare(h) for h, _, _ in halves]
    for name, params in SETTINGS.items():
        methods[name] = [ho.labels(p, params) for p in preps]

    vis = np.concatenate([~np.isnan(h.ball[:, 0]) for h, _, _ in halves])
    print(f"real provided tracking, graded halves: ball visible on {vis.mean():.1%} of frames\n")
    for merged in (False, True):
        gt = encode(pseudo_gt.labels(0.0, merged))
        spells = pm.evaluate(gt, gt)["n_gt_spells"]
        print(f"=== pseudo ground truth, merged={merged}  ({spells} possessions)")
        print(f"{'method':38}{'spellF1':>9}{'90% band':>16}{'macroF1':>9}{'alive/dead':>11}{'team acc':>9}{'pred spells':>12}")
        for name, per_half in methods.items():
            r = pm.evaluate(gt, assemble(per_half, halves), bootstrap_n=500)
            print(f"{name:38}{r['spell_f1_mean']:9.3f}{r['spell_bootstrap_low']:8.3f}-{r['spell_bootstrap_high']:5.3f}"
                  f"{r['macro_f1']:9.3f}{r['alive_dead_f1']:11.3f}{r['team_acc_both_alive']:9.3f}{r['n_pred_spells']:12d}")
        print()


if __name__ == "__main__":
    main()
