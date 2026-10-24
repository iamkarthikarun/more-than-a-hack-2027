"""Make clean DFL tracking look like the provided automatic tracking, using only tracking.

Measured on the provided tactical match (no labels used anywhere):
  * the ball is hidden on ~41% of in-play frames in long gaps, and it vanishes when it is slow, near
    a player or near a touchline, not when it is fast
  * a player track ends mostly when another player is within 2 m (occlusion), at low speed or
    near a touchline; only ~55-60% of frames have all 11 players of a team tracked
  * positions are far smoother than DFL's raw positions

So the simulator (1) smooths the DFL positions, (2) drops player tracks and (3) hides the ball with
hazard models fitted on the provided match, with gap lengths drawn from the provided match's own.
Everything is in metres with the origin in a corner, like the provided data once scaled by PITCH.

Not modelled: single-step ball jumps of 2-4 m (0.19% of ball frames), and the extra phantom tracks
that give a team 12 or 13 tracked players on ~5-8% of frames.
"""

import os
import sys
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import tracking  # noqa: E402
from games import half_video_blocks  # noqa: E402

FPS = 25
PITCH = np.array(tracking.PITCH, np.float32)
CENTRE = (PITCH / 2).astype(np.float32)      # DFL is centred; the provided data has its origin in a corner
SPEED_LAG = 4
PLAYER_NEG_RATE = 20                          # keep 1 in 20 negatives when fitting, re-weighted


def _speed(xy, lag=SPEED_LAG):
    out = np.full(xy.shape[:-1], np.nan, np.float32)
    out[lag:] = np.linalg.norm(xy[lag:] - xy[:-lag], axis=-1) * FPS / lag
    return out


def player_features(xy):
    """(T, K, 4): log1p nearest-other-player distance, touchline distance, goal-line distance,
    log1p speed. NaN where the player is not tracked."""
    T, K, _ = xy.shape
    nn = np.full((T, K), np.nan, np.float32)
    eye = np.arange(K)
    for a in range(0, T, 2000):
        x = xy[a:a + 2000]
        d = np.linalg.norm(x[:, :, None] - x[:, None], axis=-1)
        d[:, eye, eye] = np.nan
        d = np.where(np.isnan(d), np.inf, d)
        m = d.min(2)
        m[np.isinf(m)] = np.nan
        nn[a:a + 2000] = m
    nn[np.isnan(xy[..., 0])] = np.nan
    touch = np.clip(np.minimum(xy[..., 1], PITCH[1] - xy[..., 1]), 0, PITCH[1] / 2)
    goal = np.clip(np.minimum(xy[..., 0], PITCH[0] - xy[..., 0]), 0, PITCH[0] / 2)
    return np.stack([np.log1p(nn), touch, goal, np.log1p(_speed(xy))], -1).astype(np.float32)


def ball_features(ball, xy):
    """(T, 5): log1p ball speed, log1p nearest-player distance, touchline, goal-line, players within 5 m."""
    d = np.linalg.norm(xy - ball[:, None, :], axis=-1)
    dd = np.where(np.isnan(d), np.inf, d)
    nn = dd.min(1)
    nn[np.isinf(nn)] = np.nan
    touch = np.clip(np.minimum(ball[:, 1], PITCH[1] - ball[:, 1]), 0, PITCH[1] / 2)
    goal = np.clip(np.minimum(ball[:, 0], PITCH[0] - ball[:, 0]), 0, PITCH[0] / 2)
    return np.stack([np.log1p(_speed(ball)), np.log1p(nn), touch, goal, (dd < 5).sum(1)], -1).astype(np.float32)


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))


def second_diff_stats(xy):
    """Median, p90, p99 of the |second difference| of position in mm per frame^2."""
    a2 = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=-1).ravel()
    a2 = a2[~np.isnan(a2)] * 1000
    return np.percentile(a2, [50, 90, 99])


def smooth(xy, sigma):
    """Gaussian-smooth each tracked stretch of each object (xy: (T, K, 2) or (T, 2))."""
    if sigma <= 0:
        return xy.copy()
    squeeze = xy.ndim == 2
    x = xy[:, None] if squeeze else xy
    out = np.full_like(x, np.nan)
    idx = np.arange(len(x))
    for k in range(x.shape[1]):
        v = ~np.isnan(x[:, k, 0])
        if v.sum() < 10:
            continue
        filled = np.stack([np.interp(idx, idx[v], x[v, k, c]) for c in (0, 1)], -1)
        out[v, k] = gaussian_filter1d(filled, sigma, axis=0, mode="nearest")[v]
    return out[:, 0] if squeeze else out


def _renewal_mask(p, draw_gap, rng):
    """Hidden-mask of an alternating process: onsets follow per-frame probabilities p, gap lengths
    come from draw_gap(rng). Onsets are drawn by inverting the cumulative hazard."""
    T = len(p)
    H = np.cumsum(-np.log1p(-np.clip(np.nan_to_num(p, nan=0.0), 0, 0.5)))
    miss = np.zeros(T, bool)
    t = 0
    while t < T:
        j = int(np.searchsorted(H, (H[t - 1] if t else 0.0) + rng.exponential()))
        if j >= T:
            break
        g = draw_gap(rng)
        miss[j:j + g] = True
        t = j + g
    return miss


@dataclass
class DegradeModel:
    ball_lr: object
    player_lr: object
    ball_gaps: np.ndarray        # frames, the provided match's own ball gaps
    smooth_sigma: float
    target_visible: float        # ball visible share of in-play frames on the provided match
    target_deficit: np.ndarray   # share of frames a team has 0, 1, 2, >=3 of its 11 players untracked
    diagnostics: dict
    player_scale: float = 1.0    # multiplier on the fitted track-end hazard
    player_mu: float = 5.0       # log-normal location (log frames) of a player's absence
    player_sigma: float = 1.2


def _ours():
    t = tracking.load()
    return t, t.xy * PITCH, t.ball[:, :2] * PITCH, half_video_blocks("BAR-ATH")


def _gap_lengths(vis, blocks):
    out = []
    for _, a, b in blocks:
        v = vis[a:b + 1]
        e = np.diff(np.r_[1, v.astype(int), 1])
        s, f = np.where(e == -1)[0], np.where(e == 1)[0]
        out += [fe - ss for ss, fe in zip(s, f) if ss > 0 and fe < len(v)]
    return np.array(out)


def _fit_hazard(X, y, w, half_id, name):
    """Fit on all halves; also report cross-half skill so we know the covariates really help."""
    diag = {}
    skill = []
    for h in (0, 1):
        tr, te = half_id != h, half_id == h
        m = _lr().fit(X[tr], y[tr], logisticregression__sample_weight=w[tr])
        p = m.predict_proba(X[te])[:, 1]
        base = np.average(y[tr], weights=w[tr])
        ll_m = log_loss(y[te], p, sample_weight=w[te], labels=[0, 1])
        ll_b = log_loss(y[te], np.full(te.sum(), base), sample_weight=w[te], labels=[0, 1])
        skill.append((1 - ll_m / ll_b, roc_auc_score(y[te], p, sample_weight=w[te])))
    diag[name + "_log_loss_gain_vs_constant"] = [round(float(s[0]), 4) for s in skill]
    diag[name + "_auc"] = [round(float(s[1]), 3) for s in skill]
    return _lr().fit(X, y, logisticregression__sample_weight=w), diag


def fit(calibration_half, seed=0, verbose=True):
    """Learn the degradation from the provided match. calibration_half is a clean DFL half used only
    to set the two scalar knobs that need a simulation to calibrate (player absence length, smoothing)."""
    rng = np.random.RandomState(seed)
    t, xy, ball, blocks = _ours()
    vis = ~np.isnan(ball[:, 0])
    seen = ~np.isnan(xy[..., 0])
    diag = {}

    # --- ball: onset hazard from covariates, gap length from the match's own gaps
    bf = ball_features(ball, xy)
    X, y, hid = [], [], []
    for h, (_, a, b) in enumerate(blocks):
        i = np.arange(a, b)
        ok = vis[i] & np.isfinite(bf[i]).all(1)
        X.append(bf[i][ok]); y.append((~vis[i + 1])[ok]); hid.append(np.full(ok.sum(), h))
    X, y, hid = np.concatenate(X), np.concatenate(y).astype(int), np.concatenate(hid)
    ball_lr, d = _fit_hazard(X, y, np.ones(len(y)), hid, "ball")
    diag.update(d)
    diag["ball_onsets"] = int(y.sum())
    gaps = _gap_lengths(vis, blocks)

    # --- players: the hazard that a tracked player's track ends on the next frame
    pf = player_features(xy)
    Xp, yp, wp, hp = [], [], [], []
    for h, (_, a, b) in enumerate(blocks):
        i = np.arange(a, b)
        ok = seen[i] & np.isfinite(pf[i]).all(-1)
        ii, kk = np.nonzero(ok)
        lab = (~seen[i + 1])[ii, kk]
        keep = lab | (rng.rand(len(lab)) < 1.0 / PLAYER_NEG_RATE)
        ii, kk, lab = ii[keep], kk[keep], lab[keep]
        Xp.append(pf[i][ii, kk]); yp.append(lab.astype(int))
        wp.append(np.where(lab, 1.0, PLAYER_NEG_RATE)); hp.append(np.full(len(lab), h))
    Xp, yp, wp, hp = (np.concatenate(v) for v in (Xp, yp, wp, hp))
    player_lr, d = _fit_hazard(Xp, yp, wp, hp, "player")
    diag.update(d)
    diag["player_track_ends"] = int(yp.sum())

    # --- targets from the provided match: ball visibility, and how many players of a team go untracked
    sel = np.concatenate([np.arange(a, b + 1) for _, a, b in blocks])
    deficits = [deficit_pmf((seen[sel][:, t.team == lab]).sum(1), 11) for lab in "HA"]
    target_deficit = np.mean(deficits, 0)
    diag["target_deficit_0_1_2_3plus"] = np.round(target_deficit, 3).tolist()
    diag["target_ball_visible"] = round(float(vis[sel].mean()), 3)

    model = DegradeModel(ball_lr, player_lr, gaps, 3.0, float(vis[sel].mean()), target_deficit, diag)
    model.smooth_sigma = calibrate_smoothing(model, calibration_half, xy, blocks)
    calibrate_players(model, calibration_half)
    diag["smooth_sigma"] = model.smooth_sigma
    diag["player_scale_mu_sigma"] = [model.player_scale, round(model.player_mu, 2), model.player_sigma]
    if verbose:
        for k, v in diag.items():
            print(f"  {k}: {v}")
    return model


def calibrate_smoothing(model, half, ours_xy, blocks):
    """Pick the Gaussian sigma (frames) whose jitter statistics are closest to the provided data's."""
    target = np.concatenate([[second_diff_stats(ours_xy[a:b + 1])] for _, a, b in blocks]).mean(0)
    xy = half.xy + CENTRE
    best, best_err = 0.0, np.inf
    for s in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        err = np.abs(np.log(second_diff_stats(smooth(xy, s)) / target)).sum()
        if err < best_err:
            best, best_err = float(s), err
    return best


def _player_prob(model, xy):
    f = player_features(xy)
    p = np.zeros(xy.shape[:2], np.float32)
    ok = np.isfinite(f).all(-1)
    p[ok] = model.player_lr.predict_proba(f[ok])[:, 1]
    return p


def _draw_player_gap(model):
    return lambda rng: max(1, int(np.exp(rng.normal(model.player_mu, model.player_sigma))))


def deficit_pmf(tracked, clean):
    """Share of frames with 0, 1, 2, >=3 players of a team untracked, given how many are on the pitch."""
    d = np.clip(clean - tracked, 0, 3).astype(int)
    return np.bincount(d, minlength=4) / len(d)


def calibrate_players(model, half):
    """Pick the hazard scale and the absence-length distribution so the share of frames with 0, 1,
    2, >=3 of a team's players untracked matches the provided match."""
    xy = smooth(half.xy + CENTRE, model.smooth_sigma)
    p0 = _player_prob(model, xy)
    present = np.isfinite(xy[..., 0])
    clean = {lab: present[:, half.team == lab].sum(1) for lab in "HA"}
    best = (np.inf, None)
    for scale in (0.25, 0.5, 1.0):
        for mu in np.arange(3.0, 8.01, 0.5):
            for sigma in (0.8, 1.2, 1.6, 2.0):
                model.player_scale, model.player_mu, model.player_sigma = scale, float(mu), sigma
                errs = []
                for seed in (1, 2):
                    r = np.random.RandomState(seed)
                    lost = np.stack([_renewal_mask(scale * p0[:, k], _draw_player_gap(model), r)
                                     for k in range(xy.shape[1])], 1)
                    tracked = present & ~lost
                    pmf = np.mean([deficit_pmf(tracked[:, half.team == lab].sum(1), clean[lab])
                                   for lab in "HA"], 0)
                    errs.append(np.abs(pmf - model.target_deficit).sum())
                if np.mean(errs) < best[0]:
                    best = (np.mean(errs), (scale, float(mu), sigma))
    model.player_scale, model.player_mu, model.player_sigma = best[1]
    model.diagnostics["player_deficit_l1_after_fit"] = round(float(best[0]), 3)


def _ball_hazard(model, b, xy, mode):
    """Relative per-frame hazard of the ball vanishing. Only its shape matters: the level is set
    afterwards so the half's visible share matches the provided match's."""
    if mode == "constant":
        return np.full(len(b), 1.0, np.float32)
    p = np.zeros(len(b), np.float32)
    f = ball_features(b, xy)
    ok = np.isfinite(f).all(1)
    p[ok] = model.ball_lr.predict_proba(f[ok])[:, 1]
    return p


def _ball_gap_drawer(model):
    gaps = model.ball_gaps
    return lambda r: max(1, int(gaps[r.randint(len(gaps))] * np.exp(r.normal(0, 0.1))))


def _scale_to_visibility(model, p0):
    """The multiplier on p0 whose simulated ball-visible share equals the provided match's. The
    provided match shows the same share in both halves, so this is done per half."""
    lo, hi = 1e-4, 1e4
    for _ in range(16):
        c = np.sqrt(lo * hi)
        share = np.mean([1 - _renewal_mask(c * p0, _ball_gap_drawer(model), np.random.RandomState(s)).mean()
                         for s in (1, 2, 3)])
        lo, hi = (c, hi) if share > model.target_visible else (lo, c)
    return np.sqrt(lo * hi)


def apply(model, half, seed=0, ball=True, players=True, smooth_positions=True, ball_hazard="learned"):
    """A copy of a dfl.Half whose tracking looks like the provided match's. Coordinates are metres,
    origin in a corner. Labels are untouched: they come from the original, clean ball annotation.
    ball_hazard: 'learned' (covariates: slow ball, near a player or a touchline) or 'constant'
    (same visible share and gap lengths, independent of play)."""
    rng = np.random.RandomState(seed)
    xy = half.xy + CENTRE
    b = half.ball[:, :2] + CENTRE
    if smooth_positions:
        xy = smooth(xy, model.smooth_sigma)
    if players:
        p = _player_prob(model, xy) * model.player_scale
        lost = np.stack([_renewal_mask(p[:, k], _draw_player_gap(model), rng) for k in range(xy.shape[1])], 1)
        xy = np.where(lost[..., None], np.nan, xy)
    if ball:
        p0 = _ball_hazard(model, b, xy, ball_hazard)
        hidden = _renewal_mask(_scale_to_visibility(model, p0) * p0, _ball_gap_drawer(model), rng)
        b = np.where(hidden[:, None], np.nan, b)
    return replace(half, xy=xy.astype(np.float32),
                   ball=np.column_stack([b, half.ball[:, 2]]).astype(np.float32))
