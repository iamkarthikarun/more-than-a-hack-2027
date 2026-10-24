"""Hold-the-last-owner possession labels from tracking.

The shipped baseline gives the ball to the nearest player and calls every frame without a ball
"dead". The provided match has no ball on 41% of frames, mostly in long gaps, so that throws away most
of the match. Here:

  * the owner changes only on a frame where the ball is visible AND a player is within `r_control`
    metres; it is held through gaps and through frames where the ball is in flight
  * dead ball needs its own evidence at a visible frame: the ball outside the pitch, or hardly moving
    over `still_frames` frames (centred window, so the start and end of a stoppage are timed well)
  * the dead/alive state is held through a gap for at most `dead_hold_max` frames
  * runs shorter than `min_spell` (possession) or `min_dead` (stoppage) frames are merged away

Input is a dfl.Half-like object in metres with the origin in a corner (what degrade.apply returns).
"""

from dataclasses import dataclass

import numpy as np

L, W = 105.0, 68.0
HOME, AWAY = 0, 1


@dataclass(frozen=True)
class Params:
    r_control: float = 3.0
    d_still: float = 1.0
    still_frames: int = 50
    dead_hold_max: int = 0          # frames; 0 means hold without limit
    min_spell: int = 75
    min_dead: int = 50
    margin: float = 0.5


def prepare(half):
    """The parameter-free part: who is nearest the ball, how far, where the ball is."""
    ball = half.ball[:, :2]
    vis = ~np.isnan(ball[:, 0])
    d = np.linalg.norm(half.xy - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.argmin(1)
    return {
        "ball": ball, "vis": vis,
        "d1": d[np.arange(len(d)), nearest],
        "team": np.where(half.team[nearest] == "H", HOME, AWAY).astype(np.int8),
    }


def _ffill(values, valid, max_age=0, default=0):
    """Forward-fill values over frames where valid is False, at most max_age frames (0 = unlimited).
    Frames before the first valid one take the first valid value."""
    n = len(values)
    last = np.maximum.accumulate(np.where(valid, np.arange(n), -1))
    out = values[np.maximum(last, 0)].copy()
    if not valid.any():
        return np.full(n, default, values.dtype)
    out[: np.flatnonzero(valid)[0]] = values[np.flatnonzero(valid)[0]]
    if max_age:
        stale = (np.arange(n) - last > max_age) & (last >= 0)
        out[stale] = default
    return out


def merge_short(lab, min_spell, min_dead):
    """Relabel runs shorter than their minimum with the label of the longer neighbouring run, shortest
    first, until none is left."""
    lab = lab.copy()
    for _ in range(10000):
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts, ends = np.r_[0, change], np.r_[change, len(lab)]
        lens = ends - starts
        vals = lab[starts]
        need = np.where(vals == "D", min_dead, min_spell)
        short = lens < need
        if not short.any() or len(starts) == 1:
            break
        left = np.r_[np.iinfo(np.int64).max, lens[:-1]]      # missing neighbour = never the shorter
        right = np.r_[lens[1:], np.iinfo(np.int64).max]
        local_min = short & (lens < left) & (lens <= right)
        for i in np.flatnonzero(local_min):
            lv = vals[i - 1] if i > 0 else None
            rv = vals[i + 1] if i + 1 < len(vals) else None
            if lv is None:
                pick = rv
            elif rv is None:
                pick = lv
            else:
                pick = lv if lens[i - 1] >= lens[i + 1] else rv
            lab[starts[i]:ends[i]] = pick
    return lab


def labels(prep, p=Params()):
    """'H' / 'A' / 'D' for every frame."""
    ball, vis = prep["ball"], prep["vis"]
    n = len(vis)

    control = vis & (prep["d1"] < p.r_control)
    owner = _ffill(prep["team"], control, default=HOME)

    oob = vis & ((ball[:, 0] < -p.margin) | (ball[:, 0] > L + p.margin)
                 | (ball[:, 1] < -p.margin) | (ball[:, 1] > W + p.margin))
    k = p.still_frames // 2
    disp = np.full(n, np.inf)
    disp[k:n - k] = np.linalg.norm(ball[2 * k:] - ball[:n - 2 * k], axis=1)
    still = vis & (np.nan_to_num(disp, nan=np.inf) < p.d_still)
    dead = _ffill((oob | still).astype(np.int8), vis, max_age=p.dead_hold_max, default=0)

    lab = np.where(dead == 1, "D", np.where(owner == HOME, "H", "A"))
    return merge_short(lab, p.min_spell, p.min_dead)
