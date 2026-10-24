"""Render the tactical tracking as a real-time MP4, one window at a time (default 10 minutes).

    uv run python analysis/make_video.py --list                 # the windows that tile both graded blocks
    uv run python analysis/make_video.py --window 1             # first 10 minutes from kick-off
    uv run python analysis/make_video.py --all --workers 6
    uv run python analysis/make_video.py --start 11185 --seconds 60

What is on screen, besides players (numbered by tracker slot, not by person) and the ball with a 1 s trail:
  * a slot that stops being tracked leaves a fading ring where it was last seen;
  * a slot that comes back somewhere a player could not have run to (more than 10 m/s plus 1 m over the gap)
    is joined to its old spot by a magenta line, so teleports and slot hand-offs are easy to spot;
  * a ball that is not tracked leaves a ring at its last position, the banner counts how long it has been
    missing, and when it returns an orange line shows how far it moved while unseen;
  * a timeline under the pitch shows ball tracked / not tracked, Kaggle-labels and (public window only)
    Official, with the playhead; the last Kaggle events are listed as they happen.
Labels and events are only for looking at the data. Nothing here feeds the pipeline, and RULES.md forbids
tuning on the public ground truth. Output goes to analysis/figures/videos/, which is git-ignored.
"""

import argparse
import os
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402
from analysis import tracking  # noqa: E402
from analysis.video_time import CLOCK_TO_VIDEO, frame_to_timecode, half_of  # noqa: E402

FPS = 25
PL, PW = tracking.PITCH
S, M = 11, 3                       # pixels per metre, metres of margin around the pitch
HDR, FTR = 64, 150
PH = int((PW + 2 * M) * S)
W = int((PL + 2 * M) * S)
W += W % 2
H = HDR + PH + FTR
OUT_DIR = os.path.join(ROOT, "analysis", "figures", "videos")
GT_CSV = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

GHOST = 2 * FPS                    # how long a lost slot leaves a ring
TELE = FPS                         # how long a teleport line stays
BALL_GHOST = 15 * FPS
BALL_JUMP = 40
V_MAX, SLACK = 10.0, 1.0           # m/s and metres: what a player could cover while untracked

BGR = lambda h: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))  # noqa: E731  '#rrggbb' -> (b, g, r)
HOME, AWAY = BGR("#d62728"), BGR("#1f77b4")
GRASS, GRASS_DARK, DARK = BGR("#3f8f4f"), BGR("#388046"), (32, 32, 32)
WHITE, YELLOW, MAGENTA, ORANGE, GREY, RED, GREEN = (255, 255, 255), (0, 255, 255), (255, 0, 255), (0, 150, 255), (170, 170, 170), (60, 60, 230), (80, 200, 80)
STATE_COL = {"H": HOME, "A": AWAY, "D": GREY, "X": (70, 70, 70)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def px(x, y):
    return int(round((x + M) * S)), int(round(HDR + (PW + M - y) * S))


def mix(a, b, t):
    t = min(max(t, 0.0), 1.0)
    return tuple(int(round(p * (1 - t) + q * t)) for p, q in zip(a, b))


def txt(img, s, org, scale=0.55, col=WHITE, th=1, outline=False):
    # Hershey glyph spacing grows with thickness, so a thick black copy drifts; offset thin copies instead.
    if outline:
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.putText(img, s, (org[0] + dx, org[1] + dy), FONT, scale, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


def ascii_(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def mmss(seconds):
    return f"{int(seconds) // 60:02d}:{seconds - 60 * (int(seconds) // 60):04.1f}"


def pitch_bg():
    img = np.full((H, W, 3), DARK, np.uint8)
    img[HDR:HDR + PH] = mix(GRASS, (0, 0, 0), 0.25)
    (x0, y0), (x1, y1) = px(0, PW), px(PL, 0)
    img[y0:y1 + 1, x0:x1 + 1] = GRASS
    for k in range(0, 20, 2):
        a, b = px(k * PL / 20, 0)[0], px((k + 1) * PL / 20, 0)[0]
        img[y0:y1 + 1, a:b] = GRASS_DARK
    line = dict(color=WHITE, thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), **line)
    cv2.line(img, px(PL / 2, 0), px(PL / 2, PW), **line)
    cv2.circle(img, px(PL / 2, PW / 2), int(9.15 * S), **line)
    cv2.circle(img, px(PL / 2, PW / 2), 3, WHITE, -1, cv2.LINE_AA)
    for left in (True, False):
        sx = (lambda d: d) if left else (lambda d: PL - d)
        for depth, wide in ((16.5, 40.32), (5.5, 18.32)):
            cv2.rectangle(img, px(sx(0), PW / 2 + wide / 2), px(sx(depth), PW / 2 - wide / 2), **line)
        cv2.rectangle(img, px(sx(0), PW / 2 + 3.66), px(sx(-2), PW / 2 - 3.66), **line)
        c = px(sx(11), PW / 2)
        cv2.circle(img, c, 3, WHITE, -1, cv2.LINE_AA)
        r = int(9.15 * S)
        cv2.ellipse(img, c, (r, r), 0, -53 if left else 127, 53 if left else 233, WHITE, 2, cv2.LINE_AA)
    return img


def windows(minutes):
    out, size = [], int(minutes * 60 * FPS)
    for p, a, b in games.half_video_blocks(games.GAME):
        for i, s in enumerate(range(a, b + 1, size), 1):
            e = min(s + size - 1, b)
            out.append(dict(half=p, start=s, end=e, name=f"half{p}_w{i}_{frame_to_timecode(s)[:8].replace(':', '-')}"))
    return out


def load_extras():
    labels, official, ev = None, None, None
    try:
        from analysis import events as ev_mod
        from analysis import pseudo_gt
        raw = ev_mod.load_events()
        labels = pseudo_gt.labels(ev=raw)
        ev = raw[raw.period.isin([1, 2])].copy()
        ev["frame"] = [int(round((c + CLOCK_TO_VIDEO[int(p)]) * FPS)) for c, p in zip(ev.clock, ev.period)]
        ev = ev.sort_values("frame", kind="stable").reset_index(drop=True)
    except FileNotFoundError:
        pass
    if os.path.exists(GT_CSV):
        official = pd.read_csv(GT_CSV).label.values
    return labels, official, ev


def reappearances(vis, last, xy):
    """(frame, slot, last_seen_index, distance_m, gap_frames, plausible) for every slot that comes back."""
    fr, sl = np.nonzero(vis[1:] & ~vis[:-1])
    out = []
    for f, s in zip(fr + 1, sl):
        j = last[f - 1, s]
        if j < 0:
            continue
        d = float(np.hypot(*(xy[f, s] - xy[j, s])))
        out.append((int(f), int(s), int(j), d, int(f - j), d <= V_MAX * (f - j) / FPS + SLACK))
    return out


def render(job):
    cv2.setNumThreads(1)
    t0 = time.time()
    start, end, name = job["start"], job["end"], job["name"]
    out = os.path.join(OUT_DIR, name + ".mp4")
    os.makedirs(OUT_DIR, exist_ok=True)
    t = tracking.load()
    labels, official, events = (None, None, None) if job["no_labels"] else load_extras()
    n = end - start + 1
    lo = max(0, start - GHOST - 1)

    xy = t.xy[lo:end + 1] * np.array(tracking.PITCH, np.float32)
    vis = ~np.isnan(xy[..., 0])
    last = np.maximum.accumulate(np.where(vis, np.arange(len(xy))[:, None], -1), axis=0)
    X = np.clip(np.nan_to_num((xy[..., 0] + M) * S), 3, W - 3).astype(np.int32)
    Y = np.clip(np.nan_to_num(HDR + (PW + M - xy[..., 1]) * S), HDR + 3, HDR + PH - 3).astype(np.int32)
    tele = reappearances(vis, last, xy)
    tele_f = np.array([e[0] for e in tele] + [10 ** 9])
    col = [HOME if x == "H" else AWAY for x in t.team]

    ball = t.ball[:, :2] * np.array(tracking.PITCH, np.float32)
    bvis = ~np.isnan(ball[:, 0])
    blast = np.maximum.accumulate(np.where(bvis, np.arange(t.n), -1))
    BX = np.clip(np.nan_to_num((ball[:, 0] + M) * S), 3, W - 3).astype(np.int32)
    BY = np.clip(np.nan_to_num(HDR + (PW + M - ball[:, 1]) * S), HDR + 3, HDR + PH - 3).astype(np.int32)
    bj = []
    for f in np.flatnonzero(bvis[lo + 1:end + 1] & ~bvis[lo:end]) + lo + 1:
        j = int(blast[f - 1])
        if j >= 0 and f - j >= 5:
            bj.append((int(f), j, float(np.hypot(*(ball[f] - ball[j]))), int(f - j)))
    bj_f = np.array([e[0] for e in bj] + [10 ** 9])

    bg = pitch_bg()
    x0, x1 = 150, W - 20
    fy = HDR + PH
    edges = np.linspace(0, n, x1 - x0 + 1).astype(int)
    frac = np.add.reduceat(bvis[start:end + 1].astype(np.float32), edges[:-1]) / np.maximum(np.diff(edges), 1)
    rows = [("ball tracked", None), ("Kaggle-labels", labels), ("Official (public)", official)]
    for r, (lab, arr) in enumerate(rows):
        y = fy + 8 + 20 * r
        txt(bg, lab, (10, y + 12), 0.45)
        for c in range(x1 - x0):
            if r == 0:
                colr = mix(RED, GREEN, float(frac[c]))
            elif arr is None:
                colr = STATE_COL["X"]
            else:
                colr = STATE_COL.get(arr[start + min(int((c + .5) / (x1 - x0) * n), n - 1)], STATE_COL["X"])
            bg[y:y + 14, x0 + c] = colr
    step = 60 if n / FPS >= 240 else 10
    for s in range(-(-start // (step * FPS)) * step * FPS, end + 1, step * FPS):
        cx = x0 + int((s - start) / n * (x1 - x0))
        cv2.line(bg, (cx, fy + 66), (cx, fy + 72), WHITE, 1)
        txt(bg, mmss(s / FPS), (cx - 20, fy + 88), 0.4)

    # Ask libx264 for near-lossless quality so the 0.36-scale slot numbers stay readable.
    os.environ.setdefault("OPENCV_FFMPEG_WRITER_OPTIONS", "crf;14|preset;veryfast")
    for codec in ("avc1", "mp4v"):   # Linux wheels often lack H.264; MPEG-4 plays in VLC and Windows players
        vw = cv2.VideoWriter(out + ".part.mp4", cv2.VideoWriter_fourcc(*codec), FPS, (W, H))
        if vw.isOpened():
            break
    else:
        raise RuntimeError("OpenCV cannot write avc1 or mp4v here")
    print(f"  {name}: encoding with {codec}", flush=True)
    ev_frames = events.frame.values if events is not None else None
    for f in range(start, end + 1):
        i = f - lo
        img = bg.copy()

        # slots that are not tracked now: ring where last seen
        for s in np.flatnonzero(~vis[i]):
            j = last[i, s]
            if j >= 0 and i - j <= GHOST:
                cv2.circle(img, (int(X[j, s]), int(Y[j, s])), 9, mix(col[s], GRASS, (i - j) / GHOST), 1, cv2.LINE_AA)
        # teleports and recent comebacks
        a, b = np.searchsorted(tele_f, i - TELE + 1), np.searchsorted(tele_f, i, "right")
        for ef, s, j, d, gap, ok in tele[a:b]:
            k = (i - ef) / TELE
            if ok:
                if i - ef < 8:
                    cv2.circle(img, (int(X[ef, s]), int(Y[ef, s])), 13, WHITE, 1, cv2.LINE_AA)
                continue
            cv2.line(img, (int(X[j, s]), int(Y[j, s])), (int(X[ef, s]), int(Y[ef, s])), mix(MAGENTA, GRASS, k * .7), 1, cv2.LINE_AA)
            cv2.circle(img, (int(X[ef, s]), int(Y[ef, s])), 14, mix(MAGENTA, GRASS, k * .7), 2, cv2.LINE_AA)
        # players
        for s in np.flatnonzero(vis[i]):
            c = (int(X[i, s]), int(Y[i, s]))
            cv2.circle(img, c, 9, col[s], -1, cv2.LINE_AA)
            cv2.circle(img, c, 9, WHITE, 1, cv2.LINE_AA)
            lab = str(s + 1)
            cv2.putText(img, lab, (c[0] - 4 * len(lab), c[1] + 4), FONT, 0.36, WHITE, 1, cv2.LINE_AA)

        # ball
        pts = [(int(BX[g]), int(BY[g])) for g in range(max(0, f - FPS), f + 1) if bvis[g]]
        for g in range(max(0, f - FPS), f):
            if bvis[g] and bvis[g + 1]:
                cv2.line(img, (int(BX[g]), int(BY[g])), (int(BX[g + 1]), int(BY[g + 1])),
                         mix(YELLOW, GRASS, (f - g) / FPS), 2, cv2.LINE_AA)
        if bvis[f]:
            cv2.circle(img, pts[-1], 7, YELLOW, -1, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 7, (0, 0, 0), 1, cv2.LINE_AA)
            hidden = 0.0
        else:
            j = int(blast[f])
            hidden = (f - j) / FPS if j >= 0 else float("inf")
            if j >= 0 and f - j <= BALL_GHOST:
                c = (int(BX[j]), int(BY[j]))
                cv2.circle(img, c, 10, mix(YELLOW, GRASS, (f - j) / BALL_GHOST), 2, cv2.LINE_AA)
                txt(img, f"ball last seen {hidden:.1f}s ago", (c[0] + 14, c[1] - 8), 0.45, YELLOW, outline=True)
        a, b = np.searchsorted(bj_f, f - BALL_JUMP + 1), np.searchsorted(bj_f, f, "right")
        for ef, j, d, gap in bj[a:b]:
            cv2.line(img, (int(BX[j]), int(BY[j])), (int(BX[ef]), int(BY[ef])), ORANGE, 2, cv2.LINE_AA)
            txt(img, f"ball back: {d:.0f} m in {gap / FPS:.1f} s unseen", (int(BX[ef]) + 12, int(BY[ef]) + 18), 0.45, ORANGE, outline=True)

        # banner
        half = half_of(f)
        clock = f"{'1st' if half == 1 else '2nd'} half {mmss(f / FPS - CLOCK_TO_VIDEO[half])}" if half else "outside the graded halves"
        txt(img, f"Barcelona v Athletic   video {frame_to_timecode(f)}   frame {f}   {clock}", (12, 24), 0.62)
        nh, na = int(vis[i][t.team == "H"].sum()), int(vis[i][t.team == "A"].sum())
        txt(img, f"Barcelona slots tracked {nh}", (12, 52), 0.55, mix(HOME, WHITE, .35))
        txt(img, f"Athletic {na}", (290, 52), 0.55, mix(AWAY, WHITE, .5))
        txt(img, "ball tracked" if bvis[f] else f"BALL NOT TRACKED  {hidden:.1f} s", (420, 52), 0.55, GREEN if bvis[f] else RED, 1)
        if labels is not None or official is not None:
            for k, (nm, arr) in enumerate((("Kaggle-labels", labels), ("Official", official))):
                v = arr[f] if arr is not None else "X"
                s = f"{nm} {v if v != 'X' else '-'}"
                w = cv2.getTextSize(s, FONT, 0.55, 1)[0][0]
                txt(img, s, (W - 14 - w - 190 * (1 - k), 52), 0.55, mix(STATE_COL[v], WHITE, .3) if v != "X" else GREY)

        # footer: playhead and recent Kaggle events
        cx = x0 + int((f - start) / n * (x1 - x0))
        cv2.line(img, (cx, fy + 3), (cx, fy + 66), WHITE, 2)
        if events is not None:
            k = int(np.searchsorted(ev_frames, f, "right")) - 1
            shown = 0
            while k >= 0 and shown < 3 and (f - ev_frames[k]) / FPS <= 8:
                r = events.iloc[k]
                who = "" if pd.isna(r.player) else ascii_(r.player)[:28]
                txt(img, f"{(f - r.frame) / FPS:4.1f} s ago   {mmss(r.clock)}   {ascii_(r['name'])[:20]:<20} {r.team}   {who}",
                    (12, fy + 108 + 18 * shown), 0.5, mix(HOME if r.team == "H" else AWAY, WHITE, .5))
                k, shown = k - 1, shown + 1
        vw.write(img)
        if (f - start) % 2500 == 0 and f > start:
            print(f"  {name}: {(f - start) / n:4.0%}  {time.time() - t0:5.0f}s", flush=True)
    vw.release()
    os.replace(out + ".part.mp4", out)

    vanish = int((vis[:-1] & ~vis[1:])[max(0, start - lo):].sum())
    bad = sum(1 for e in tele if e[0] >= start - lo and not e[5])
    stats = (f"{name}: {os.path.getsize(out) / 1e6:.0f} MB, {time.time() - t0:.0f}s | slots lost {vanish}, comebacks that are not "
             f"physically reachable {bad} | ball tracked {bvis[start:end + 1].mean():.1%}, returned after a gap of 5+ frames {len(bj)} times")
    print(stats, flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--window", type=int, help="1-based window number from --list")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--start", type=int)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-labels", action="store_true", help="hide Kaggle-labels, Official and the event ticker")
    a = ap.parse_args()

    ws = windows(a.minutes)
    if a.list:
        for i, w in enumerate(ws, 1):
            p = w["half"]
            print(f"{i:2d}  {w['name']:<28} frames {w['start']}-{w['end']}  video {frame_to_timecode(w['start'])[:8]}-{frame_to_timecode(w['end'])[:8]}"
                  f"  clock {mmss(w['start'] / FPS - CLOCK_TO_VIDEO[p])}-{mmss(w['end'] / FPS - CLOCK_TO_VIDEO[p])}")
        return
    if a.start is not None:
        jobs = [dict(start=a.start, end=min(a.start + int(a.seconds * FPS) - 1, tracking.load().n - 1), half=half_of(a.start),
                     name=f"custom_{a.start}_{int(a.seconds)}s")]
    elif a.all:
        jobs = ws
    elif a.window:
        jobs = [ws[a.window - 1]]
    else:
        ap.error("choose --list, --window N, --all or --start FRAME")
    for j in jobs:
        j["no_labels"] = a.no_labels
    if len(jobs) == 1:
        render(jobs[0])
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            for _ in ex.map(render, jobs):
                pass


if __name__ == "__main__":
    main()
