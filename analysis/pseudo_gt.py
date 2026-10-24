"""Event-derived possession labels for the provided match: an approximate proxy for the ground truth.

Built from the third-party event log (analysis/events.py), whose use the organisers confirmed in writing
(2026-10-24). It is NOT the organisers' ground truth, which comes from corrected tracking, and event
times are only good to about a second. Use it to test methods on the real tracking, never to tune them
closely: the final evaluation is on other, unreleased matches.

Rules, set from CHALLENGE.md before looking at any agreement with the public ground truth:
  * a stoppage event (Out, Free kick, Corner, Start delay, Goal, Offside Pass, ...) makes the ball dead
  * the next restart event (Throw in, Goal kick, Pass: free kick taken / corner taken / KO, ...) makes
    it alive again, and its team has the ball
  * while alive, the owner is the team of the latest control event (Control, Pass, Take on, Cross,
    shots, goalkeeper actions, Ball recovery, ...); touches without control (Ball touch, Header,
    Aerial, Challenge, Clearance, Flick-on, ...) do not change it
Event wall-clock times are mapped to video frames by the kick-off anchors in games.py, one offset per half.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402
from analysis import events as ev_mod  # noqa: E402
from analysis import hold_owner as ho  # noqa: E402

FPS = 25
N_FRAMES = 199319
CONTROL = {"Control", "Pass", "Chipped pass", "Launch", "Cross", "Take on", "Throw in", "Goalkeeper throw",
           "GK kick from hands", "Goal kick", "Corner", "Goal", "Miss", "Miss NPGL", "Attempt saved",
           "Attempt blocked", "Post", "Collected Ball", "Catch", "Smother", "Ball recovery", "Offside Pass"}
STOP = {"Out", "Free kick", "Corner", "Start delay", "Goal", "Offside Pass", "Free kick (PEN)", "End"}
SHOTS = {"Goal", "Miss", "Miss NPGL", "Attempt saved", "Attempt blocked", "Post", "Penalty faced"}
MERGE_FRAMES = 75   # 3 s, the CHALLENGE.md proxy for controlled possession
FALLBACK_S = 2.0    # a control event this long after a stoppage restarts play even without a restart tag


def _is_restart(row):
    """Free kicks and corners are taken as Pass, Cross, Launch or Chipped pass, so go by the tag."""
    return row["name"] in {"Throw in", "Goal kick", "Drop of ball"} or any(
        k in row["extras"] for k in ("Free kick taken", "Corner taken", "KO "))


def offsets(ev):
    """Seconds to add to an event's wall time to get video seconds, per half, from the kick-off anchors."""
    g = games.GAMES[games.GAME]
    out = {}
    for p, anchor in ((1, g["kickoff"]), (2, g["kickoff_2nd"])):
        ko = ev[(ev.period == p) & (ev["name"] == "Pass") & ev.extras.str.contains("KO Start")].iloc[0]
        out[p] = games.anchor_video_frame(anchor) / FPS - ko.wall
    return out


def walk(e):
    """The labelling rules applied to one half's events, in time order. Yields one dict per event:
    the event row, the state before and after (alive, owner), what the rules did with it in `how`
    ('restart', 'restart (penalty)', 'restart (fallback)', 'control', 'stop', combinations such as
    'control+stop', or None when the event is ignored), `flip` when a live possession changed hands, and
    `state`, the label that holds from this event until the next one."""
    alive, owner, pen, stop_clock = False, None, False, -1e9
    for _, r in e.iterrows():
        n = r["name"]
        alive0, owner0, how = alive, owner, None
        if n == "Free kick (PEN)":
            pen = True
        tagged = _is_restart(r)
        penalty = pen and n in SHOTS
        fallback = n in CONTROL and r.clock - stop_clock > FALLBACK_S and n not in STOP
        if (tagged or penalty or fallback) and not alive:
            alive = True
            how = "restart" if tagged else ("restart (penalty)" if penalty else "restart (fallback)")
            pen = pen and n not in SHOTS
            if r.team in ("H", "A"):
                owner = r.team
        elif alive and n in CONTROL and r.team in ("H", "A"):
            owner = r.team
            how = "control"
        if n in STOP:
            alive, stop_clock = False, r.clock
            how = f"{how}+stop" if how else "stop"
        yield {"row": r, "alive0": alive0, "owner0": owner0, "alive": alive, "owner": owner, "how": how,
               "flip": bool(alive0 and alive and owner0 and owner != owner0),
               "state": owner if alive and owner else "D"}


def labels(shift_s=0.0, merged=False, ev=None):
    """'H' / 'A' / 'D' on the graded frames of both halves, 'X' elsewhere. shift_s moves every event in
    video time (used only to check the alignment against the public ground truth)."""
    ev = ev_mod.load_events() if ev is None else ev
    off = offsets(ev)
    lab = np.full(N_FRAMES, "X", dtype="<U1")
    for (p, a, b) in games.half_video_blocks(games.GAME):
        frames, states = [], []
        for d in walk(ev[ev.period == p]):
            frames.append(int(round((d["row"].wall + off[p] + shift_s) * FPS)))
            states.append(d["state"])
        frames = np.array(frames)
        order = np.argsort(frames, kind="stable")
        frames, states = frames[order], [states[i] for i in order]
        seg = np.full(b - a + 1, "D", dtype="<U1")
        for i, f in enumerate(frames):
            lo = max(f - a, 0)
            hi = (max(frames[i + 1] - a, 0) if i + 1 < len(frames) else b - a + 1)
            if lo < b - a + 1:
                seg[lo:min(hi, b - a + 1)] = states[i]
        lab[a:b + 1] = ho.merge_short(seg, MERGE_FRAMES, 0) if merged else seg
    return lab
