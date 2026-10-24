"""Convert between frame ids, tactical-video timecodes, event-log clocks and Smart Tagging seconds.

    uv run python analysis/video_time.py frame 18009          # what is at this frame?
    uv run python analysis/video_time.py time 00:12:00.360    # which frame is this timecode?
    uv run python analysis/video_time.py clock 1 273.9        # 1st-half match clock 273.9 s
    uv run python analysis/video_time.py events "Goal kick"   # every such event with its timecode
    uv run python analysis/video_time.py around 18009 --seconds 8
    uv run python analysis/video_time.py timeline             # where everything sits in the feed

The tactical video is one continuous recording at 25 fps, from its first frame (frame 0, 00:00:00.000)
through pre-match, both halves, drinks breaks, halftime and post-match. Frame i is at i / 25 seconds.
That is also the submission row, ATD frame_id - 1, and Smart Tagging `round(seconds * 25)`.

Event-log clock to video time (constants verified on the provided match: the log's match clock is an
exact straight shift of wall time inside each half, drinks breaks included, and the two halves' offsets
agree with the organisers' kick-off anchors to 0.15 s):
    1st half  video seconds = match clock + 446.444
    2nd half  video seconds = match clock + 1849.601     (the clock restarts at 45:00.000)
Event times are only good to about +-1 s: they are operator-logged. The video file itself may carry a
constant start offset; check by seeking to 00:07:27.440 (1st-half kick-off) and 01:15:49.640 (2nd half).
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402

FPS = 25
CLOCK_TO_VIDEO = {1: 446.444, 2: 1849.601}


def frame_to_timecode(frame):
    ms = int(frame) * 40                       # 1000 / 25 is exactly 40 ms
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def timecode_to_seconds(text):
    """'HH:MM:SS.mmm', 'MM:SS.mmm' or plain seconds."""
    parts = [float(p) for p in re.split(":", text.strip())]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def seconds_to_frame(seconds):
    return int(round(seconds * FPS))


def half_of(frame):
    for p, a, b in games.half_video_blocks(games.GAME):
        if a <= frame <= b:
            return p
    return 0


def where(frame):
    n = games.GAMES[games.GAME]["n_video_frames"]
    blocks = games.half_video_blocks(games.GAME)
    if frame < blocks[0][1]:
        return "before the graded 1st half (pre-match)"
    if frame <= blocks[0][2]:
        return "graded 1st half"
    if frame < blocks[1][1]:
        return "between the halves (halftime)"
    if frame <= blocks[1][2]:
        return "graded 2nd half"
    return "after the graded 2nd half (post-match)" if frame < n else "past the end of the video"


def describe(frame):
    p = half_of(frame)
    lines = [f"video frame {frame}  (submission row {frame}, ATD frame_id {frame + 1})",
             f"timecode    {frame_to_timecode(frame)}   = {frame / FPS:.3f} s  = Smart Tagging second {frame / FPS:.2f}",
             f"where       {where(frame)}"]
    if p:
        lines.append(f"match clock {frame / FPS - CLOCK_TO_VIDEO[p]:.1f} s  ({p}{'st' if p == 1 else 'nd'}-half clock)")
    return lines


def _events():
    from analysis import events as ev_mod
    return ev_mod.load_events()


def _event_frame(row):
    return seconds_to_frame(row.clock + CLOCK_TO_VIDEO[row.period])


def _print_events(rows):
    print(f"{'half':>4} {'clock':>9} {'timecode':>13} {'frame':>7}  {'team':4} {'event':22} extras")
    for _, r in rows.iterrows():
        f = _event_frame(r)
        print(f"{int(r.period):>4} {r.clock:8.1f}s {frame_to_timecode(f):>13} {f:7d}  {str(r.team):4} {r['name'][:22]:22} {r.extras[:50]}")


def timeline():
    n = games.GAMES[games.GAME]["n_video_frames"]
    (_, a1, b1), (_, a2, b2) = games.half_video_blocks(games.GAME)
    g = games.GAMES[games.GAME]
    rows = [("video starts", 0), ("graded 1st half starts", a1),
            ("1st-half kick-off (ball struck)", games.anchor_video_frame(g["kickoff"])),
            ("graded 1st half ends", b1), ("graded 2nd half starts", a2),
            ("2nd-half kick-off (ball struck)", games.anchor_video_frame(g["kickoff_2nd"])),
            ("graded 2nd half ends", b2), ("video ends", n - 1)]
    for name, f in rows:
        print(f"  {frame_to_timecode(f)}  frame {f:6d}  {name}")
    print("event log, from the match clock:")
    for p, label in ((1, "1st-half whistle (End event, clock 3101.686)"), (2, "2nd-half whistle (End event, clock 5758.152)")):
        c = 3101.686 if p == 1 else 5758.152
        f = seconds_to_frame(c + CLOCK_TO_VIDEO[p])
        print(f"  {frame_to_timecode(f)}  frame {f:6d}  {label}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("frame").add_argument("frame", type=int)
    sub.add_parser("time").add_argument("timecode")
    c = sub.add_parser("clock"); c.add_argument("half", type=int, choices=(1, 2)); c.add_argument("seconds", type=float)
    e = sub.add_parser("events"); e.add_argument("name"); e.add_argument("--team", choices=("H", "A")); e.add_argument("--limit", type=int, default=40)
    a = sub.add_parser("around"); a.add_argument("frame", type=int); a.add_argument("--seconds", type=float, default=8.0)
    sub.add_parser("timeline")
    args = ap.parse_args()

    if args.cmd == "frame":
        print("\n".join(describe(args.frame)))
    elif args.cmd == "time":
        print("\n".join(describe(seconds_to_frame(timecode_to_seconds(args.timecode)))))
    elif args.cmd == "clock":
        print("\n".join(describe(seconds_to_frame(args.seconds + CLOCK_TO_VIDEO[args.half]))))
    elif args.cmd == "events":
        ev = _events()
        ev = ev[(ev.period > 0) & (ev["name"].str.lower() == args.name.lower())]
        if args.team:
            ev = ev[ev.team == args.team]
        print(f"{len(ev)} '{args.name}' events" + (f" by {args.team}" if args.team else "") + f" (showing {min(len(ev), args.limit)})")
        _print_events(ev.head(args.limit))
    elif args.cmd == "around":
        ev = _events()
        ev = ev[ev.period > 0]
        t = args.frame / FPS
        near = ev[[abs(_event_frame(r) / FPS - t) <= args.seconds for _, r in ev.iterrows()]]
        print("\n".join(describe(args.frame)))
        print(f"\nevents within +-{args.seconds:g} s ({len(near)}):")
        _print_events(near)
    else:
        timeline()


if __name__ == "__main__":
    main()
