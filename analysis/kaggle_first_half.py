"""Descriptive analysis of the Kaggle event log's FIRST HALF. Analysis only: nothing here is a method.

    uv run python analysis/kaggle_first_half.py possession    # what counts as possession
    uv run python analysis/kaggle_first_half.py dead          # what counts as dead ball
    uv run python analysis/kaggle_first_half.py gaps          # every missing-ball stretch, and what the log says around it

Names: "Official" is the organisers' data, "Kaggle" the third-party event log, "Kaggle-labels" the H/A/D
frame labels derived from it by analysis/pseudo_gt.py. Only the first half is read, so the second half stays
untouched for later. Kaggle times are operator-logged, good to about +-1 s.
"""

import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402
from analysis import pseudo_gt, tracking  # noqa: E402
from analysis.events import load_events  # noqa: E402
from analysis.video_time import frame_to_timecode  # noqa: E402

FPS = 25
PASSES = {"Pass", "Chipped pass", "Launch", "Cross"}
NEAR = 2 * FPS   # +-2 s, the timing noise of the log
CSV = os.path.join(ROOT, "data", "_cache", "kaggle_first_half_gaps.csv")


def load_half():
    ev = load_events()
    off = pseudo_gt.offsets(ev)[1]
    steps = []
    for d in pseudo_gt.walk(ev[ev.period == 1]):
        r = d["row"]
        d.update(frame=int(round((r.wall + off) * FPS)), clock=r.clock, name=r["name"], team=r.team,
                 extras=r.extras, xlsx=int(r.xlsx_row))
        steps.append(d)
    _, a, b = games.half_video_blocks(games.GAME)[0]
    t = tracking.load()
    ball = t.ball[a:b + 1, :2] * np.array(tracking.PITCH, np.float32)
    return steps, a, b, pseudo_gt.labels()[a:b + 1], ~np.isnan(ball[:, 0]), ball


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def role(name):
    r = [x for x, s in (("control", pseudo_gt.CONTROL), ("stop", pseudo_gt.STOP),
                        ("restart", {"Throw in", "Goal kick", "Drop of ball"})) if name in s]
    return "+".join(r) or "ignored"


# ------------------------------------------------------------------------------------------ possession
def spells_of(steps, b):
    spells, cur = [], None

    def close(i, d_end):
        nonlocal cur
        cur.update(end_frame=d_end["frame"], end_name=d_end["name"], end_how=d_end["how"],
                   end_kind="dead ball" if d_end["state"] == "D" else "turnover")
        spells.append(cur)
        cur = None

    for i, d in enumerate(steps):
        st = d["state"]
        if st in ("H", "A"):
            if cur is None or cur["team"] != st:
                if cur is not None:
                    close(i, d)
                cur = dict(team=st, start_frame=d["frame"], start_name=d["name"], start_how=d["how"], n_control=0, n_pass=0)
            if d["how"] in ("control", "restart") or (d["how"] or "").startswith("restart"):
                cur["n_control"] += 1
                cur["n_pass"] += d["name"] in PASSES
        elif cur is not None:
            close(i, d)
    if cur is not None:
        cur.update(end_frame=b + 1, end_name="(end of half)", end_how=None, end_kind="end of half")
        spells.append(cur)
    return spells


def part_possession():
    steps, a, b, lab, vis, ball = load_half()
    print(f"KAGGLE FIRST HALF: {len(steps)} events between {frame_to_timecode(steps[0]['frame'])} and "
          f"{frame_to_timecode(steps[-1]['frame'])}\n")

    print("1. Which events the rules treat as carrying possession (control), as a stoppage, a restart, or ignoring")
    c = Counter(d["name"] for d in steps)
    flips = Counter(d["name"] for d in steps if d["flip"])
    restarts = Counter(d["name"] for d in steps if (d["how"] or "").startswith("restart"))
    print(f"   {'event':22}{'n':>5}  {'role':16}{'used as a restart':>18}{'changed possession':>20}")
    for name, n in c.most_common(28):
        print(f"   {name:22}{n:5d}  {role(name):16}{restarts[name]:18d}{flips[name]:20d}")
    rest = sum(n for _, n in c.most_common()[28:])
    print(f"   ... {len(c) - 28} more event types, {rest} events in all, mostly rare (cards, substitutions, goalkeeper actions)\n")

    sp = spells_of(steps, b)
    dur = np.array([(s["end_frame"] - s["start_frame"]) / FPS for s in sp])
    print(f"2. Possession spells in the first half (Kaggle-labels): {len(sp)}")
    print(f"   by team: Barcelona {sum(s['team'] == 'H' for s in sp)}, Athletic {sum(s['team'] == 'A' for s in sp)}")
    print(f"   duration: mean {dur.mean():.1f} s | median {np.median(dur):.1f} | p10 {pct(dur, 10):.1f} | p90 {pct(dur, 90):.1f} | max {dur.max():.0f}")
    print(f"   controlled actions per spell: median {np.median([s['n_control'] for s in sp]):.0f}, "
          f"passes per spell: median {np.median([s['n_pass'] for s in sp]):.0f}, "
          f"spells with a single controlled action: {sum(s['n_control'] <= 1 for s in sp)}")
    print(f"   spells shorter than 3 s: {int((dur < 3).sum())}, shorter than 1 s: {int((dur < 1).sum())}\n")

    print("3. How a spell starts and how it ends")
    start_kind = Counter("restart after a stoppage" if (s["start_how"] or "").startswith("restart") else "turnover while play was live" for s in sp)
    print("   starts:", dict(start_kind))
    print("   first event of a spell, by type:", Counter(s["start_name"] for s in sp).most_common(8))
    print("   ends:  ", dict(Counter(s["end_kind"] for s in sp)))
    print("   event that ends a spell, by type:", Counter(s["end_name"] for s in sp).most_common(10), "\n")

    ign = Counter(d["name"] for d in steps if d["how"] is None and d["alive0"] and d["alive"]
                  and d["team"] in ("H", "A") and d["team"] != d["owner0"])
    print("4. Touches by the team WITHOUT the ball that did not change possession (ignored by the rules)")
    print(f"   {sum(ign.values())} events: " + ", ".join(f"{k} {v}" for k, v in ign.most_common(10)))
    own = Counter(d["name"] for d in steps if d["how"] is None and d["alive0"] and d["alive"] and d["team"] == d["owner0"])
    print(f"   (and {sum(own.values())} ignored events by the team that already had it: " + ", ".join(f"{k} {v}" for k, v in own.most_common(5)) + ")")


# ------------------------------------------------------------------------------------------ dead ball
REASONS = ("Drinks Break", "Injury", "Referee delay", "Awaiting officials decision", "Opposing player", "Other reason")


def subtype(d):
    n, x = d["name"], d["extras"]
    if n == "Out":
        m = re.search(r"Out for [a-z\- ]+", x)
        return m.group(0).strip() if m else "Out"
    if n == "Free kick":
        return "free kick (foul)" if "Foul" in x else "free kick"
    if n == "Start delay":
        return "delay: " + next((r for r in REASONS if r in x), "other")
    return {"Corner": "corner awarded", "Offside Pass": "offside", "Goal": "goal", "End": "end of half"}.get(n, n)


def restart_kind(d):
    x, n = d["extras"], d["name"]
    if "Free kick taken" in x:
        return f"free kick taken ({n})"
    if "Corner taken" in x:
        return f"corner taken ({n})"
    if "KO " in x:
        return "kick-off"
    if n in ("Throw in", "Goal kick", "Drop of ball"):
        return n.lower()
    return f"{d['how']} ({n})"


def dead_stretches(steps):
    out, cur = [], None
    for i, d in enumerate(steps):
        if cur is None and d["alive0"] and not d["alive"]:
            cur = dict(i=i, start=d, inside=[])
        elif cur is not None:
            if d["alive"] and not d["alive0"]:
                cur.update(end=d, seconds=(d["frame"] - cur["start"]["frame"]) / FPS)
                out.append(cur)
                cur = None
            elif not (d["name"] in pseudo_gt.STOP and d["clock"] == cur["start"]["clock"]):
                cur["inside"].append(d["name"])
    if cur is not None:
        cur.update(end=None, seconds=float("nan"))
        out.append(cur)
    return out


def part_dead():
    steps, a, b, lab, vis, ball = load_half()
    ds = dead_stretches(steps)
    done = [s for s in ds if s["end"] is not None]
    total = sum(s["seconds"] for s in done)
    print(f"KAGGLE FIRST HALF: {len(ds)} stoppages ({len(done)} with a logged restart). Kaggle-labels call "
          f"{np.mean(lab == 'D'):.1%} of the graded first half dead.\n")

    print("1. What starts a dead-ball stretch, and how long it lasts (stoppage event to restart event, seconds)")
    by = {}
    for s in done:
        by.setdefault(subtype(s["start"]), []).append(s)
    print(f"   {'stoppage':32}{'n':>3}{'total s':>9}{'median':>8}{'p90':>7}{'max':>7}   most common restart")
    for k, v in sorted(by.items(), key=lambda kv: -sum(x["seconds"] for x in kv[1])):
        sec = np.array([x["seconds"] for x in v])
        rs = Counter(restart_kind(x["end"]) for x in v).most_common(2)
        print(f"   {k:32}{len(v):3d}{sec.sum():9.0f}{np.median(sec):8.1f}{pct(sec, 90):7.1f}{sec.max():7.0f}   " + "; ".join(f"{r} {n}" for r, n in rs))
    print(f"   all: {total:.0f} s of {(b - a + 1) / FPS:.0f} s = {total * FPS / (b - a + 1):.1%} of the half")

    print("\n2. What the log records INSIDE a stoppage (events between the stoppage and the restart)")
    ins = Counter(n for s in done for n in s["inside"])
    print("   " + ", ".join(f"{k} {v}" for k, v in ins.most_common(12)))

    print("\n3. How play is judged to restart")
    print("   ", dict(Counter(restart_kind(s["end"]) for s in done)))
    fb = [s for s in done if (s["end"]["how"] or "").startswith("restart (fallback")]
    print(f"   {len(fb)} restarts are only inferred (a control event 2+ s after the stoppage, no restart tag):")
    for s in fb[:6]:
        print(f"      {subtype(s['start']):22} at {frame_to_timecode(s['start']['frame'])} -> {s['end']['name']} after {s['seconds']:.1f} s")

    print("\n4. The longest stoppages")
    for s in sorted(done, key=lambda x: -x["seconds"])[:6]:
        print(f"   {s['seconds']:6.0f} s  {frame_to_timecode(s['start']['frame'])}  {subtype(s['start']):26} -> {restart_kind(s['end']):28} "
              f"inside: {', '.join(sorted(set(s['inside']))[:5]) or '-'}")


# ------------------------------------------------------------------------------------------ gaps
def hidden_runs(vis):
    d = np.diff(np.r_[1, vis.astype(int), 1])
    return list(zip(np.flatnonzero(d == -1), np.flatnonzero(d == 1)))


def boundary_distance(p):
    return float(min(p[0], tracking.PITCH[0] - p[0], p[1], tracking.PITCH[1] - p[1]))


def gap_class(seg):
    if (seg == "D").all():
        return "inside dead ball"
    if not (seg == "D").any():
        return "inside live play"
    if seg[0] != "D" and seg[-1] == "D":
        return "live -> dead (ball lost as play stops)"
    if seg[0] == "D" and seg[-1] != "D":
        return "dead -> live (ball back as play restarts)"
    return "mixed (a stoppage inside the gap)"


def part_gaps():
    steps, a, b, lab, vis, ball = load_half()
    frames = np.array([d["frame"] for d in steps])
    runs = hidden_runs(vis)
    rows = []
    for s, e in runs:
        sf, ef = a + s, a + e
        seg = lab[s:e]
        before = np.searchsorted(frames, sf, side="left") - 1
        after = np.searchsorted(frames, ef, side="left")
        inside = [d for d in steps if sf <= d["frame"] < ef]
        near_on = [d["name"] for d in steps if abs(d["frame"] - sf) <= NEAR]
        near_off = [d["name"] for d in steps if abs(d["frame"] - ef) <= NEAR]
        cls = gap_class(seg)
        rows.append(dict(
            gap=len(rows) + 1, start_frame=sf, end_frame=ef, start_timecode=frame_to_timecode(sf), seconds=(e - s) / FPS,
            censored=bool(s == 0 or e == len(vis)), cls=cls,
            frac_D=float(np.mean(seg == "D")), frac_H=float(np.mean(seg == "H")), frac_A=float(np.mean(seg == "A")),
            label_before=lab[s - 1] if s > 0 else "", label_after=lab[e] if e < len(lab) else "",
            last_event_before=steps[before]["name"] if before >= 0 else "", last_before_team=steps[before]["team"] if before >= 0 else "",
            dt_before_s=(sf - frames[before]) / FPS if before >= 0 else np.nan,
            first_event_after=steps[after]["name"] if after < len(steps) else "", first_after_team=steps[after]["team"] if after < len(steps) else "",
            dt_after_s=(frames[after] - ef) / FPS if after < len(steps) else np.nan,
            events_inside=len(inside), inside_names="; ".join(sorted(set(d["name"] for d in inside)))[:80],
            near_onset="; ".join(sorted(set(near_on)))[:80], near_offset="; ".join(sorted(set(near_off)))[:80],
            stop_near_onset=any("stop" in (d["how"] or "") for d in steps if abs(d["frame"] - sf) <= NEAR),
            restart_near_offset=any((d["how"] or "").startswith("restart") for d in steps if abs(d["frame"] - ef) <= NEAR),
            onset_boundary_m=boundary_distance(ball[s - 1]) if s > 0 and vis[s - 1] else np.nan,
            offset_boundary_m=boundary_distance(ball[e]) if e < len(vis) and vis[e] else np.nan))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    df.to_csv(CSV, index=False)

    hid = (~vis).sum()
    print(f"OFFICIAL FIRST HALF (frames {a}-{b}): ball hidden on {hid / len(vis):.1%} of frames in {len(df)} gaps "
          f"({int(df.censored.sum())} touch the edge of the half). Full table: data/_cache/kaggle_first_half_gaps.csv\n")
    sec = df.seconds.values
    print(f"1. Gap length: median {np.median(sec):.1f} s, mean {sec.mean():.1f}, p10 {pct(sec, 10):.1f}, p90 {pct(sec, 90):.1f}, max {sec.max():.0f}\n")

    print("2. What Kaggle says was happening during each gap (Kaggle-labels over the gap's frames)")
    print(f"   {'class':46}{'gaps':>5}{'hidden s':>10}{'share of hidden time':>22}")
    tot = sec.sum()
    for cls, g in sorted(df.groupby("cls"), key=lambda kv: -kv[1].seconds.sum()):
        print(f"   {cls:46}{len(g):5d}{g.seconds.sum():10.0f}{g.seconds.sum() / tot:22.1%}")
    print(f"   frames hidden while Kaggle-labels say dead ball: {np.mean(lab[~vis] == 'D'):.1%}; Barcelona: {np.mean(lab[~vis] == 'H'):.1%}; "
          f"Athletic: {np.mean(lab[~vis] == 'A'):.1%}\n")

    ok = df[~df.censored]
    print("3. Before and after (non-edge gaps), Kaggle events nearest each side of the gap")
    print("   last event before the gap:      " + ", ".join(f"{k} {v}" for k, v in Counter(ok.last_event_before).most_common(8)))
    print(f"      time from it to the gap start (s): median {ok.dt_before_s.median():.1f}, p90 {ok.dt_before_s.quantile(.9):.1f}")
    print("   first event after the gap:      " + ", ".join(f"{k} {v}" for k, v in Counter(ok.first_event_after).most_common(8)))
    print(f"      time from the gap end to it (s):   median {ok.dt_after_s.median():.1f}, p90 {ok.dt_after_s.quantile(.9):.1f}")

    base_edge = np.mean([boundary_distance(p) < 4 for p in ball[vis][::25]])
    print(f"   gap starts within +-2 s of a stoppage event: {ok.stop_near_onset.mean():.0%}; "
          f"gap ends within +-2 s of a real restart (restart tag, throw-in, goal kick, penalty): {ok.restart_near_offset.mean():.0%}")
    print(f"   ball at its last visible frame before the gap: within 4 m of the pitch edge in "
          f"{np.mean(ok.onset_boundary_m.dropna() < 4):.0%} of gaps; first visible frame after: {np.mean(ok.offset_boundary_m.dropna() < 4):.0%}"
          f"   (for comparison, {base_edge:.0%} of all visible-ball frames)")
    live = df[df.cls == "inside live play"]
    print(f"   the {len(live)} gaps inside live play: median {live.seconds.median():.1f} s, "
          f"{int((live.seconds < 5).sum())} shorter than 5 s, {int((live.seconds > 15).sum())} longer than 15 s; "
          f"hidden live time by team: Barcelona {(live.seconds * live.frac_H).sum():.0f} s, Athletic {(live.seconds * live.frac_A).sum():.0f} s")

    print("\n4. Examples")
    shown = set()
    for cls in ("inside dead ball", "live -> dead (ball lost as play stops)", "dead -> live (ball back as play restarts)", "inside live play", "mixed (a stoppage inside the gap)"):
        g = ok[ok.cls == cls]
        if not len(g):
            continue
        for label, row in (("a typical one", g.iloc[(g.seconds - g.seconds.median()).abs().argsort().iloc[0]]),
                           ("the longest", g.loc[g.seconds.idxmax()])):
            if row.gap in shown:
                continue
            shown.add(row.gap)
            print(f"\n   [{cls}] {label}: gap #{int(row.gap)}, {row.start_timecode}, hidden {row.seconds:.1f} s, "
                  f"Kaggle-labels dead {row.frac_D:.0%} / Barcelona {row.frac_H:.0%} / Athletic {row.frac_A:.0%}")
            for d in steps:
                if row.start_frame - 5 * FPS <= d["frame"] <= row.end_frame + 5 * FPS:
                    mark = "   <- ball hidden from here" if abs(d["frame"] - row.start_frame) < 1 else ""
                    where = "before" if d["frame"] < row.start_frame else ("inside" if d["frame"] < row.end_frame else "after")
                    print(f"      {where:6} {(d['frame'] - row.start_frame) / FPS:+6.1f} s  {str(d['team']):1} {d['name'][:20]:20} {d['extras'][:34]}{mark}")
            print(f"      ... ball visible again at {frame_to_timecode(int(row.end_frame))}")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "possession"
    {"possession": part_possession, "dead": part_dead, "gaps": part_gaps}[part]()
