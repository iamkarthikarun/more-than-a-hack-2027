"""Loader for the open Sportec/DFL match data (CC BY 4.0, doi 10.6084/m9.figshare.28196177).

Used as OTHER-game material: the hackathon allows data from other matches, and this one carries
per-frame possession and ball-status labels that the provided match does not.

Layout differences from the provided tactical ATD (see analysis/tracking.py):
  - one XML <FrameSet> per person, not one text line per frame
  - coordinates in metres with the origin at the pitch centre (ours are 0-1, origin at a corner)
  - every person has a real id; the provided data has anonymous track slots
  - the ball has a height Z and BallPossession (1 home, 2 away) / BallStatus (1 alive, 0 dead)

Files live in data/external/dfl/, which `data/*` in .gitignore keeps out of git.
"""

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFL_DIR = os.path.join(ROOT, "data", "external", "dfl")
FPS = 25
PITCH = (105.0, 68.0)

_HEAD = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"(?: Z="(-?[\d.]+)")?[^>]*?'
                    r'(?: BallPossession="(\d)" BallStatus="(\d)")?/>')


@dataclass
class Half:
    name: str
    n0: int                 # first frame number of this half in the source file
    xy: np.ndarray          # (T, P, 2) metres, NaN when not tracked
    team: np.ndarray        # (P,) 'H' or 'A'
    person: np.ndarray      # (P,) source person ids (real identities; the hackathon data has none)
    ball: np.ndarray        # (T, 3) x, y, z metres, NaN when not tracked
    possession: np.ndarray  # (T,) 1 home, 2 away, 0 missing
    status: np.ndarray      # (T,) 1 alive, 0 dead, -1 missing

    @property
    def n(self):
        return self.ball.shape[0]

    def labels(self):
        """H / A / D per frame, built the way the challenge describes its own ground truth."""
        out = np.full(self.n, "X", dtype="<U1")
        out[self.status == 0] = "D"
        alive = self.status == 1
        out[alive & (self.possession == 1)] = "H"
        out[alive & (self.possession == 2)] = "A"
        return out


def _teams(match_id):
    path = next(os.path.join(DFL_DIR, f) for f in os.listdir(DFL_DIR)
                if f.startswith("DFL_02_01_matchinformation") and match_id in f)
    root = ET.parse(path).getroot()
    return {t.get("TeamId"): ("H" if t.get("Role") == "home" else "A") for t in root.iter("Team")}


def load(match_id="DFL-MAT-J03WOH"):
    """{'firstHalf': Half, 'secondHalf': Half}, indexed from the half's first frame."""
    path = next(os.path.join(DFL_DIR, f) for f in os.listdir(DFL_DIR)
                if f.startswith("DFL_04_03_positions_raw_observed") and match_id in f)
    side = _teams(match_id)
    sets = {}   # (section, person) -> (team, {N: (x, y, z, m, poss, status)})
    cur = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                section, team, person = _HEAD.search(line).groups()
                if team == "referee":   # present in some matches; belongs to neither side
                    cur = None
                    continue
                cur = {}
                sets[(section, person)] = ("BALL" if team == "BALL" else side[team], cur)
            elif line.startswith("<Frame ") and cur is not None:
                m = _FRAME.search(line)
                n, x, y, z, poss, stat = m.groups()
                cur[int(n)] = (float(x), float(y), float(z) if z else np.nan,
                               int(poss) if poss else 0, int(stat) if stat else -1)

    halves = {}
    for section in ("firstHalf", "secondHalf"):
        ball = next(d for (s, _), (t, d) in sets.items() if s == section and t == "BALL")
        ns = np.array(sorted(ball))
        n0, T = int(ns[0]), int(ns[-1] - ns[0] + 1)
        people = [(p, t, d) for (s, p), (t, d) in sets.items() if s == section and t != "BALL"]
        xy = np.full((T, len(people), 2), np.nan, np.float32)
        for j, (_, _, d) in enumerate(people):
            for n, v in d.items():
                if 0 <= n - n0 < T:
                    xy[n - n0, j] = v[:2]
        b = np.full((T, 3), np.nan, np.float32)
        poss = np.zeros(T, np.int8); stat = np.full(T, -1, np.int8)
        for n, v in ball.items():
            i = n - n0
            b[i] = v[:3]; poss[i] = v[3]; stat[i] = v[4]
        halves[section] = Half(section, n0, xy, np.array([t for _, t, _ in people]),
                               np.array([p for p, _, _ in people]), b, poss, stat)
    return halves
