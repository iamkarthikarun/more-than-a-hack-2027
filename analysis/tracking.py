"""Fast loader for the tactical ATD, without kloppy.

Raw line: "<frame_id>:<slot1>;...;<slot46>:<ball>", each object "x,y,speed", NaN when unseen.
Coordinates are normalised 0-1. Slots 1-24 are Barcelona (H), 25-46 Athletic (A); slot ids are
tracker slots, not people.

The parsed arrays are cached under data/_cache/, which `data/*` in .gitignore keeps out of git.
Derived frame-level tracking must not be published (RULES.md), so never move the cache elsewhere.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths, DATA_DIR, half_video_blocks  # noqa: E402

FPS = 25
N_SLOTS = 46
PITCH = (105.3, 68.0)
CACHE = os.path.join(DATA_DIR, "_cache", "tactical_atd.npz")


@dataclass
class Tracking:
    xy: np.ndarray        # (N, 46, 2) float32, normalised
    speed: np.ndarray     # (N, 46) float32, units to be confirmed (see DATA.md vs ATD XML)
    ball: np.ndarray      # (N, 3) float32: x, y, speed
    team: np.ndarray      # (46,) 'H' or 'A'

    @property
    def n(self):
        return self.xy.shape[0]

    def metres(self, a):
        """Scale a normalised (..., 2) array to metres."""
        return a * np.array(PITCH, dtype=np.float32)


def _teams(meta_path):
    root = ET.parse(meta_path).getroot()
    names = {t.get("id"): t.find("Name").text for t in root.iter("Team")}
    home = next(i for i, n in names.items() if n == "Barcelona")
    return np.array(["H" if p.get("teamId") == home else "A" for p in root.iter("Player")])


def _parse(raw_path):
    with open(raw_path) as f:
        text = f.read().replace(":", ",").replace(";", ",")
    df = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32)
    ids = df.iloc[:, 0].to_numpy()
    if not (ids[0] == 1 and np.all(np.diff(ids) == 1)):
        raise ValueError("frame ids are not 1..N contiguous")
    v = df.iloc[:, 1:].to_numpy()
    players = v[:, : N_SLOTS * 3].reshape(-1, N_SLOTS, 3)
    return players[..., :2], players[..., 2], v[:, N_SLOTS * 3:]


def load(refresh=False):
    """Tracking indexed by video frame: row i is video frame i (ATD frame_id = i + 1)."""
    paths = asset_paths()
    team = _teams(paths["atd_meta"])
    if not refresh and os.path.exists(CACHE):
        z = np.load(CACHE)
        return Tracking(z["xy"], z["speed"], z["ball"], team)
    xy, speed, ball = _parse(paths["atd_raw"])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, xy=xy, speed=speed, ball=ball)
    return Tracking(xy, speed, ball, team)


def segments():
    """(name, first_frame, last_frame) for every block of the match, graded or not."""
    (_, h1a, h1b), (_, h2a, h2b) = half_video_blocks("BAR-ATH")
    return [("pre-match", 0, h1a - 1), ("1st half", h1a, h1b), ("halftime", h1b + 1, h2a - 1),
            ("2nd half", h2a, h2b), ("post-match", h2b + 1, 199318)]


def graded_mask(n=199319):
    m = np.zeros(n, bool)
    for _, a, b in half_video_blocks("BAR-ATH"):
        m[a:b + 1] = True
    return m
