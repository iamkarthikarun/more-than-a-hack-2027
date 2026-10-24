"""Read the third-party event log of the provided match (Kaggle, `2445027 Barcelona-Athletic Club.xlsx`).

Standard library only, so no spreadsheet package is needed. The file is human-produced event data about
the provided match; the organisers confirmed in writing (2026-10-24) that teams may use it. It stays in
data/external/kaggle/, which `data/*` in .gitignore keeps out of git: its licence is unknown, so it
must not be redistributed.
"""

import os
import sys
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "data", "external", "kaggle", "2445027 Barcelona-Athletic Club.xlsx")
_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _col(ref):
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = n * 26 + ord(ch.upper()) - 64
    return n - 1


def read_sheets(path=DEFAULT_PATH):
    """{sheet name: DataFrame}, first row as the header, empty columns kept as '' headers."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        shared = ["".join(t.text or "" for t in si.iter(_M + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out = {}
    for sheet in ET.fromstring(z.read("xl/workbook.xml")).iter(_M + "sheet"):
        target = rels[sheet.get(_R + "id")].lstrip("/")
        root = ET.fromstring(z.read(target if target.startswith("xl/") else "xl/" + target))
        rows = []
        for row in root.iter(_M + "row"):
            cells = {}
            for c in row.findall(_M + "c"):
                v = c.find(_M + "v")
                if c.get("t") == "inlineStr":
                    val = "".join(t.text or "" for t in c.iter(_M + "t"))
                elif v is None:
                    continue
                elif c.get("t") == "s":
                    val = shared[int(v.text)]
                elif c.get("t") in ("str", "e"):
                    val = v.text
                elif c.get("t") == "b":
                    val = v.text == "1"
                else:
                    f = float(v.text)
                    val = int(f) if f.is_integer() else f
                cells[_col(c.get("r"))] = val
            rows.append(cells)
        width = max((max(r) for r in rows if r), default=-1) + 1
        table = [[r.get(i) for i in range(width)] for r in rows]
        if table:
            out[sheet.get("name")] = pd.DataFrame(table[1:], columns=[h if h is not None else "" for h in table[0]])
    return out


def load_events(path=DEFAULT_PATH):
    """Clean event table, ordered by wall clock.

    wall   seconds since midnight on match day (the Timestamp column, to the millisecond)
    clock  the match clock in seconds as given in `Time` (restarts at 45:00 for the 2nd half)
    team   'H' (Barcelona) or 'A' (Athletic Club)
    period 1 or 2, from the Start and End events; 0 before kick-off and after the final whistle
    """
    df = read_sheets(path)["Sheet"]
    df = df.loc[:, [c for c in df.columns if c]].copy()
    df["name"] = df["Event"].astype(str).str.replace(r"^\d+\s+", "", regex=True).str.strip()
    parts = df["Time"].astype(str).str.extract(r"^(\d+):(\d+\.\d+)")
    df["clock"] = parts[0].astype(float) * 60 + parts[1].astype(float)
    df["wall"] = (df["Timestamp"].astype(float) % 1.0) * 86400.0
    df["team"] = df["Team"].map({"Barcelona": "H", "Athletic Club": "A"})
    df["extras"] = df["Extras"].fillna("").astype(str)
    df = df.sort_values(["wall", "Id"], kind="stable").reset_index(drop=True)

    def edge(name, text):
        hit = df[(df["name"] == name) & df["extras"].str.contains(text)]
        return float(hit["wall"].min()) if len(hit) else np.nan

    t1, e1 = edge("Start", "First"), edge("End", "Period=First")
    t2, e2 = edge("Start", "Second"), edge("End", "Period=Second")
    df["period"] = np.select([(df["wall"] >= t1) & (df["wall"] <= e1), (df["wall"] >= t2) & (df["wall"] <= e2)], [1, 2], 0)
    return df[["name", "team", "Player", "extras", "clock", "wall", "period", "Id"]].rename(columns={"Player": "player"})


if __name__ == "__main__":
    for name, df in read_sheets(*sys.argv[1:2]).items():
        print(name, df.shape)
        print(df.head(3).to_string()[:1500])
