"""Download the open DFL/Sportec match files that analysis/dfl.py reads, checked against MD5.

    uv run python analysis/fetch_dfl.py                      # the two matches used so far
    uv run python analysis/fetch_dfl.py DFL-MAT-J03WPY       # any other match in the dataset
    uv run python analysis/fetch_dfl.py --list               # every match with its download size
    uv run python analysis/fetch_dfl.py --verify-only        # check files already on disk

Source: https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0). Attribute the dataset paper:
"An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data, 2025,
https://www.nature.com/articles/s41597-025-04505-y

Files go to data/external/dfl/, which `data/*` in .gitignore keeps out of git. A file already on
disk with the right MD5 is skipped and a partial one resumes. Downloads use ranged requests
because some corporate proxies stall whole-file requests at 0 bytes. Behind a TLS-inspecting
proxy, point SSL_CERT_FILE at the CA bundle.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "dfl")
DEFAULT_MATCHES = ("DFL-MAT-J03WOH", "DFL-MAT-J03WN1")
KINDS = ("matchinformation", "events_raw", "positions_raw_observed")


def listing():
    with urllib.request.urlopen(ARTICLE, timeout=60) as r:
        return json.load(r)["files"]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ranged(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.status, r.read()


def fetch(entry, dest, chunk_mb, verify_only):
    name, size, url = entry["name"], entry["size"], entry["download_url"]
    out = os.path.join(dest, name)
    have = os.path.getsize(out) if os.path.exists(out) else 0
    if have == size and md5_of(out) == entry["computed_md5"]:
        print(f"ok       {name}")
        return True
    if verify_only:
        print(f"MISSING  {name}" if not have else f"BAD      {name}")
        return False
    if have >= size:
        os.remove(out)
        have = 0
    step = chunk_mb << 20
    t0, pos = time.time(), have
    with open(out, "ab") as fh:
        while pos < size:
            end = min(pos + step, size) - 1
            for attempt in range(6):
                try:
                    status, body = ranged(url, pos, end)
                    if status in (200, 206) and len(body) == end - pos + 1:
                        break
                except OSError:
                    pass
                time.sleep(2 + attempt)
            else:
                print(f"FAILED   {name} at byte {pos}; re-run to resume")
                return False
            fh.write(body)
            fh.flush()
            pos = end + 1
            if size > 5e6:
                print(f"  {pos / size:6.1%} {pos / 1e6:7.1f} MB {time.time() - t0:5.0f}s  {name[:40]}", flush=True)
    good = md5_of(out) == entry["computed_md5"]
    print(f"{'ok' if good else 'BAD MD5':8} {name}")
    return good


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("matches", nargs="*", default=list(DEFAULT_MATCHES), help="e.g. DFL-MAT-J03WOH")
    p.add_argument("--dir", default=DEST)
    p.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    p.add_argument("--chunk-mb", type=int, default=16)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    files = listing()
    if a.list:
        sizes = {}
        for f in files:
            m = re.search(r"DFL-MAT-\w+", f["name"]).group(0)
            sizes[m] = sizes.get(m, 0) + f["size"]
        for m, s in sorted(sizes.items()):
            print(f"{m}  {s / 1e6:7.0f} MB")
        return 0

    os.makedirs(a.dir, exist_ok=True)
    ok = True
    for match in a.matches:
        chosen = [f for f in files if match in f["name"] and any(k in f["name"] for k in a.kinds)]
        if not chosen:
            print(f"no files for {match}; try --list")
            ok = False
        for f in sorted(chosen, key=lambda f: f["size"]):
            ok &= fetch(f, a.dir, a.chunk_mb, a.verify_only)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
