# Analysis environment

Python 3.12 and every library are declared in `../pyproject.toml`, pinned to the tested versions.
There is no lockfile, so `uv` resolves the pinned libraries' own dependencies fresh.

## Set up (macOS or Linux/WSL2, x86_64 or Apple silicon)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
git clone -b dev https://github.com/iamkarthikarun/more-than-a-hack-2027.git
cd more-than-a-hack-2027
uv sync                                              # fetches Python 3.12 and all libraries
```

## Branches

- `main` stays a clean copy of the fork's starting point.
- `dev` is the working branch for all experiments. Day-to-day: `git pull` on `dev`, commit,
  `git push`. For something risky, branch from it (`git switch -c exp/<name> dev`) and merge back
  into `dev` when it works.

On Linux the CUDA build of torch is installed, which is several GB. For a GPU you only need
the Windows NVIDIA driver; no CUDA toolkit inside WSL2. Check it:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Data

The match data is gated and is not in git. Put `tactical/` (and optionally `broadcast/`) under
`data/`, or point `MTAH_DATA_DIR` at the folder holding them. On WSL2, keep it on the Linux
filesystem (`~/...`), not under `/mnt/c`, which is much slower.

## Run

```bash
uv run python analysis/overview_plot.py    # coverage overview -> analysis/figures/overview.png
uv run python analysis/make_gif.py         # 24 s animated clip -> analysis/figures/clip.gif
```

![24 s of play: Barcelona (red), Athletic (blue), ball (yellow)](figures/clip.gif)

`analysis/tracking.py` parses the tracking file once and caches it in `data/_cache/`.
Everything in `data/` and `analysis/figures/` is git-ignored on purpose, except the example
`clip.gif` above: RULES.md forbids publishing the data or anything that reconstructs it.

## Other-match data (open DFL/Sportec, CC BY 4.0)

`analysis/dfl.py` reads other matches from https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0,
attribute the dataset paper in `analysis/fetch_dfl.py`), which carry per-frame possession and
ball-status labels the provided match lacks. `fetch_dfl.py` downloads them into
`data/external/dfl/` and checks every file against the publisher's MD5, so any machine ends up
with byte-identical copies. The positions file is 350-420 MB per match.

```bash
uv run python analysis/fetch_dfl.py                  # the two matches used so far (~740 MB)
uv run python analysis/fetch_dfl.py --list           # all 7 matches and their sizes
uv run python analysis/fetch_dfl.py --verify-only    # check what is already on disk
uv run python analysis/dfl_baseline.py               # nearest-player baseline, scored per match
```

Matches used so far: `DFL-MAT-J03WOH` and `DFL-MAT-J03WN1`.
