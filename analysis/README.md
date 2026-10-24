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
uv run python analysis/fetch_dfl.py                  # two matches (~740 MB); name others to add them
uv run python analysis/fetch_dfl.py --list           # all 7 matches and their sizes
uv run python analysis/fetch_dfl.py --verify-only    # check what is already on disk
uv run python analysis/dfl_baseline.py               # nearest-player baseline on every downloaded match
```

A fresh machine needs the download step above. Each match scores the same whichever subset is
downloaded, because its hidden-ball gap pattern is fixed per match.

Reference, spellF1 with the ball hidden at the +-50 frame window (from a clean-clone run on a second
machine, with J03WN1 and J03WOH also reproduced exactly on the first):
`J03WMX 0.232, J03WN1 0.207, J03WOH 0.220, J03WOY 0.174, J03WPY 0.225, J03WQQ 0.241, J03WR9 0.188`,
mean 0.212.

## Making DFL look like the provided match

DFL is clean (ball always tracked, all 22 players, raw 1 cm-quantised positions); the provided tactical
tracking is not. `analysis/degrade.py` learns the difference from the provided match's tracking alone
(no labels) and applies it to DFL; `analysis/degrade_eval.py` fits it, checks the result and scores the
baseline on it.

```bash
uv run python analysis/degrade_eval.py                                          # every downloaded match
uv run python analysis/degrade_eval.py --matches DFL-MAT-J03WOH DFL-MAT-J03WN1  # exact subset
```

What was measured on the provided match:
- **Ball:** hidden on 41% of in-play frames in ~150 gaps (median 10.7 s). It vanishes when it is
  *slow* (median 2.0 m/s against 5.2), close to a player, or near a touchline (25% of onsets within
  4 m of one, against 13%), not when it is fast. Only 149 onsets, so the covariate model is weak
  (cross-half AUC 0.64 and 0.74); a constant-hazard control is kept next to it.
- **Players:** a track ends mostly under occlusion (62% within 2 m of another player, against 12%),
  at low speed or near a touchline (cross-half AUC ~0.87). All 11 of a team are tracked on only
  ~55-60% of frames.
- **Positions** are far smoother than DFL's raw ones (velocity autocorrelation 0.999 against 0.879),
  so DFL is smoothed (Gaussian, 2 frames), not made noisier.
- **Not modelled:** 2-4 m single-step ball jumps (0.19% of ball frames) and extra phantom tracks that
  give a team a 12th or 13th player on ~5-8% of frames. The 3+ players-missing tail is under-produced
  (about half of the provided 8-10%).

Result, nearest-player baseline, spellF1 at the +-50 frame window, two matches (J03WOH, J03WN1):

| ball condition | spellF1 |
|---|---|
| always visible, clean players | 0.292 |
| hidden: provided gap pattern copied | 0.214 |
| hidden: constant hazard, real gap lengths | 0.201 |
| hidden: learned hazard, real gap lengths | 0.231 |
| learned ball hazard + players dropped + smoothed | 0.225 |

The masks agree within about 0.03, so the exact mask is not the main uncertainty. The learned hazard
hides the ball less often at real possession changes (32%) than overall (42%), because it favours slow
and dead-ball moments where "ball hidden means dead" happens to be right; the constant hazard is the
pessimistic bound. These are simulated; the organisers' published full-match floor is 0.2433.

## Hold the last owner

`analysis/hold_owner.py` replaces "nearest player, and call it dead when there is no ball". The owner
changes only on a frame where the ball is visible and a player is within `r_control` metres, and is
held through gaps and flight. Dead ball needs evidence at a visible frame (ball outside the pitch, or
hardly moving over 2 s), held through gaps. Runs shorter than `min_spell` / `min_dead` are merged away.
`analysis/hold_owner_eval.py` scores it on degraded DFL, choosing settings on the other matches only.

```bash
uv run python analysis/hold_owner_eval.py --matches DFL-MAT-J03WOH DFL-MAT-J03WN1   # ~9 min
```

spellF1 on the held-out match, mean of two matches (tuned on the other one):

| ball condition | nearest + vote | hold the last owner |
|---|---|---|
| clean ball | 0.292 | 0.552 |
| constant hazard (pessimistic) | 0.186 | 0.348 |
| learned hazard | 0.226 | 0.374 |

Against labels with every spell under 3 s merged (a proxy for "controlled possession"): 0.345 to 0.660,
0.207 to 0.392, 0.259 to 0.426. Chosen settings: `r_control` 0.75 to 1.0 m, dead state held without
limit, `d_still` 1 to 2 m. **`min_spell` follows the label definition** (10 frames on DFL's labels, 75 on
the merged ones), so it must not be tuned on DFL; it is the open question about the organisers' ground
truth.

What is still weak: with the ball hidden, dead/alive F1 is 0.75 to 0.83 (0.97 with a clean ball) and
team accuracy on live frames 0.73 to 0.83 (0.87 clean). Both gaps are what player-only inference during
ball gaps would have to close.

Limits: two matches, DFL's labels (finer than the organisers'), and a simulated mask. The held-out
gain is large and consistent across both matches, but it is not a score on the real match.
