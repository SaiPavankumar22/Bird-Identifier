# `models_v1_5species/` — the ARCHIVED starter model (v1, 5 classes)

The original Hacktoberfest starter model, kept for comparison and for anyone who
wants the small, 5-species, 89k-parameter version that needs no backbone download.
`identify.py` does **not** use this by default — pass
`--models-dir models_v1_5species` to ask it.

| | |
|---|---|
| **Version** | v1 — the original starter, 5 species from one park |
| **Classes** | **5** — American Robin, Dark-eyed Junco, Mountain Chickadee, Pine Siskin, Song Sparrow |
| **Architecture** | BirdCallNet `tiny`, **89,333** parameters |
| **Training data** | NPS Rocky Mountain sound library only (public domain), 18 recordings |
| **Training windows** | 143 total — 104 train / 39 validation |
| **Split** | one entire recording per species held out |

## Held-out numbers (39 validation windows)

| metric | value |
|---|---|
| top-1 accuracy | **43.6 %** (chance = 20 %) |
| top-5 accuracy | 100 % |
| American Robin | 100 % recall (8/8) |
| Dark-eyed Junco | 0 % recall — all 8 predicted Mountain Chickadee |
| Mountain Chickadee | 62.5 % recall / 31.2 % precision |
| Pine Siskin | 42.9 % recall / 50.0 % precision |
| Song Sparrow | 12.5 % recall (6/8 predicted Dark-eyed Junco) |

Leave-one-recording-out cross-validation on the same data: **0.463 ± 0.379**
(18 folds). Full per-class table and confusion matrix: [`report.json`](report.json).

## Behavioural difference that matters

On an 11-clip check (2 held-out species, 5 unknown species, 4 synthetic non-bird
sounds), v1 returned a *confident* flag on 10 of 11 clips — **9 of them wrong**: it
called silence, a sine tone, white noise and four unknown species "Mountain
Chickadee" or similar, confidently. The default model in
[`../models_v4/`](../models_v4/README.md) answered `no bird` on all 4 non-bird
clips and flagged most unknown species *low*.
