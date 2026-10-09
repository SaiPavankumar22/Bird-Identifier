# Bird Finder

A small, **offline** classifier for bird recordings. Point it at a clip you recorded and
it gives you a best guess, a shortlist, and an honest confidence flag. No internet, no
account, no audio leaves your device.

Two models ship with the repo:

| | directory | classes | held-out top-1 | what it's for |
|---|---|---|---|---|
| **v4 (default)** | [`models_v4/`](models_v4/README.md) | 93 (91 species + `unknown bird` + `no bird`) | **84.6 %** grouped CV, **100 %** on a 65-clip frozen set | frozen Perch 2.0 backbone + our heads, trained on permissively licensed audio only; needs a 413 MB backbone download |
| **v1 (archived)** | [`models_v1_5species/`](models_v1_5species/README.md) | 5 | 43.6 % | the original 89k-parameter starter; tiny, no extra download |

---

## Quick start

```bash
python -m pip install -r requirements.txt   # needs ffmpeg on PATH for .mp3/.m4a

# one-time: the frozen Perch 2.0 backbone (413 MB, Apache-2.0) used by v4
hf download justinchuby/Perch-onnx perch_v2_no_dft.onnx --local-dir models_v4

python src/identify.py path/to/recording.mp3                                   # v4
python src/identify.py --models-dir models_v1_5species path/to/recording.mp3   # v1
python src/identify.py --json --no-log path/to/recording.mp3                   # machine-readable
python tools/field_log.py                                                      # your local field notebook
```

`hf` comes with `pip install huggingface_hub`. The backbone can also live anywhere
if you set `PERCH_BACKBONE=/path/to/perch_v2_no_dft.onnx`.

---

## How v4 works

1. audio → mono float32 at 32 kHz (the backbone's own input; no mel, no dB)
2. 5.0 s windows, 2.5 s hop, near-silence dropped, capped at 12 windows
3. each window → frozen Perch 2.0 → 1536-d embedding
4. windows → mean → L2-normalise
5. pooled vector → species head (93 logits) + bird/no-bird head (2 logits)

The **confidence flag** is a fixed rule, set before looking at results: *confident* if
the pooled probability ≥ 50 % and window agreement ≥ 50 % over ≥ 3 windows;
*moderately* if probability ≥ 35 %; otherwise *low*. It reports agreement, not
correctness.

Head classes are imported from `tools/train_head.py`, not re-declared, so the code
that saved the weights is the code that loads them.

---

## Results

| | v1 | **v4** |
|---|---|---|
| classes | 5 | **93** |
| chance level | 20 % | 1.1 % |
| split | 1 whole recording per species | 3-fold grouped CV (16,591 clips) |
| **top-1** | 43.6 % | **84.6 % ± 0.9** |
| **top-5** | 100 % | **96.8 %** |
| macro-F1 | — | 0.798 |
| bird/no-bird AUC | — | **0.946 ± 0.002** |
| recordist-disjoint top-1 | 0.463 ± 0.379 (LOCO) | 0.833 ± 0.020 |
| frozen test (65 clips, 5 species) | — | **1.000** |

### On audio it should decline

The same 11 clips asked of both models: 2 held-out species, 5 species neither model
knows, and 4 synthetic non-bird sounds.

| clip | truth | v1 | **v4** |
|---|---|---|---|
| American_Robin…HollowellParkTrail.mp3 | American Robin (held out) | **American Robin** (confident, 89 %) | **American Robin** (confident, 91 %) |
| Song_Sparrow…HollowellPark.mp3 | Song Sparrow (held out) | Mountain Chickadee (confident, 77 %) | **Song Sparrow** (confident, 98 %) |
| unknown_Common_Raven…PineridgeCampsite1.mp3 | Common Raven (unknown) | Mountain Chickadee (confident, 88 %) | Northern Raven (confident, 99 %) |
| unknown_Hermit_Thrush…BierstadtLake.mp3 | Hermit Thrush (unknown) | Mountain Chickadee (confident, 76 %) | Mountain Chickadee (low, 12 %) |
| unknown_Mountain_Bluebird…TwinOwl.mp3 | Mountain Bluebird (unknown) | Mountain Chickadee (confident, 81 %) | Eurasian Wren (low, 7 %) |
| unknown_Steller's_Jay…HollowellPark.mp3 | Steller's Jay (unknown) | Mountain Chickadee (confident, 100 %) | Eurasian Jay (moderately, 48 %) |
| demo_AMERICAN_CROW…CubLakeTrail.mp3 | American Crow (unknown) | Pine Siskin (low, 43 %) | Northern Raven (low, 16 %) |
| non_target_silence_10s.wav | silence | Mountain Chickadee (confident, 100 %) | **no bird** (low, 27 %) |
| non_target_sine_440_5s.wav | 440 Hz sine | Mountain Chickadee (confident, 100 %) | **no bird** (low, 29 %) |
| non_target_sweep_100to2000_5s.wav | sweep | American Robin (confident, 80 %) | **no bird** (moderately, 39 %) |
| non_target_whitenoise_5s.wav | white noise | Pine Siskin (confident, 100 %) | **no bird** (moderately, 39 %) |

- **v1 said "confident" 10 times — 9 of them wrong.**
- **v4 got 4/4 non-bird clips and 2/2 held-out species.**
- The unfavourable half: **v4 still names a species on 5/5 unknown clips.** It hedges
  with the confidence flag instead of refusing. The Raven row is a taxonomy alias
  (`Northern Raven` *is* *Corvus corax*); the genuinely bad row is Steller's Jay →
  Eurasian Jay at 48 %, flagged *moderately*.

Raw reports and the full write-up are published with the models on
[Hugging Face](https://huggingface.co/SaiPavankumar22/Bird-finder) (`evaluation/`,
`COMPARISON.md`).

---

## Data and licences

| source | clips | used for | licence |
|---|---|---|---|
| iNaturalist | 11,605 (91 species) | v4 species | CC0 |
| [DCASE 2018 Task B](https://dcase.community/challenge2018/task-bird-audio-detection) | 6,600 | v4 `unknown bird` / `no bird` | CC BY 4.0 |
| [NPS Rocky Mountain sound library](https://www.nps.gov/subjects/sound/soundlibrary.htm) | 18 | v1, v4 | public domain |

264 Xeno-canto British recordings (CC BY-NC-ND / BY-NC-SA) were **excluded** from
v4; retraining with them added back changes top-1 by −0.0004, so the licence is free.
The permissive training audio is published as
[`SaiPavankumar22/bird-sounds`](https://huggingface.co/datasets/SaiPavankumar22/bird-sounds)
with per-row licence and attribution.

- **Code** (`src/`, `tools/`): MIT, see [`LICENSE`](LICENSE).
- **Perch 2.0 backbone** (`perch_v2_no_dft.onnx`): Apache-2.0, Google Research, via
  [`justinchuby/Perch-onnx`](https://huggingface.co/justinchuby/Perch-onnx). Not
  redistributed in this repo.
- **Trained heads and v1 weights**: MIT, learned from CC0 / CC BY 4.0 / public-domain
  audio only.

---

## Repository map

```
├── README.md
├── LICENSE                      ← MIT
├── requirements.txt             ← torch, librosa, onnxruntime
├── src/
│   ├── identify.py              ← the CLI: windowed inference + confidence flag
│   ├── perch_v3.py              ← Perch backbone + heads (used by v4)
│   └── model.py                 ← BirdCallNet (v1's PyTorch fallback)
├── tools/
│   ├── field_log.py             ← local SQLite/JSONL field notebook (no network)
│   ├── train_head.py            ← head definitions loaded by perch_v3.py
│   └── __init__.py
├── models_v4/                   ← default model: heads + bundle + labels (+ backbone you download)
└── models_v1_5species/          ← archived 5-species CNN (ONNX + PyTorch)
```

---

## Limitations (read these)

- **Not a general bird ID.** 91 species is a small slice of the birds that exist.
- **Cross-validation is an upper bound.** Perch 2.0 was trained on iNaturalist and
  Xeno-canto, which sit inside these folds. The uncontaminated number is the 65-clip
  frozen test: top-1 1.000, AUC 0.830, 5 species.
- **The confidence flag is not a correctness detector.** It reports agreement, not
  truth.
- **13 species have fewer than 10 clips**, so per-class recall is noisy for them.
- First v4 call loads the 413 MB backbone (~25 s on CPU). Batch clips in one process.

Week 1 of the Hacktoberfest 2026 Open-Source AI Challenge — Touch Grass.
