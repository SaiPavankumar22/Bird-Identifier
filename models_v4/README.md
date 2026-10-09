# `models_v4/` — the v4 model (frozen Perch 2.0 + permissively licensed head)

The default model. A frozen Perch 2.0 audio backbone turns each 5 s window into a
1536-d embedding; two small heads trained by us turn the pooled embedding into a
species guess and a bird / no-bird score. The heads were trained on
**permissively licensed audio only** (CC0 / CC BY 4.0 / public domain).

```bash
python src/identify.py YOUR_RECORDING.mp3                                   # v4 (default)
python src/identify.py YOUR_RECORDING.mp3 --models-dir models_v1_5species   # v1
```

## Files

| file | bytes | what it is |
|---|---|---|
| `species_head.pt` | 573,453 | `linear-cosine` head, 1536 → **93** classes |
| `bird_head.pt` | 14,325 | `linear` head, 1536 → 2 (bird / no-bird), the two-stage score |
| `bundle.json` | 759 | what the weights expect: window, rate, head names, licence, metrics |
| `label_map.json` / `label_index.json` | 2,304 each | class name → index (**93** entries) |
| `perch_v2_no_dft.onnx` | 413,350,933 | the frozen backbone — **not committed**; see below |

93 classes = 91 species + `unknown bird` + `no bird`.

### The backbone file

`perch_v2_no_dft.onnx` (413 MB, Apache-2.0, Google's Perch 2.0 via
[`justinchuby/Perch-onnx`](https://huggingface.co/justinchuby/Perch-onnx)) is over
GitHub's 100 MB per-file limit, so it is downloaded separately:

```bash
hf download justinchuby/Perch-onnx perch_v2_no_dft.onnx --local-dir models_v4
```

`src/perch_v3.py` looks for it in this directory, then in every sibling directory,
then at `$PERCH_BACKBONE`. Without it, `identify.py` names the exact file it is
looking for instead of failing mid-run.

A self-contained copy (heads + backbone) is published at
[`SaiPavankumar22/Bird-finder`](https://huggingface.co/SaiPavankumar22/Bird-finder).

## Measured results (3-fold grouped CV, 16,591 training clips, 93 classes)

| protocol | species top-1 | top-5 | bird/no-bird AUC |
|---|---|---|---|
| clip-grouped CV, `linear-cosine` (shipped) | **0.846 ± 0.009** | **0.968** | 0.946 ± 0.002 |
| **recordist-disjoint** CV, `linear-cosine` | **0.833 ± 0.020** | 0.962 | 0.935 ± 0.029 |
| frozen test set (65 clips, 5 species, never trained on) | **1.000** | 1.000 | 0.830 |

Heads compared and rejected: `linear` (0.768), `mlp1` (0.797), `mlp2` (0.782).

Adding the 264 non-commercial British clips back into training gives 0.8454 vs the
shipped 0.8458 — **−0.0004** — so excluding them costs nothing.

Raw reports and the full write-up: `evaluation/` and `COMPARISON.md` in
[`SaiPavankumar22/Bird-finder`](https://huggingface.co/SaiPavankumar22/Bird-finder).

## Training data

| source | clips | licence |
|---|---|---|
| iNaturalist | 11,605 | CC0 |
| DCASE 2018 Task B | 6,600 | CC BY 4.0 (binary bird / no-bird only) |
| NPS Rocky Mountain | 18 | public domain |
| Xeno-canto British set | 264 | CC BY-NC-ND / BY-NC-SA — **excluded** |

The permissive audio is published as
[`SaiPavankumar22/bird-sounds`](https://huggingface.co/datasets/SaiPavankumar22/bird-sounds).
13 species still have fewer than 10 clips — those are the weak classes.

## Honest limitations

- **CV is an upper bound.** Perch 2.0 was trained on iNaturalist and Xeno-canto,
  which are inside these folds. The uncontaminated number is the frozen test:
  65 clips, 5 species, top-1 1.000, AUC 0.830.
- **It still names a species for birds it does not know.** On an 11-clip check it
  named one on 5/5 unknown clips, hedging with the confidence flag rather than
  refusing. It got all 4 non-bird clips right (`no bird`) and 2/2 held-out species.
- First call loads the 413 MB backbone (~25 s on CPU). Batch clips in one process.
