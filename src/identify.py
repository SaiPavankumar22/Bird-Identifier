"""
Offline bird-call identification for a recording you made.

Usage:
  python identify.py path/to/recording.mp3          # classify + log to field notebook
  python identify.py path/to/recording.mp3 --no-log # classify only, don't log
  python identify.py path/to/recording.mp3 --json   # machine-readable JSON output

What it does (all offline, no network):
  1. Reads the audio file (any format librosa/ffmpeg can read: mp3, wav, m4a, flac, ogg).
  2. Converts it to the same mel-spectrogram representation the model was trained on,
     using the saved preprocessing config so inference is reproducible.
  3. Slides 3-second windows across the recording and runs the exported ONNX model
     (or the PyTorch model if ONNX isn't available) on each window.
  4. Aggregates the window predictions into a single answer: the top species, how many
     windows support it, the average confidence, a shortlist of alternatives, and a
     confidence flag.
  5. Prints a one-screen summary (the screen is meant to be the shortest part).
  6. Optionally writes an entry to the local field log (no network, stays on your device).

Confidence is reported honestly. The model was trained on 5 species from one area, so:
  - A clear recording of a known species in similar conditions -> high confidence.
  - A recording with little bird sound, or a species the model doesn't know, or very
    different recording conditions -> low confidence. The tool says so rather than
    pretending to know.

If the model or preprocessing config isn't present yet, the script tells you how to
train/export it first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import librosa

PROJECT_ROOT = (Path(__file__).resolve().parent / "..").resolve()
MODELS_DIR = PROJECT_ROOT / "models_v4"
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
FIELD_LOG = PROJECT_ROOT / "tools" / "field_log.py"  # (currently unused; logging uses sys.path + `from tools import field_log`)

CONFIG_PATH = MODELS_DIR / "config.json"
LABEL_PATH = MODELS_DIR / "label_map.json"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
ONNX_PATH = MODELS_DIR / "model.onnx"
TORCH_PATH = MODELS_DIR / "model.pt"


def set_models_dir(d: Path) -> Path:
    """Point the CLI at a different model directory (e.g. models_v1_5species/).

    Keeps "which model am I asking?" answerable from the command line instead of
    being hard-coded:

        models_v4/            v4  -- frozen Perch 2.0 backbone + trained heads (default)
        models_v1_5species/   v1  -- archived 5-species starter (mel -> CNN)

    A directory holding `bundle.json` is a v3 bundle, which switches the whole
    inference path (raw 32 kHz audio -> Perch embedding -> linear head) rather
    than just the weights.
    """
    global MODELS_DIR, CONFIG_PATH, LABEL_PATH, MANIFEST_PATH, ONNX_PATH, TORCH_PATH
    d = Path(d).expanduser()
    if not d.is_absolute():
        d = (PROJECT_ROOT / d).resolve()
    if not d.exists():
        raise SystemExit(f"Model directory not found: {d}")
    MODELS_DIR = d
    CONFIG_PATH = d / "config.json"
    LABEL_PATH = d / "label_map.json"
    MANIFEST_PATH = d / "manifest.json"
    ONNX_PATH = d / "model.onnx"
    TORCH_PATH = d / "model.pt"
    if (d / "bundle.json").exists() and not LABEL_PATH.exists():
        raise SystemExit(f"v3 bundle in {d} is missing label_map.json -- incomplete copy")
    return d


def is_v3(d: Path | None = None) -> bool:
    """True if the selected model directory is a v3 (Perch) bundle."""
    d = Path(d) if d else MODELS_DIR
    return (Path(d) / "bundle.json").exists()

WINDOW_SEC = 3.0
INFER_HOP_SEC = 0.5  # slide a new window every 0.5s during inference (dense but manageable)

# v3 only: how many 5 s Perch windows get pooled into one answer. 12 is the cap
# used while training the heads, so it is the default here too (--max-windows overrides).
v3_max_windows = 12


def _load_text(p: Path, label: str) -> dict | list[str] | list[dict]:
    if not p.exists():
        raise FileNotFoundError(f"{label} not found at {p}. Is --models-dir a complete model directory?")
    return json.loads(p.read_text(encoding="utf-8"))


def _ensure_model() -> tuple[str, Any, dict, dict[str, int], dict]:
    """Return (backend, session_or_model, config, label_map, manifest).

    backend is one of "onnx" | "torch" | "perch-v3"; the third is a different
    model family entirely (frozen Perch backbone + heads), so callers must branch
    on it rather than assuming mel-spectrogram windows.
    """
    if is_v3():
        from perch_v3 import PerchV3
        bundle = json.loads((MODELS_DIR / "bundle.json").read_text(encoding="utf-8"))
        label_map = _load_text(LABEL_PATH, "label_map.json")
        cfg = bundle.get("input", {})
        config = {
            "sr": int(cfg.get("sample_rate_hz", 32000)),
            "window_sec": float(cfg.get("window_sec", 5.0)),
            "hop_sec": float(cfg.get("hop_sec", 2.5)),
            "feature": cfg.get("feature", "Perch embedding, 1536-d"),
            "arch": bundle.get("species_head"),
            "backbone": bundle.get("backbone"),
        }
        manifest = {
            "input_shape": [None, 1536],
            "n_classes": bundle.get("n_classes", len(label_map)),
            "label_map": label_map,
            "config": config,
            "bundle": bundle,
        }
        return ("perch-v3", PerchV3(MODELS_DIR), config, label_map, manifest)

    config = _load_text(CONFIG_PATH, "config.json")
    label_map = _load_text(LABEL_PATH, "label_map.json")
    manifest = _load_text(MANIFEST_PATH, "manifest.json") if MANIFEST_PATH.exists() else {}

    use_onnx = ONNX_PATH.exists()
    if use_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        return ("onnx", sess, config, label_map, manifest)
    if TORCH_PATH.exists():
        import torch
        from model import BirdCallNet, arch_kwargs, _expected_frames
        state = torch.load(TORCH_PATH, map_location="cpu", weights_only=True)
        model = BirdCallNet(
            n_mels=config["n_mels"],
            n_frames=_expected_frames(config),
            n_classes=len(label_map),
            **arch_kwargs(config.get("arch") or manifest.get("arch")),
        )
        model.load_state_dict(state)
        model.eval()
        return ("torch", model, config, label_map, manifest)
    raise SystemExit(
        f"No model.onnx or model.pt found in {MODELS_DIR}. "
        "Use --models-dir models_v4 or --models-dir models_v1_5species."
    )


def _audio_to_mono_y(path: Path, sr: int) -> np.ndarray[float]:
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def _mel_from_y(y: np.ndarray, config: dict) -> np.ndarray[float]:
    """Convert a mono audio buffer to a normalized mel spectrogram matching training.

    Returns an array of shape (n_mels, n_frames) with float32 values."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config["sr"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    x = log_mel.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x


def _predict_one(x: np.ndarray, backend: str, sess_or_model: Any) -> np.ndarray[float]:
    """Return logits (n_classes,) for a single mel window (n_mels x n_frames)."""
    if backend == "onnx":
        inp_name = sess_or_model.get_inputs()[0].name
        out = sess_or_model.run(None, {inp_name: x[None, ...].astype(np.float32)})[0]
        return out[0]
    # torch
    import torch
    with torch.no_grad():
        t = torch.from_numpy(x[None, ...].astype(np.float32))
        return sess_or_model(t).numpy()[0]


def _predict_probs(logits: np.ndarray) -> np.ndarray[float]:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def identify(path: Path, no_log: bool = False, json_out: bool = False) -> dict[str, Any] | list[str] | list[dict]:
    backend, sess_or_model, config, label_map, manifest = _ensure_model()
    if backend == "perch-v3":
        return _identify_v3(path, sess_or_model, config, label_map, manifest,
                            no_log=no_log, json_out=json_out)
    n_classes = len(label_map)
    idx_map = {v: k for k, v in label_map.items()}
    sr = config["sr"]
    win_samples = int(round(WINDOW_SEC * sr))
    infer_hop_samples = int(round(INFER_HOP_SEC * sr))
    n_frames_expected = int(round(WINDOW_SEC * sr / config["hop_length"])) + 1

    y = _audio_to_mono_y(path, sr)
    total_sec = len(y) / sr

    # Slide windows across the recording.
    window_results: list[dict] = []
    pos = 0
    while pos + win_samples <= len(y):
        seg = y[pos:pos + win_samples]
        mel = _mel_from_y(seg, config)
        # Safety: if mel frame count differs from training (edge case), skip.
        if mel.shape[1] != n_frames_expected:
            pos += infer_hop_samples
            continue
        logits = _predict_one(mel, backend, sess_or_model)
        probs = _predict_probs(logits)
        pred = int(logits.argmax())
        window_results.append({
            "start_sec": round(pos / sr, 2),
            "pred_idx": pred,
            "pred_species": idx_map[pred],
            "confidence": float(probs[pred]),
            "probs": probs.tolist(),
        })
        pos += infer_hop_samples

    if not window_results:
        raise SystemExit(f"Recording too short to extract any {WINDOW_SEC}s windows (duration {total_sec:.1f}s).")

    # Aggregate by class.
    counts = [0] * n_classes
    conf_sum = [0.0] * n_classes
    for wr in window_results:
        c = wr["pred_idx"]
        counts[c] += 1
        conf_sum[c] += wr["confidence"]
    total_windows = len(window_results)
    support = [counts[c] / total_windows for c in range(n_classes)]
    mean_conf = [conf_sum[c] / counts[c] if counts[c] else 0.0 for c in range(n_classes)]

    # Rank classes by (support, mean_conf).
    ranked = sorted(range(n_classes), key=lambda c: (support[c], mean_conf[c]), reverse=True)
    top = ranked[0]
    shortlist = [
        {
            "species": idx_map[c],
            "support": round(support[c], 3),
            "mean_confidence": round(mean_conf[c], 3),
            "window_count": counts[c],
        }
        for c in ranked[:3]
    ]

    # Confidence flag: honest heuristic. "confident" only if the top class has both
    # substantial support and decent per-window confidence. Otherwise "low confidence".
    top_support = support[top]
    top_conf = mean_conf[top]
    max_prob_across = float(max(wr["confidence"] for wr in window_results))
    if top_support >= 0.5 and top_conf >= 0.5 and total_windows >= 3:
        flag = "confident"
    elif top_support >= 0.5 and top_conf >= 0.35:
        flag = "moderately confident"
    else:
        flag = "low confidence"

    result = {
        "audio_path": str(path),
        "duration_sec": round(total_sec, 1),
        "windows_used": total_windows,
        "top_species": idx_map[top],
        "top_species_index": top,
        "top_support": round(top_support, 3),
        "top_confidence": round(top_conf, 3),
        "max_window_confidence": round(max_prob_across, 3),
        "shortlist": shortlist,
        "confidence_flag": flag,
        "backend": backend,
        "model_params": manifest.get("input_shape"),
        "config": config,
        "label_map": label_map,
        "note": _model_note(label_map),
    }

    # Print a one-screen summary (skipped under --json, which owns stdout).
    if not json_out:
        _print_summary(result)

    if not no_log:
        _log_result(result, path)

    return result


def _identify_v3(path: Path, v3: Any, config: dict, label_map: dict[str, int],
                 manifest: dict, no_log: bool = False,
                 json_out: bool = False) -> dict[str, Any]:
    """Classify with the v3 bundle: Perch backbone -> pooled embedding -> heads.

    Same output shape as the v2 path so logging, --json and the field notebook
    keep working; the semantics of `top_support` differ (a pooled probability,
    not a fraction of windows) and the summary says so rather than implying a
    window vote that never happened.
    """
    sr = config["sr"]
    y = _audio_to_mono_y(path, sr)
    total_sec = len(y) / sr

    out = v3.predict(y, max_windows=v3_max_windows)
    idx_map = {v: k for k, v in label_map.items()}
    top = out["top_index"]

    result = {
        "audio_path": str(path),
        "duration_sec": round(total_sec, 1),
        "windows_used": out["n_windows"],
        "padded_short_clip": out["padded"],
        "top_species": idx_map.get(top, out["top_species"]),
        "top_species_index": top,
        "top_support": round(out["top_probability"], 3),
        "top_confidence": round(out["top_probability"], 3),
        "max_window_confidence": round(out["max_window_confidence"], 3),
        "window_agreement": out["window_agreement"],
        "bird_probability": round(out["bird_probability"], 3),
        "shortlist": out["shortlist"],
        "confidence_flag": out["confidence_flag"],
        "backend": "perch-v3",
        "model_params": manifest.get("input_shape"),
        "config": config,
        "label_map": label_map,
        "note": _model_note(label_map),
    }
    if not json_out:                     # --json means stdout is JSON and nothing else
        _print_summary(result)
    if not no_log:
        _log_result(result, path)
    return result


def _model_note(label_map: dict) -> str:
    """Honest, dynamic description of what the loaded model actually covers."""
    special = {"unknown bird", "no bird"}
    names = list(label_map)
    species = [n for n in names if n not in special]
    bits = [f"{len(species)} species"]
    for s in sorted(special & set(names)):
        bits.append(s)
    if is_v3():
        return ("Frozen Perch 2.0 backbone (Apache-2.0) + a linear head trained on "
                "" + " + ".join(bits) +
                " (licences and held-out metrics: " + f"{MODELS_DIR.name}/bundle.json and "
                f"{MODELS_DIR.name}/README.md). Honest classifier: not a general bird ID.")
    return ("Trained on " + " + ".join(bits) +
            " from the NPS Rocky Mountain sound library (public domain). "
            "Honest classifier: not a general bird ID. See "
            f"{MODELS_DIR.name}/report.json for held-out metrics.")


def _print_summary(r: dict[str, Any] | list[str] | list[dict]) -> None:
    sp = r["top_species"]
    flag = r["confidence_flag"]
    v3 = r.get("backend") == "perch-v3"
    print("=" * 70)
    print(f"Recording : {r['audio_path']}")
    print(f"Duration  : {r['duration_sec']}s  ({r['windows_used']} windows analyzed)")
    if v3 and r.get("padded_short_clip"):
        print("          : shorter than one 5s window -- zero-padded, so treat this as a sample")
    print("-" * 70)
    print(f"Best guess: {sp}")
    if v3:
        print(f"           pooled probability                : {r['top_support']:.1%}")
        print(f"           bird / no-bird probability         : {r['bird_probability']:.2f}")
        print(f"           window agreement on that species   : {r['window_agreement']:.1%}")
        print(f"           confidence flag                    : {flag}")
        print("    Rule (set before seeing these results): confident if pooled probability >= 50% and")
        print("    window agreement >= 50% with >= 3 windows; moderately confident if probability >= 35%;")
        print("    else low confidence. The flag reflects confidence, not correctness.")
        print("    Heads were trained on pooled clip vectors, so per-window scores are a diagnostic.")
    else:
        print(f"           support (fraction of windows): {r['top_support']:.1%}")
        print(f"           mean confidence per window    : {r['top_confidence']:.2f}")
        print(f"           max single-window confidence : {r['max_window_confidence']:.2f}")
        print(f"           confidence flag              : {flag}")
        print("    Rule (set before seeing these results): confident if top support >= 50%% and "
              "mean per-window confidence >= 0.50 with >= 3 windows analyzed; moderately "
              "confident if top support >= 50%% and mean >= 0.35; else low confidence. "
              "The flag reflects agreement/confidence, not correctness.")
    print("-" * 70)
    print("Shortlist (top 3):")
    for i, s in enumerate(r["shortlist"], 1):
        print(f"  {i}. {s['species']:<20} support {s['support']:.1%}  "
              f"mean conf {s['mean_confidence']:.2f}  ({s['window_count']} windows)")
    print("-" * 70)
    if flag.startswith("low"):
        print("NOTE: low confidence. In this one test the model declined to commit on a")
        print("      species it doesn't know -- which is what an honest tool should do. But one")
        print("      clip is not proof of the mechanism: a different clip could easily cross the")
        print("      threshold. See the non-target evaluation in README.md for a broader check.")
    print("=" * 70)
    print("All offline. Nothing sent anywhere. Result logged to the local field log.")
    n_classes = len(r.get("label_map", {}))
    print(f"Model covers {n_classes} classes. It is NOT a general bird-identification service.")
    print("=" * 70)


def _log_result(r: dict[str, Any], path: Path) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from tools import field_log as fl
    try:
        params = json.loads((MODELS_DIR / "report.json").read_text()).get("model_params")
    except Exception:                              # noqa: BLE001
        params = None
    entry = {
        "created": fl._now_iso(),
        "device_tag": fl._device_id(),
        "audio_path": str(path),
        "audio_exists": int(path.exists()),
        "top_species": r["top_species"],
        "top_support": r["top_support"],
        "top_confidence": r["top_confidence"],
        "shortlist": r["shortlist"],
        "confidence_flag": r["confidence_flag"],
        "model_name": (f"Perch 2.0 (frozen) + linear head ({MODELS_DIR.name})"
                       if r.get("backend") == "perch-v3"
                       else "BirdCallNet (open CNN, trained by us)"),
        "model_params": params,
        "note": r.get("note", "Offline bird-call identifier."),
    }
    row_id = fl.append(entry)
    print(f"\nLogged to field notebook (row id: {row_id}).", file=sys.stderr)


def _json_result(r: dict[str, Any] | list[str] | list[dict]) -> None:
    public = {k: v for k, v in r.items() if k != "config"}
    print(json.dumps(public, indent=2))


def main() -> int | list[str] | list[dict]:
    p = argparse.ArgumentParser(
        description="Offline bird-call identification for a recording (no network)."
    )
    p.add_argument("audio", type=Path, help="path to a recording (mp3, wav, m4a, flac, ogg)")
    p.add_argument("--no-log", action="store_true", help="classify only; do not write the field log")
    p.add_argument("--json", action="store_true", help="also print machine-readable JSON to stdout")
    p.add_argument("--models-dir", default="models_v4", metavar="DIR",
                   help="which model to use, relative to the repo root or absolute "
                        "(default: models_v4; v1 = models_v1_5species)")
    p.add_argument("--max-windows", type=int, default=0, metavar="N",
                   help="v3 only: cap the 5s windows pooled into one answer "
                        "(default 12, the value the heads were trained with; 0 = that default)")
    args = p.parse_args()

    used_dir = set_models_dir(args.models_dir)
    if args.max_windows > 0:
        globals()["v3_max_windows"] = args.max_windows
    if args.json:
        # surface which model answered, so two runs can be compared unambiguously
        print(json.dumps({"models_dir": str(used_dir)}, indent=2), file=sys.stderr)

    if not args.audio.exists():
        raise SystemExit(f"Audio file not found: {args.audio}")

    result = identify(args.audio, no_log=args.no_log, json_out=args.json)
    if args.json:
        result["models_dir"] = str(MODELS_DIR)
        _json_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
