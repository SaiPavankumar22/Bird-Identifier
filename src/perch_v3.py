"""Offline inference for the v3 model: frozen Perch 2.0 backbone + trained heads.

Why v3 exists: v1 (5 species) and v2 (95 classes) train a small CNN from scratch
on a few thousand windows, which caps out at 47.1% species top-1. v3 instead
freezes Google's Perch 2.0 embedding model (Apache-2.0, 1536-d, trained on
~10k species) and trains only a small head on cached embeddings. Same 95-class
label space as v2, so `--models-dir` is the only thing that changes.

What runs at inference time, in order:
  1. audio  -> mono float32 @ 32 kHz (the backbone's own rate; no mel, no dB).
  2. windows -> 5.0 s hops of 2.5 s, dropping near-silence (RMS < 5e-4),
     capped at 12 windows: the exact windowing `tools/perch_embed.py` used in
     training, so the head sees features from the same distribution.
  3. each window -> backbone -> 1536-d embedding.
  4. windows -> mean -> L2 normalise  (= `pooled_features` in training).
  5. pooled vector -> species head (95 logits) and bird/no-bird head (2 logits).

Heads are imported from `tools.train_head` rather than re-declared here, so the
module definitions used to *save* the weights are byte-for-byte the ones used to
*load* them -- a duplicated class definition is exactly how a silently wrong
state_dict load happens.

Everything is local and offline: one ONNX file, two PyTorch head files.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = (Path(__file__).resolve().parent / "..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_head import make_head  # noqa: E402  (shared definitions)

SR = 32000            # backbone rate; fixed by the ONNX graph, not configurable
WIN_SEC = 5.0
HOP_SEC = 2.5
WIN_SAMPLES = int(WIN_SEC * SR)
MIN_RMS = 5e-4        # same near-silence floor the training pass used
MAX_WINDOWS = 12      # same per-clip cap the training pass used
EMB_DIM = 1536


class PerchV3:
    """Loads the backbone + both heads once and answers `predict()` calls."""

    def __init__(self, models_dir: Path, threads: int = 0):
        self.dir = Path(models_dir)
        bundle_path = self.dir / "bundle.json"
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"no bundle.json in {self.dir} -- this is not a v3 model directory"
            )
        self.bundle: dict[str, Any] = json.loads(bundle_path.read_text(encoding="utf-8"))
        self.label_map: dict[str, int] = json.loads(
            (self.dir / "label_map.json").read_text(encoding="utf-8")
        )
        self.names: list[str] = [""] * len(self.label_map)
        for name, idx in self.label_map.items():
            self.names[idx] = name
        self.n_classes = len(self.label_map)

        backbone = find_backbone(self.dir, self.bundle.get("backbone_file", "perch_v2_no_dft.onnx"))
        if backbone is None:
            raise FileNotFoundError(
                f"Perch backbone not found. Looked in {self.dir} and its siblings "
                "for the file named in bundle.json (413 MB, Apache-2.0); set "
                "PERCH_BACKBONE=/path/to/perch_v2_no_dft.onnx or copy it next to "
                "bundle.json to enable this model."
            )
        self.backbone_path = backbone
        self.backbone = _load_backbone(backbone, threads)

        d = int(EMB_DIM)
        self.species_head = _load_head(self.dir / "species_head.pt", d)
        self.bird_head = _load_head(self.dir / "bird_head.pt", d)
        self.species_head.eval()
        self.bird_head.eval()

    # ---------------------------------------------------------------- audio
    def windows(self, y: np.ndarray, max_windows: int = MAX_WINDOWS
                ) -> tuple[list[tuple[int, float]], bool]:
        """(start_sample, start_sec) for loud-enough windows, plus a `padded` flag.

        Short recordings (< 5 s) are zero-padded to one window rather than
        rejected, because a 3-second phone clip is a normal thing to point at a
        bird; `padded` is reported so the caller can say so out loud.
        """
        padded = False
        if y.size < WIN_SAMPLES:
            y = np.pad(y, (0, WIN_SAMPLES - y.size))
            padded = True
        hop = int(round(HOP_SEC * SR))
        out: list[tuple[int, float]] = []
        pos = 0
        while pos + WIN_SAMPLES <= len(y):
            seg = y[pos:pos + WIN_SAMPLES]
            rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
            if rms >= MIN_RMS:
                out.append((pos, pos / SR))
            pos += hop
            if len(out) >= max_windows:
                break
        return out, padded

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, y: np.ndarray, max_windows: int = MAX_WINDOWS) -> dict[str, Any]:
        """Classify one recording. `y` is mono float32 at 32 kHz."""
        win, padded = self.windows(y, max_windows)
        if not win:
            # Nothing above the silence floor: fall back to every window anyway so
            # the tool can still answer "I hear no bird" instead of refusing.
            padded = padded or y.size < WIN_SAMPLES
            if y.size < WIN_SAMPLES:
                y = np.pad(y, (0, WIN_SAMPLES - y.size))
            win, _ = self.windows_with_floor(y, floor=0.0)
        if not win:
            raise ValueError("could not extract any audio window from this recording")

        batch = np.stack([y[s:s + WIN_SAMPLES] for s, _ in win]).astype(np.float32)
        emb = self.backbone_run(batch)                       # (n, 1536) float32

        pooled = emb.mean(axis=0, keepdims=True).astype(np.float32)
        n = float(np.linalg.norm(pooled))
        pooled = pooled / max(n, 1e-8)                       # == pooled_features()

        with torch.no_grad():
            t = torch.from_numpy(pooled)
            species_logits = self.species_head(t).numpy()[0]
            bird_logits = self.bird_head(t).numpy()[0]
            # Per-window scores are a *diagnostic* only: the heads were trained on
            # pooled clip vectors, so these are used for agreement, never for rank.
            tw = torch.from_numpy(emb)
            win_logits = self.species_head(tw).numpy()

        probs = _softmax(species_logits)
        win_probs = _softmax(win_logits)
        bird_prob = float(_softmax(bird_logits)[1])

        top = int(np.argmax(probs))
        agree = float((win_probs.argmax(axis=1) == top).mean())
        top_p = float(probs[top])

        if top_p >= 0.50 and agree >= 0.50 and len(win) >= 3:
            flag = "confident"
        elif top_p >= 0.35:
            flag = "moderately confident"
        else:
            flag = "low confidence"

        order = np.argsort(-probs)[:3]
        shortlist = [{
            "species": self.names[i],
            "support": round(float(probs[i]), 3),
            "mean_confidence": round(float(win_probs[:, i].mean()), 3),
            "window_count": int((win_probs.argmax(axis=1) == i).sum()),
        } for i in order]

        return {
            "n_windows": len(win),
            "padded": padded,
            "top_index": top,
            "top_species": self.names[top],
            "top_probability": top_p,
            "window_agreement": round(agree, 3),
            "bird_probability": bird_prob,
            "max_window_confidence": float(win_probs.max()),
            "shortlist": shortlist,
            "confidence_flag": flag,
            "probs": probs,
            "window_starts_sec": [round(s, 2) for _, s in win],
        }

    def windows_with_floor(self, y: np.ndarray, floor: float
                           ) -> tuple[list[tuple[int, float]], bool]:
        if y.size < WIN_SAMPLES:
            y = np.pad(y, (0, WIN_SAMPLES - y.size))
        hop = int(round(HOP_SEC * SR))
        out: list[tuple[int, float]] = []
        pos = 0
        while pos + WIN_SAMPLES <= len(y):
            seg = y[pos:pos + WIN_SAMPLES]
            if float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) >= floor:
                out.append((pos, pos / SR))
            pos += hop
            if len(out) >= MAX_WINDOWS:
                break
        return out, False

    def backbone_run(self, batch: np.ndarray) -> np.ndarray:
        name = self.backbone.get_inputs()[0].name
        out = self.backbone.run(None, {name: batch})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1, EMB_DIM)


def find_backbone(models_dir: Path, filename: str) -> Path | None:
    """Locate the shared 413 MB Perch ONNX file.

    The backbone is not part of any single model directory -- it is one Apache-2.0
    file shared by every Perch-head bundle (v3, v4, ...), so it is searched rather
    than copied: this directory first (the documented layout), then every sibling
    of it, then $PERCH_BACKBONE. A model dir must not have to carry 413 MB just
    because another one already has it.
    """
    env = os.environ.get("PERCH_BACKBONE", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    models_dir = Path(models_dir)
    seen: list[Path] = [models_dir / filename]
    parent = models_dir.parent
    if parent.is_dir():
        seen += sorted(d / filename for d in parent.iterdir() if d.is_dir())
    for cand in seen:
        if cand.is_file():
            return cand
    return None


def _load_backbone(path: Path, threads: int) -> Any:
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _load_head(path: Path, d: int) -> torch.nn.Module:
    """Rebuild a saved head from its own metadata -- never guess the architecture."""
    if not path.exists():
        raise FileNotFoundError(f"head not found: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    name = blob["head"]
    if int(blob["n_features"]) != d:
        raise ValueError(f"{path} expects {blob['n_features']}-d features, backbone gives {d}")
    head = make_head(name, d, int(blob["n_classes"]))
    result = head.load_state_dict(blob["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"{path}: state_dict mismatch {result}")
    head.eval()
    return head


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
