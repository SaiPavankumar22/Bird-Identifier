#!/usr/bin/env python3
"""Train and compare classifier heads on cached Perch 2.0 embeddings.

Why heads on a frozen backbone: Perch's model card says the embeddings were
"trained with the goal of being linearly separable" and that "training a simple
linear classifier on top of the model's outputs should work well". So the honest
experiment is a sweep over head capacity on one fixed, cached feature matrix:
each head costs seconds, every head sees exactly the same features, and any
difference is attributable to the head rather than to a different data pipeline.

Method (the part that decides whether a number is publishable):
  * Grouped k-fold over CLIPS, never windows. All windows of a recording live in
    the same fold, so a recording cannot leak into its own test set.
  * The frozen test set (never trained on, never selected with) is excluded from
    CV entirely and evaluated once at the end.
  * Metrics are CLIP-level: window probabilities are averaged per clip first, so
    a 3-minute recording cannot outvote twenty 10-second ones.
  * Species metrics count only class_kind == species; bird/no-bird AUC is
    reported separately. Fold mean and spread are always printed together -
    a headline without spread is not a result.

Heads: linear (logistic), linear-cosine (normalized), mlp1, mlp2.

Usage (on the VM):
  python tools/train_head.py --embeddings ~/bird_dataset/embeddings \
      --frozen ~/ref/frozen_test.csv --heads linear,mlp1,mlp2 --out ~/reports/head.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SPECIES_SPECIAL = {"unknown bird", "no bird"}


# --------------------------------------------------------------------- data
def load_cache(embed_dir: Path) -> tuple[np.ndarray, list[dict]]:
    emb = np.fromfile(embed_dir / "embeddings.bin", dtype=np.float32).reshape(-1, 1536)
    with (embed_dir / "index.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != emb.shape[0]:
        raise SystemExit(f"index has {len(rows)} rows but embeddings have {emb.shape[0]}")
    return emb, rows


def label_encoder(rows: list[dict]) -> tuple[dict[str, int], list[str]]:
    names = sorted({r["class_name"] for r in rows})
    enc = {n: i for i, n in enumerate(names)}
    return enc, names


def build_groups(rows: list[dict], frozen_clips: set[str], enc: dict[str, int]):
    """Per-clip records: label, kind, source, and the window rows belonging to it.

    Rows whose class is absent from `enc` (excluded by --permissive-only) are
    skipped, but `enumerate(rows)` still runs over the full list so the index stays
    aligned with the embedding matrix.
    """
    clips: dict[str, dict] = {}
    for i, r in enumerate(rows):
        if r["class_name"] not in enc:
            continue
        c = clips.setdefault(r["clip_id"], {
            "clip_id": r["clip_id"], "label": enc[r["class_name"]],
            "class_name": r["class_name"], "kind": r.get("class_kind", ""),
            "source": r.get("source", ""), "license": r.get("license", ""),
            "observer": r.get("observer", ""), "rows": []})
        c["rows"].append(i)
    kept = [c for c in clips.values() if c["clip_id"] not in frozen_clips]
    held = [c for c in clips.values() if c["clip_id"] in frozen_clips]
    return kept, held


def pooled_features(emb: np.ndarray, clips: list[dict]) -> np.ndarray:
    """Mean-pool each clip's windows into one L2-normalised feature vector.

    Clip-level features are what both training and scoring use, so a clip's
    windows can never be trained against a different clip's label (indexing the
    window matrix with clip indices is exactly the bug this replaces).
    L2 normalisation matches how the embeddings behave best: nearest-centroid on
    normalised clip vectors reaches ~0.93 top-1 on this data.
    """
    x = np.stack([emb[c["rows"]].mean(axis=0) for c in clips]).astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def grouped_folds(clips: list[dict], k: int, seed: int = 0,
                  group_by: str = "clip") -> list[int]:
    """Deterministic stratified folds.

    group_by="clip"      - every clip of a label is spread across folds.
    group_by="observer"  - all clips recorded by the SAME recordist land in the
                           same fold. This is the stricter, more honest test: if
                           a model only works on voices it has already heard,
                           clip-level splitting will hide that.
    """
    rng = np.random.default_rng(seed)
    if group_by == "observer":
        buckets: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(clips):
            key = (c.get("observer") or "").strip() or f"__clip_{i}"
            buckets[key].append(i)

        def stable(seed_key: str) -> int:
            return int(hashlib.sha1(seed_key.encode()).hexdigest()[:8], 16)

        # Greedy, largest bucket first: pick the fold that is currently poorest in
        # this bucket's classes (then smallest). Buckets often span several labels
        # (one recordist, many species) so assigning by first label would pile
        # thousands of clips into one fold and leave another fold empty.
        fold_of = [0] * len(clips)
        fold_n = [0] * k
        fold_cls: list[defaultdict] = [defaultdict(int) for _ in range(k)]
        for key in sorted(buckets, key=lambda kk: (-len(buckets[kk]), kk)):
            idx = buckets[key]
            labels_here = [clips[i]["label"] for i in idx]
            distinct = set(labels_here)
            best = min(range(k), key=lambda f: (
                sum(fold_cls[f][c] for c in distinct), fold_n[f], stable(f"{key}#{f}")))
            for i in idx:
                fold_of[i] = best
            fold_n[best] += len(idx)
            for c in labels_here:
                fold_cls[best][c] += 1
        return fold_of

    by_label: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(clips):
        by_label[c["label"]].append(i)
    fold_of = [0] * len(clips)
    for label in sorted(by_label):
        idx = np.array(by_label[label])
        rng.shuffle(idx)
        # a label with <k clips still gets spread as far as possible
        for pos, clip_i in enumerate(idx):
            fold_of[clip_i] = pos % k
    return fold_of


# ------------------------------------------------------------------ heads
class LinearHead(nn.Module):
    def __init__(self, d: int, c: int, cosine: bool = False):
        super().__init__()
        self.cosine = cosine
        self.fc = nn.Linear(d, c, bias=not cosine)
        if cosine:
            self.scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cosine:
            return self.fc(x)
        w = nn.functional.normalize(self.fc.weight, dim=1)
        return self.scale * (nn.functional.normalize(x, dim=1) @ w.t())


class MLP(nn.Module):
    def __init__(self, d: int, c: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, c))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_head(name: str, d: int, c: int) -> nn.Module:
    if name == "linear":
        return LinearHead(d, c, cosine=False)
    if name == "linear-cosine":
        return LinearHead(d, c, cosine=True)
    if name == "mlp1":
        return MLP(d, c, [768], 0.2)
    if name == "mlp2":
        return MLP(d, c, [1024, 512], 0.3)
    raise SystemExit(f"unknown head: {name}")


# -------------------------------------------------------------- training
def clip_weights(clips: list[dict]) -> torch.Tensor:
    """Total weight 1 per clip, so long recordings do not dominate the loss."""
    w = torch.tensor([1.0 / len(c["rows"]) for c in clips], dtype=torch.float32)
    return w / w.sum() * len(clips)


def class_weights(y: np.ndarray, n_cls: int, alpha: float) -> np.ndarray:
    """Inverse-frequency class weights, softened by alpha.

    alpha=1 fully balances classes; alpha=0 disables weighting. Needed because
    DCASE contributes thousands of no-bird / unknown-bird clips against a few
    hundred species clips, which otherwise collapses the head onto those two
    classes and gives species top-1 = 0 while top-5 still looks respectable.
    """
    if alpha <= 0:
        return np.ones(n_cls, dtype=np.float32)
    counts = np.bincount(y, minlength=n_cls).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (len(y) / (n_cls * counts)) ** alpha
    w = w / w.mean()
    return w.astype(np.float32)


def train_head(head: nn.Module, X: np.ndarray, y: np.ndarray, w: np.ndarray,
               tr: np.ndarray, te: np.ndarray, epochs: int, lr: float,
               device: str, seed: int,
               cls_w: np.ndarray | None = None) -> tuple[nn.Module, float]:
    """Fit on tr, keep the epoch with the best val loss carved from tr."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # early-stopping split inside the training fold (never the test fold)
    perm = rng.permutation(len(tr))
    n_val = max(1, int(0.1 * len(tr)))
    va, fit = tr[perm[:n_val]], tr[perm[n_val:]]
    if len(fit) < 2:
        fit, va = tr, tr

    head = head.to(device)
    Xtr = torch.from_numpy(X[fit]).to(device)
    ytr = torch.from_numpy(y[fit]).to(device)
    wtr = torch.from_numpy(w[fit]).to(device)
    wtr = wtr / wtr.sum() * len(fit)
    if cls_w is not None:
        wtr = wtr * torch.from_numpy(cls_w[y[fit]]).to(device)
        wtr = wtr / wtr.mean()
    Xva = torch.from_numpy(X[va]).to(device)
    yva = torch.from_numpy(y[va]).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_state, best_loss, best_ep = None, float("inf"), -1

    for ep in range(epochs):
        head.train()
        opt.zero_grad(set_to_none=True)
        # per-window CE, weighted so every CLIP contributes the same total weight
        # (otherwise a 3-minute recording outvotes twenty 10-second ones)
        per = nn.functional.cross_entropy(head(Xtr), ytr, reduction="none")
        loss = (per * wtr).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            vl = nn.functional.cross_entropy(head(Xva), yva).item()
        if vl < best_loss:
            best_loss, best_ep = vl, ep
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, best_loss


def predict(head: nn.Module, X: np.ndarray, idx: np.ndarray, device: str,
            batch: int = 8192) -> np.ndarray:
    outs = []
    head.eval()
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            xb = torch.from_numpy(X[idx[i:i + batch]]).to(device)
            outs.append(torch.softmax(head(xb), dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), np.float32)


# -------------------------------------------------------------- metrics
def topk_acc(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return float("nan")
    k = min(k, probs.shape[1])
    top = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def macro_f1(pred: np.ndarray, y: np.ndarray, n_cls: int) -> float:
    f1s = []
    for c in range(n_cls):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        p, r = tp / (tp + fp), tp / (tp + fn)
        # tp == 0 with the class still present in truth or predictions -> F1 = 0
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else float("nan")


def binary_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), no sklearn needed."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(probs: np.ndarray, clips: list[dict], n_cls: int, names: list[str]) -> dict:
    y = np.array([c["label"] for c in clips])
    kind = np.array([c["kind"] for c in clips])
    pred = probs.argmax(axis=1)
    sp = kind == "species"
    # bird/no-bird uses class_kind (species / bird / nonbird), NOT class_name:
    # comparing "no bird" against kind would make the negative class empty.
    is_bird = kind != "nonbird"
    species_ix = [i for i, n in enumerate(names) if n not in SPECIES_SPECIAL]
    score = probs[:, species_ix].sum(axis=1)
    res = {
        "n_clips": int(len(clips)),
        "species_clips": int(sp.sum()),
        "species_top1": topk_acc(probs[sp], y[sp], 1),
        "species_top5": topk_acc(probs[sp], y[sp], 5),
        "species_macro_f1": macro_f1(pred[sp], y[sp], n_cls),
        "overall_top1": topk_acc(probs, y, 1),
        "bird_nobird_auc": binary_auc(score[is_bird], score[~is_bird]),
        "n_bird_clips": int(is_bird.sum()),
        "n_nonbird_clips": int((~is_bird).sum()),
    }
    return res


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Train heads on cached Perch embeddings")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--frozen", default="", help="frozen_test.csv to exclude and score once")
    ap.add_argument("--heads", default="linear,linear-cosine,mlp1,mlp2")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permissive-only", action="store_true",
                    help="drop NC/ND clips so the shipped head never trains on them")
    ap.add_argument("--debug", action="store_true",
                    help="dump a few predictions from fold 0")
    ap.add_argument("--class-weight-alpha", type=float, default=1.0,
                    help="0 = no class balancing, 1 = full inverse frequency")
    ap.add_argument("--skip-binary", action="store_true",
                    help="skip the two-stage bird/no-bird head")
    ap.add_argument("--group-by", choices=["clip", "observer"], default="clip",
                    help="observer = all clips from one recordist stay in one fold")
    ap.add_argument("--save-dir", default="",
                    help="train final heads on all training clips and save them here")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    print(f"device: {device}")

    emb, rows = load_cache(Path(args.embeddings))
    # Which rows are allowed to DEFINE the label space. Under --permissive-only the
    # NC/ND rows are excluded from the encoder, so a species whose only clips are
    # NC/ND loses its class entirely instead of keeping an output it can never win.
    # `rows` itself is NEVER filtered: build_groups enumerates it and uses that index
    # into `emb`, so removing entries would silently misalign features and labels.
    if args.permissive_only:
        rows_enc = [r for r in rows if _permissive(r.get("license", ""))]
    else:
        rows_enc = rows
    enc, names = label_encoder(rows_enc)
    n_cls = len(names)
    print(f"windows: {emb.shape[0]}  classes: {n_cls}")
    if args.permissive_only and n_cls < len({r['class_name'] for r in rows}):
        gone = sorted({r["class_name"] for r in rows} - set(names))
        print(f"--permissive-only: {len(gone)} class(es) have no permissive clips and "
              f"are excluded from the label space: {gone}")

    frozen: set[str] = set()
    if args.frozen and Path(args.frozen).exists():
        with open(args.frozen, newline="", encoding="utf-8") as f:
            frozen = {r["clip_id"] for r in csv.DictReader(f)}
        print(f"frozen clips excluded from training: {len(frozen)}")
    elif args.frozen:
        raise SystemExit(f"frozen set not found: {args.frozen}")

    clips, held = build_groups(rows, frozen, enc)
    if args.permissive_only:
        before = len(clips)
        clips = [c for c in clips if _permissive(c["license"])]
        print(f"--permissive-only: dropped {before - len(clips)} clips")
    if not clips:
        raise SystemExit("no training clips left")
    if len(held) != len(frozen):
        missing = sorted(frozen - {c["clip_id"] for c in held})
        print(f"WARNING: {len(missing)} frozen clip(s) have no class left in this run "
              f"and were not scored: {missing[:5]}")

    Xc = pooled_features(emb, clips)
    y_clip = np.array([c["label"] for c in clips], dtype=np.int64)
    w_clip = clip_weights(clips).numpy()
    fold_of = grouped_folds(clips, args.k, args.seed, args.group_by)
    cls_w = class_weights(y_clip, n_cls, args.class_weight_alpha)
    n_obs = len({(c.get("observer") or "").strip() for c in clips if (c.get("observer") or "").strip()})
    print(f"training clips: {len(clips)}  folds: {args.k}  group_by={args.group_by} "
          f"({n_obs} named observers)  class-weight alpha={args.class_weight_alpha} "
          f"(max w={cls_w.max():.1f})")

    report: dict = {"n_classes": n_cls, "k": args.k, "heads": {},
                    "group_by": args.group_by, "n_observers": n_obs,
                    "n_windows": int(emb.shape[0]), "n_train_clips": len(clips),
                    "frozen_clips": len(frozen), "device": device}

    for head_name in args.heads.split(","):
        head_name = head_name.strip()
        if not head_name:
            continue
        t0 = time.time()
        fold_metrics = []
        for fold in range(args.k):
            te = np.array([i for i, f in enumerate(fold_of) if f == fold])
            tr = np.array([i for i, f in enumerate(fold_of) if f != fold])
            if len(te) == 0 or len(tr) == 0:
                continue
            head = make_head(head_name, emb.shape[1], n_cls)
            head, best_val = train_head(head, Xc, y_clip, w_clip, tr, te,
                                        args.epochs, args.lr, device, args.seed + fold,
                                        cls_w=cls_w)
            te_clips = [clips[i] for i in te]
            probs_clip = predict(head, Xc, te, device)
            m = evaluate(probs_clip, te_clips, n_cls, names)
            m["best_val_loss"] = round(best_val, 4)
            fold_metrics.append(m)
            print(f"  [{head_name}] fold {fold}: species top1={m['species_top1']:.3f} "
                  f"top5={m['species_top5']:.3f} f1={m['species_macro_f1']:.3f} "
                  f"auc={m['bird_nobird_auc']:.3f} (n_sp={m['species_clips']}, "
                  f"bird={m['n_bird_clips']}, nonbird={m['n_nonbird_clips']})")
            if fold == 0 and _truthy(args.debug):
                for j in range(min(8, len(te_clips))):
                    order = np.argsort(-probs_clip[j])[:5]
                    print(f"      truth={te_clips[j]['class_name']!r} "
                          f"pred={names[order[0]]!r} top5={[names[i] for i in order]}")
        agg = summarise(fold_metrics)
        agg["folds"] = fold_metrics
        agg["seconds"] = round(time.time() - t0, 1)
        report["heads"][head_name] = agg
        print(f"  => {head_name}: species top1 {agg['species_top1_mean']:.3f} "
              f"± {agg['species_top1_std']:.3f} | top5 {agg['species_top5_mean']:.3f} "
              f"± {agg['species_top5_std']:.3f} | auc {agg['bird_nobird_auc_mean']:.3f} "
              f"± {agg['bird_nobird_auc_std']:.3f} ({agg['seconds']}s)")

    # --- two-stage bird/no-bird head -------------------------------------------------
    # The species head is trained with class-balanced loss, so its summed species
    # probability is a mediocre bird/no-bird score (rank AUC ~0.75). A dedicated
    # binary head on the same frozen embeddings is the right tool for that question
    # and does not compete with the species head for capacity.
    if not args.skip_binary:
        y_bin = np.array([0 if c["kind"] == "nonbird" else 1 for c in clips],
                         dtype=np.int64)
        fold_aucs = []
        for fold in range(args.k):
            te = np.array([i for i, f in enumerate(fold_of) if f == fold])
            tr = np.array([i for i, f in enumerate(fold_of) if f != fold])
            if len(te) == 0 or len(tr) == 0:
                continue
            bh = LinearHead(emb.shape[1], 2)
            bh, _ = train_head(bh, Xc, y_bin, w_clip, tr, te, args.epochs,
                               args.lr, device, args.seed + fold)
            p = predict(bh, Xc, te, device)[:, 1]
            yt = y_bin[te]
            fold_aucs.append(binary_auc(p[yt == 1], p[yt == 0]))
        binary = {
            "auc_mean": float(np.mean(fold_aucs)),
            "auc_std": float(np.std(fold_aucs)),
            "folds": [round(a, 4) for a in fold_aucs],
        }
        report["binary_head"] = binary
        print(f"  => two-stage bird/no-bird head: AUC {binary['auc_mean']:.3f} "
              f"± {binary['auc_std']:.3f} (species head alone: "
              f"{report['heads'].get('linear-cosine', {}).get('bird_nobird_auc_mean', float('nan')):.3f})")

    if held and frozen:
        print(f"\nfrozen test set: {len(held)} clips")
        for head_name in args.heads.split(","):
            head_name = head_name.strip()
            if not head_name or head_name not in report["heads"]:
                continue
            # retrain on all training clips, then score the frozen set once
            head = make_head(head_name, emb.shape[1], n_cls)
            all_idx = np.arange(len(clips))
            head, _ = train_head(head, Xc, y_clip, w_clip, all_idx, all_idx,
                                 args.epochs, args.lr, device, args.seed,
                                 cls_w=cls_w)
            Xh = pooled_features(emb, held)
            probs_clip = predict(head, Xh, np.arange(len(held)), device)
            m = evaluate(probs_clip, held, n_cls, names)
            report["heads"][head_name]["frozen_test"] = m
            print(f"  [frozen/{head_name}] species top1={m['species_top1']:.3f} "
                  f"top5={m['species_top5']:.3f} | auc={m['bird_nobird_auc']:.3f} "
                  f"| clips={m['n_clips']} species_clips={m['species_clips']}")

    if args.save_dir:
        save_final_heads(args, emb, clips, held, y_clip, w_clip, cls_w,
                         n_cls, names, enc, report, device)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


def save_final_heads(args, emb, clips, held, y_clip, w_clip, cls_w,
                     n_cls, names, enc, report, device) -> None:
    """Fit the shipped heads on every training clip and write a loadable bundle.

    The bundle records what the weights expect (5 s @ 32 kHz Perch window, the
    exact backbone file and its licence) so the artifact can be used without
    reading this script.
    """
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    Xc = pooled_features(emb, clips)
    all_idx = np.arange(len(clips))

    species_name = args.heads.split(",")[0].strip() or "linear"
    sh = make_head(species_name, emb.shape[1], n_cls)
    sh, _ = train_head(sh, Xc, y_clip, w_clip, all_idx, all_idx, args.epochs,
                       args.lr, device, args.seed, cls_w=cls_w)

    y_bin = np.array([0 if c["kind"] == "nonbird" else 1 for c in clips],
                     dtype=np.int64)
    bh = LinearHead(emb.shape[1], 2)
    bh, _ = train_head(bh, Xc, y_bin, w_clip, all_idx, all_idx, args.epochs,
                       args.lr, device, args.seed)

    torch.save({"state_dict": sh.state_dict(), "head": species_name,
                "n_features": emb.shape[1], "n_classes": n_cls},
               out / "species_head.pt")
    torch.save({"state_dict": bh.state_dict(), "head": "linear",
                "n_features": emb.shape[1], "n_classes": 2},
               out / "bird_head.pt")
    (out / "label_map.json").write_text(
        json.dumps({n: i for i, n in enumerate(names)}, indent=2))
    (out / "label_index.json").write_text(json.dumps(enc, indent=2))
    bundle = {
        "backbone": "Perch 2.0 (Google) via justinchuby/Perch-onnx",
        "backbone_licence": "Apache-2.0",
        "backbone_file": "perch_v2_no_dft.onnx",
        "input": {"sample_rate_hz": 32000, "window_sec": 5.0,
                  "hop_sec": args.hop if hasattr(args, "hop") else 2.5,
                  "feature": "Perch embedding, 1536-d, L2-normalised per clip"},
        "species_head": species_name,
        "n_classes": n_cls,
        "n_training_clips": len(clips),
        "n_frozen_held_out": len(held),
        "class_weight_alpha": args.class_weight_alpha,
        "metrics": {k: report["heads"].get(k, {}).get("species_top1_mean")
                    for k in report["heads"]},
        "binary_head_auc": report.get("binary_head"),
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2))
    print(f"\nsaved heads -> {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}  {f.stat().st_size} bytes")


def _permissive(licence: str) -> bool:
    t = (licence or "").strip().lower()
    if not t:
        return False
    if "public domain" in t or "cc0" in t:
        return True
    if "nc" in t or "nd" in t:
        return False
    return "by" in t


def _truthy(v) -> bool:
    return bool(v)


def summarise(folds: list[dict]) -> dict:
    out: dict = {}
    if not folds:
        return out
    for key in folds[0]:
        if key in ("best_val_loss",):
            continue
        vals = [f[key] for f in folds if isinstance(f.get(key), (int, float))]
        if not vals:
            continue
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
