#!/usr/bin/env python3
"""FashionMNIST benchmark with ADZE-inspired summary statistics.

This is not population genetics. It is a sanity check for the paper's broader modeling
claim: how much information can a fully connected DNN extract from compact, structured,
rarefaction-style summaries instead of raw high-dimensional inputs?

Digest design:

1. Treat an image as a small "landscape" and define three pseudo-populations either by
   horizontal bands or vertical bands.
2. Treat columns or rows as aligned loci.
3. For several intensity thresholds, compute ADZE-like presence summaries:
   alpha_j, private pi_j, pair-private pihat_ij, and triple-active fraction.
4. Add continuous analogues: intensity means/stds, soft private excess, and soft pair
   shared intensity.
5. Optionally add multiscale pooled pixels, thresholded row/column activity profiles,
   intensity histograms, and raw pixels as ablations. This is still a fully connected
   input, not a CNN.

Run examples:

  python scripts/benchmarks/fashion_mnist_adze_stats.py --mode adze --epochs 60
  python scripts/benchmarks/fashion_mnist_adze_stats.py --mode adze_pool --epochs 60
  python scripts/benchmarks/fashion_mnist_adze_stats.py --mode adze_rich --epochs 100
  python scripts/benchmarks/fashion_mnist_adze_stats.py --mode adze_hybrid --epochs 100
  python scripts/benchmarks/fashion_mnist_adze_stats.py --mode adze_hybrid --shift-augment cardinal --epochs 70
  python scripts/benchmarks/fashion_mnist_adze_stats.py --mode raw --epochs 60
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import struct
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler


LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_idx_images(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_idx_labels(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n)


def find_fashion_mnist(root_arg: str | None):
    candidates = []
    if root_arg:
        candidates.append(Path(root_arg))
    candidates.extend(
        [
            Path.home() / "data" / "FashionMNIST" / "raw",
            Path.home() / "Downloads" / "data" / "FashionMNIST" / "raw",
        ]
    )
    for root in candidates:
        if (root / "train-images-idx3-ubyte").exists():
            return root
        if (root / "train-images-idx3-ubyte.gz").exists():
            return root
    raise SystemExit("Could not find FashionMNIST raw IDX files")


def load_fashion_mnist(root_arg=None):
    root = find_fashion_mnist(root_arg)
    def p(stem):
        raw = root / stem
        gz = root / f"{stem}.gz"
        return raw if raw.exists() else gz

    train_x = read_idx_images(p("train-images-idx3-ubyte"))
    train_y = read_idx_labels(p("train-labels-idx1-ubyte"))
    test_x = read_idx_images(p("t10k-images-idx3-ubyte"))
    test_y = read_idx_labels(p("t10k-labels-idx1-ubyte"))
    return root, train_x, train_y, test_x, test_y


def shift_images(images: np.ndarray, dy: int, dx: int):
    if dy == 0 and dx == 0:
        return images.copy()
    out = np.zeros_like(images)
    src_y0 = max(0, -dy)
    src_y1 = min(images.shape[1], images.shape[1] - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(images.shape[1], images.shape[1] + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(images.shape[2], images.shape[2] - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(images.shape[2], images.shape[2] + dx)
    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = images[:, src_y0:src_y1, src_x0:src_x1]
    return out


def shift_set(name: str):
    if name == "none":
        return [(0, 0)]
    if name == "cardinal":
        return [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    if name == "nine":
        return [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
    raise ValueError(name)


def augment_with_shifts(images: np.ndarray, labels: np.ndarray, mode: str):
    shifts = shift_set(mode)
    if shifts == [(0, 0)]:
        return images, labels, np.arange(len(images), dtype=np.int64), shifts
    aug_images = []
    aug_labels = []
    aug_groups = []
    base_groups = np.arange(len(images), dtype=np.int64)
    for dy, dx in shifts:
        aug_images.append(shift_images(images, dy, dx))
        aug_labels.append(labels.copy())
        aug_groups.append(base_groups)
    return np.concatenate(aug_images, axis=0), np.concatenate(aug_labels, axis=0), np.concatenate(aug_groups, axis=0), shifts


ADZE_VARIANTS = {
    "adze_test1": {"population_count": 3, "higher_moments": True},
    "adze_test2": {"population_count": 4, "higher_moments": False},
    "adze_test3": {"population_count": 4, "higher_moments": True},
}


def _split_indices(length: int, parts: int):
    return np.array_split(np.arange(length), parts)


def _central_moment(x: np.ndarray, order: int):
    mean = x.mean(axis=2, keepdims=True)
    return np.mean((x - mean) ** order, axis=2).astype(np.float32)


def _append_higher_moments(feats, names, per_locus: np.ndarray, base_names):
    for order in [3, 4]:
        feats.append(_central_moment(per_locus.astype(np.float32), order))
        names.extend([f"{name}_moment{order}" for name in base_names])


def band_values(images: np.ndarray, scheme: str, population_count: int = 3):
    x = images.astype(np.float32) / 255.0
    if population_count < 3:
        raise ValueError(f"population_count must be at least 3, got {population_count}")
    if scheme == "horizontal":
        bands = [x[:, idx, :].mean(axis=1) for idx in _split_indices(x.shape[1], population_count)]
    elif scheme == "vertical":
        bands = [x[:, :, idx].mean(axis=2) for idx in _split_indices(x.shape[2], population_count)]
    else:
        raise ValueError(scheme)
    return np.stack(bands, axis=1)  # (N, population_count, 28)


def adze_like_features_from_values(vals: np.ndarray, thresholds, include_higher_moments: bool = False):
    feats = []
    names = []
    population_count = vals.shape[1]
    pop_ids = list(range(population_count))
    pairs = list(combinations(pop_ids, 2))

    # Continuous summaries.
    means = vals.mean(axis=2)
    stds = vals.std(axis=2)
    feats.extend([means, stds])
    cont_names = [f"cont_{i}" for i in pop_ids]
    names.extend([f"cont_mean_{i}" for i in pop_ids])
    names.extend([f"cont_std_{i}" for i in pop_ids])
    if include_higher_moments:
        _append_higher_moments(feats, names, vals, cont_names)

    max_other = np.stack(
        [
            np.max(np.delete(vals, j, axis=1), axis=1)
            for j in pop_ids
        ],
        axis=1,
    )
    soft_private = np.maximum(vals - max_other, 0.0).mean(axis=2)
    feats.append(soft_private)
    soft_private_names = [f"soft_private_{i}" for i in pop_ids]
    names.extend(soft_private_names)
    if include_higher_moments:
        _append_higher_moments(feats, names, np.maximum(vals - max_other, 0.0), soft_private_names)

    soft_pair_rows = []
    for a, b in pairs:
        other = [p for p in pop_ids if p not in {a, b}]
        max_excluded = np.max(vals[:, other], axis=1)
        soft_pair_rows.append(np.maximum(np.minimum(vals[:, a], vals[:, b]) - max_excluded, 0.0))
    soft_pair_locus = np.stack(soft_pair_rows, axis=1)
    feats.append(soft_pair_locus.mean(axis=2).astype(np.float32))
    soft_pair_names = [f"soft_pair_{a}{b}" for a, b in pairs]
    names.extend(soft_pair_names)
    if include_higher_moments:
        _append_higher_moments(feats, names, soft_pair_locus, soft_pair_names)

    # Thresholded ADZE-like summaries.
    for t in thresholds:
        active = vals >= t
        alpha = active.mean(axis=2).astype(np.float32)
        private_locus = np.stack(
            [
                active[:, j] & ~np.any(np.delete(active, j, axis=1), axis=1)
                for j in pop_ids
            ],
            axis=1,
        )
        pair_locus = np.stack(
            [
                active[:, a] & active[:, b] & ~np.any(active[:, [p for p in pop_ids if p not in {a, b}]], axis=1)
                for a, b in pairs
            ],
            axis=1,
        )
        all_active_locus = np.all(active, axis=1, keepdims=True)
        private = private_locus.mean(axis=2).astype(np.float32)
        pair = pair_locus.mean(axis=2).astype(np.float32)
        all_active = all_active_locus.mean(axis=2).astype(np.float32)
        feats.extend([alpha, private, pair, all_active])
        alpha_names = [f"thr{t:.2f}_alpha_{i}" for i in pop_ids]
        private_names = [f"thr{t:.2f}_private_{i}" for i in pop_ids]
        pair_names = [f"thr{t:.2f}_pair_{a}{b}" for a, b in pairs]
        all_active_name = f"thr{t:.2f}_triple" if population_count == 3 else f"thr{t:.2f}_all{population_count}"
        names.extend(alpha_names)
        names.extend(private_names)
        names.extend(pair_names)
        names.append(all_active_name)
        if include_higher_moments:
            _append_higher_moments(feats, names, active.astype(np.float32), alpha_names)
            _append_higher_moments(feats, names, private_locus.astype(np.float32), private_names)
            _append_higher_moments(feats, names, pair_locus.astype(np.float32), pair_names)
            _append_higher_moments(feats, names, all_active_locus.astype(np.float32), [all_active_name])
    return np.concatenate(feats, axis=1), names


def image_moment_features(images: np.ndarray):
    x = images.astype(np.float32) / 255.0
    n = len(x)
    row_grid = np.arange(28, dtype=np.float32)[None, :, None] / 27.0
    col_grid = np.arange(28, dtype=np.float32)[None, None, :] / 27.0
    mass = x.sum(axis=(1, 2)) + 1e-6
    row_center = (x * row_grid).sum(axis=(1, 2)) / mass
    col_center = (x * col_grid).sum(axis=(1, 2)) / mass
    row_var = (x * (row_grid - row_center[:, None, None]) ** 2).sum(axis=(1, 2)) / mass
    col_var = (x * (col_grid - col_center[:, None, None]) ** 2).sum(axis=(1, 2)) / mass
    total = mass / (28 * 28)
    symmetry_lr = np.abs(x - x[:, :, ::-1]).mean(axis=(1, 2))
    symmetry_ud = np.abs(x - x[:, ::-1, :]).mean(axis=(1, 2))
    return np.stack([total, row_center, col_center, row_var, col_var, symmetry_lr, symmetry_ud], axis=1)


def pooled_7x7(images: np.ndarray):
    x = images.astype(np.float32) / 255.0
    return x.reshape(len(x), 7, 4, 7, 4).mean(axis=(2, 4)).reshape(len(x), 49)


def pooled_grid(images: np.ndarray, grid: int):
    x = images.astype(np.float32) / 255.0
    if 28 % grid != 0:
        raise ValueError(f"grid must divide 28, got {grid}")
    block = 28 // grid
    return x.reshape(len(x), grid, block, grid, block).mean(axis=(2, 4)).reshape(len(x), grid * grid)


def projection_features(images: np.ndarray):
    x = images.astype(np.float32) / 255.0
    return np.concatenate([x.mean(axis=1), x.mean(axis=2)], axis=1)


def rich_projection_features(images: np.ndarray, thresholds: np.ndarray):
    x = images.astype(np.float32) / 255.0
    parts = [
        x.mean(axis=1),
        x.mean(axis=2),
        x.std(axis=1),
        x.std(axis=2),
        x.max(axis=1),
        x.max(axis=2),
    ]
    for t in thresholds:
        active = x >= t
        parts.extend([active.mean(axis=1).astype(np.float32), active.mean(axis=2).astype(np.float32)])
    return np.concatenate(parts, axis=1)


def histogram_features(images: np.ndarray, bins: int = 16):
    x = images.astype(np.float32) / 255.0
    flat = x.reshape(len(x), -1)
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            rows.append(((flat >= lo) & (flat <= hi)).mean(axis=1))
        else:
            rows.append(((flat >= lo) & (flat < hi)).mean(axis=1))
    quantiles = np.quantile(flat, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    return np.concatenate([np.stack(rows, axis=1).astype(np.float32), quantiles.astype(np.float32)], axis=1)


def gradient_features(images: np.ndarray):
    x = images.astype(np.float32) / 255.0
    dx = np.abs(np.diff(x, axis=2))
    dy = np.abs(np.diff(x, axis=1))
    pad_dx = np.pad(dx, ((0, 0), (0, 0), (0, 1)))
    pad_dy = np.pad(dy, ((0, 0), (0, 1), (0, 0)))
    grad = np.sqrt(pad_dx * pad_dx + pad_dy * pad_dy)
    return np.concatenate(
        [
            pooled_grid((grad * 255.0).astype(np.float32), 7),
            grad.mean(axis=1),
            grad.mean(axis=2),
        ],
        axis=1,
    ).astype(np.float32)


def make_features(images: np.ndarray, mode: str):
    thresholds = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float32)
    variant = ADZE_VARIANTS.get(mode, {"population_count": 3, "higher_moments": False})
    parts = []
    names = []
    for scheme in ["horizontal", "vertical"]:
        f, n = adze_like_features_from_values(
            band_values(images, scheme, variant["population_count"]),
            thresholds,
            include_higher_moments=variant["higher_moments"],
        )
        parts.append(f)
        names.extend([f"{scheme}_{x}" for x in n])
    parts.append(image_moment_features(images))
    names.extend(["mass", "row_center", "col_center", "row_var", "col_var", "symmetry_lr", "symmetry_ud"])

    if mode in {"adze_pool", "pooled", "adze_rich", "adze_hybrid"}:
        parts.append(pooled_7x7(images))
        names.extend([f"pool7x7_{i}" for i in range(49)])
        parts.append(projection_features(images))
        names.extend([f"projection_{i}" for i in range(56)])
    if mode in {"adze_rich", "adze_hybrid"}:
        for grid in [14, 4, 2]:
            parts.append(pooled_grid(images, grid))
            names.extend([f"pool{grid}x{grid}_{i}" for i in range(grid * grid)])
        rich_thresholds = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], dtype=np.float32)
        rich_proj = rich_projection_features(images, rich_thresholds)
        parts.append(rich_proj)
        names.extend([f"rich_projection_{i}" for i in range(rich_proj.shape[1])])
        hist = histogram_features(images)
        parts.append(hist)
        names.extend([f"hist_quantile_{i}" for i in range(hist.shape[1])])
        grad = gradient_features(images)
        parts.append(grad)
        names.extend([f"gradient_{i}" for i in range(grad.shape[1])])
    if mode == "adze_hybrid":
        parts.append((images.astype(np.float32) / 255.0).reshape(len(images), -1))
        names.extend([f"pixel_{i}" for i in range(784)])
    if mode == "raw":
        parts = [(images.astype(np.float32) / 255.0).reshape(len(images), -1)]
        names = [f"pixel_{i}" for i in range(784)]
    return np.concatenate(parts, axis=1).astype(np.float32), names


def build_model(in_dim, n_classes=10, width=256, dropout=0.10, depth=3):
    layers = []
    last = in_dim
    widths = [width] * max(depth - 1, 1) + [max(width // 2, n_classes * 2)]
    for hidden in widths:
        layers.extend(
            [
                torch.nn.Linear(last, hidden),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.Dropout(dropout),
            ]
        )
        last = hidden
    layers.append(torch.nn.Linear(last, n_classes))
    return torch.nn.Sequential(
        *layers,
    )


def train_eval(X_train, y_train, X_test, y_test, args, groups=None):
    rng = np.random.default_rng(args.seed)
    if groups is None:
        idx = rng.permutation(len(X_train))
        val_n = int(round(args.val_frac * len(idx)))
        val_idx = idx[:val_n]
        tr_idx = idx[val_n:]
    else:
        groups = np.asarray(groups)
        uniq = rng.permutation(np.unique(groups))
        val_n = int(round(args.val_frac * len(uniq)))
        val_groups = set(uniq[:val_n].tolist())
        val_mask = np.fromiter((g in val_groups for g in groups), dtype=bool, count=len(groups))
        val_idx = np.flatnonzero(val_mask)
        tr_idx = np.flatnonzero(~val_mask)

    scaler = StandardScaler().fit(X_train[tr_idx])
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(X_train.shape[1], width=args.width, dropout=args.dropout, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y_train).copy(), dtype=torch.long)
    Xte = torch.as_tensor(X_test, dtype=torch.float32)

    best = {"epoch": 0, "val_acc": -1.0, "state": None}
    history = []
    for ep in range(args.epochs):
        model.train()
        tr_perm = rng.permutation(tr_idx)
        total = 0.0
        correct = 0
        seen = 0
        for start in range(0, len(tr_perm), args.batch_size):
            b = tr_perm[start : start + args.batch_size]
            xb = Xt[b].to(device)
            yb = yt[b].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(b)
            correct += int((logits.argmax(1) == yb).sum().detach().cpu())
            seen += len(b)

        if (ep + 1) % args.eval_every == 0 or ep == 0 or ep + 1 == args.epochs:
            model.eval()
            preds = []
            with torch.no_grad():
                for start in range(0, len(val_idx), args.batch_size * 4):
                    b = val_idx[start : start + args.batch_size * 4]
                    preds.append(model(Xt[b].to(device)).argmax(1).detach().cpu().numpy())
            pred_val = np.concatenate(preds)
            val_acc = accuracy_score(y_train[val_idx], pred_val)
            if val_acc > best["val_acc"]:
                best = {
                    "epoch": ep + 1,
                    "val_acc": float(val_acc),
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
            print(f"[epoch {ep+1:03d}/{args.epochs}] train_loss={total/seen:.4f} train_acc={correct/seen:.4f} val_acc={val_acc:.4f}")
        history.append({"epoch": ep + 1, "train_loss": total / seen, "train_acc": correct / seen})
        scheduler.step()

    model.load_state_dict(best["state"])
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(X_test), args.batch_size * 4):
            logits.append(model(Xte[start : start + args.batch_size * 4].to(device)).detach().cpu().numpy())
    logits = np.vstack(logits)
    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    pred = logits.argmax(axis=1)
    result = {
        "device": str(device),
        "best_epoch": int(best["epoch"]),
        "best_val_accuracy": float(best["val_acc"]),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macroF1": float(f1_score(y_test, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "labels": LABELS,
        "history": history,
        "train_rows": int(len(tr_idx)),
        "validation_rows": int(len(val_idx)),
        "validation_grouped": bool(groups is not None),
    }
    if args.save_probabilities:
        result["test_probabilities"] = probs.round(8).tolist()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument(
        "--mode",
        choices=["adze", "adze_pool", "pooled", "adze_rich", "adze_hybrid", "raw", *ADZE_VARIANTS.keys()],
        default="adze_pool",
    )
    ap.add_argument("--out", default="results/fashion_mnist_adze_stats/fashion_mnist_result.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save-probabilities", action="store_true")
    ap.add_argument("--shift-augment", choices=["none", "cardinal", "nine"], default="none")
    args = ap.parse_args()

    set_seed(args.seed)
    t0 = time.time()
    root, train_x, train_y, test_x, test_y = load_fashion_mnist(args.data_root)
    print(f"[data] {root} train={train_x.shape} test={test_x.shape}")
    train_x_aug, train_y_aug, train_groups, shifts = augment_with_shifts(train_x, train_y, args.shift_augment)
    if args.shift_augment != "none":
        print(f"[augment] shift_augment={args.shift_augment} shifts={shifts} train_rows={len(train_x_aug)}")
    X_train, feature_names = make_features(train_x_aug, args.mode)
    X_test, _ = make_features(test_x, args.mode)
    print(f"[features] mode={args.mode} dim={X_train.shape[1]}")

    metrics = train_eval(X_train, train_y_aug, X_test, test_y, args, groups=train_groups if args.shift_augment != "none" else None)
    report = {
        "purpose": "FashionMNIST fully connected DNN on ADZE-inspired image summary statistics.",
        "config": vars(args),
        "data_root": str(root),
        "base_train_rows": int(len(train_x)),
        "train_rows_after_augmentation": int(len(train_x_aug)),
        "augmentation_shifts": [[int(dy), int(dx)] for dy, dx in shifts],
        "feature_dim": int(X_train.shape[1]),
        "feature_names": feature_names,
        "elapsed_seconds": round(time.time() - t0, 2),
        **metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[result] test_accuracy={report['test_accuracy']*100:.2f}% macroF1={report['test_macroF1']:.4f} best_epoch={report['best_epoch']}")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
