# Post-hoc optimal decision threshold.
#
# The training loop reports every threshold-dependent metric (f1/precision/recall/mcc) at the
# default 0.5 probability cutoff. For an imbalanced per-residue task that cutoff is rarely the one
# that maximises MCC, so this module sweeps the validation predictions for the threshold that does,
# then reports val AND test metrics at that threshold and writes them back into `metadata.json`.
#
# The threshold is picked per fold on the validation set only and then applied to that fold's test
# set, so the test numbers stay an honest estimate (the threshold never sees test labels).
#
# The reported metric values are computed by training.py's own create_metrics (the same torchmetrics
# objects the training loop uses), with the decision threshold overridden — so the *_opt numbers are
# directly comparable to the 0.5-cutoff metrics already in the metadata. Only the threshold *search*
# is done in numpy: it is a vectorised MCC sweep whose result is identical (to ~1e-8) to torchmetrics'
# MCC, and doing it in torchmetrics would mean one full pass per candidate cutoff. torch is imported
# lazily inside metrics_at_threshold, so importing this module (e.g. for find_experiment_dirs in the
# GUI) stays cheap until metrics are actually computed.
#
# CLI:
#   cd model
#   python threshold_metrics.py                     # scan ./logs for experiments and update them
#   python threshold_metrics.py logs/my_experiment  # one or more explicit experiment dirs
#   python threshold_metrics.py --logs-root new_logs --dry-run

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# Suffix added to every threshold-tuned metric key, e.g. "val_mcc" -> "val_mcc_opt". Keeping the
# base name and only appending "_opt" means the results GUI, which namespaces metrics dynamically,
# shows them next to their 0.5-cutoff counterparts with no code change.
OPT_SUFFIX = "_opt"
THRESHOLD_KEY = "opt_threshold"

# Metrics from create_metrics that actually depend on the decision threshold. AUROC/AUPRC are
# threshold-free, so re-emitting them with an _opt suffix would just duplicate the existing values.
THRESHOLD_DEPENDENT = ("f1", "precision", "recall", "mcc")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _flatten_predictions(pred_df, ignore_label=-1):
    """Flattens a per-chunk (or per-protein) prediction frame into 1-D logit/label arrays.

    Saved predictions already drop ignored positions, but the filter is kept so the function is
    safe to point at any logit/label frame.
    """
    if len(pred_df) == 0:
        return np.empty(0), np.empty(0)

    logits = np.concatenate([np.asarray(x, dtype=float).ravel() for x in pred_df["logits"]])
    labels = np.concatenate([np.asarray(x, dtype=float).ravel() for x in pred_df["labels"]])

    keep = labels != ignore_label
    return logits[keep], labels[keep]


def optimal_mcc_threshold(logits, labels):
    """Probability threshold (`prob >= threshold`) that maximises MCC over `sigmoid(logits)`.

    Returns (threshold, best_mcc). The threshold lands on an observed probability, so applying it
    with `>=` reproduces the reported metrics exactly.
    """
    probs = sigmoid(np.asarray(logits, dtype=float))
    labels = np.asarray(labels, dtype=float)
    if probs.size == 0:
        return 0.5, 0.0

    # Sweep every distinct probability as a candidate cutoff. Sorting descending and taking cumulative
    # counts turns the whole sweep into one vectorised pass instead of one confusion matrix per point.
    order = np.argsort(-probs, kind="mergesort")
    p = probs[order]
    y = labels[order]

    total_pos = y.sum()
    total = float(len(y))
    total_neg = total - total_pos

    tp = np.cumsum(y)                       # top-k predicted positive
    fp = np.arange(1, len(y) + 1) - tp
    fn = total_pos - tp
    tn = total_neg - fp

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(denom), where=denom > 0)

    # A cutoff is only meaningful at the boundary between two distinct probabilities; ties must all
    # fall on the same side, so restrict the argmax to those boundaries.
    boundary = np.ones(len(p), dtype=bool)
    boundary[:-1] = p[1:] != p[:-1]
    boundary_idx = np.flatnonzero(boundary)

    best = boundary_idx[np.argmax(mcc[boundary_idx])]

    # Place the cutoff halfway between the lowest positive probability and the next distinct one
    # below it, rather than exactly on an observed value. A midpoint keeps the split stable when a
    # consumer (e.g. torchmetrics) recomputes sigmoid in a different float precision, so the reported
    # metrics reproduce regardless of who applies the threshold.
    lower = p[best + 1] if best + 1 < len(p) else 0.0
    threshold = (p[best] + lower) / 2.0
    return float(threshold), float(mcc[best])


def metrics_at_threshold(logits, labels, threshold, prefix="", ignore_index=-1):
    """Threshold-dependent metrics for one logit/label set, keys prefixed and `_opt`-suffixed.

    Uses training.py's create_metrics so the values come from the exact torchmetrics objects the
    training loop uses, with the decision threshold overridden. torch/training are imported lazily so
    that merely importing this module stays lightweight.
    """
    import torch
    from training import create_metrics

    _, metrics = create_metrics(ignore_index)
    metrics = metrics.clone(prefix=prefix)
    # Override the 0.5 default on every metric that thresholds (AUROC/AUPRC have no such attribute).
    for metric in metrics.values():
        if hasattr(metric, "threshold"):
            metric.threshold = float(threshold)

    lt = torch.as_tensor(np.asarray(logits, dtype=float), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(labels, dtype=float), dtype=torch.float32).int()
    metrics.update(lt, yt)

    out = {}
    for name, value in metrics.compute().items():
        base = name[len(prefix):] if prefix else name
        if base in THRESHOLD_DEPENDENT:
            out[f"{name}{OPT_SUFFIX}"] = float(value)
    return out


def _read_preds(fold_dir, split, level):
    """Loads a prediction frame or returns None if that file does not exist.

    split: "val" | "test"; level: "chunk" (per-chunk) | "stitched" (per-protein).
    """
    fold = os.path.basename(fold_dir).replace("fold_", "")
    stem = f"{split}_preds_fold_{fold}"
    name = f"{stem}_stitched.json" if level == "stitched" else f"{stem}.json"
    path = os.path.join(fold_dir, name)
    if not os.path.exists(path):
        return None
    return pd.read_json(path)


def compute_fold_threshold_metrics(fold_dir, ignore_label=-1, threshold_level="stitched"):
    """Optimal-threshold metrics for a single fold directory.

    The threshold is chosen on the validation predictions at `threshold_level` (falling back to the
    chunk level if the stitched file is absent), then applied to every available split/level.

    Returns (val_updates, test_updates) — dicts of new metric keys — or (None, None) if the fold has
    no validation predictions to tune on.
    """
    # Pick the threshold on validation, preferring the requested level.
    tune_df = _read_preds(fold_dir, "val", threshold_level)
    tune_level = threshold_level
    if tune_df is None:
        tune_df = _read_preds(fold_dir, "val", "chunk")
        tune_level = "chunk"
    if tune_df is None:
        return None, None

    tune_logits, tune_labels = _flatten_predictions(tune_df, ignore_label)
    threshold, _ = optimal_mcc_threshold(tune_logits, tune_labels)

    # Metric-key prefixes mirror the ones the training loop already writes.
    level_prefix = {"chunk": "", "stitched": "stitched_"}
    updates = {"val": {THRESHOLD_KEY: threshold}, "test": {THRESHOLD_KEY: threshold}}

    for split in ("val", "test"):
        for level, sp in level_prefix.items():
            df = _read_preds(fold_dir, split, level)
            if df is None:
                continue
            logits, labels = _flatten_predictions(df, ignore_label)
            if logits.size == 0:
                continue
            updates[split].update(
                metrics_at_threshold(logits, labels, threshold, prefix=f"{sp}{split}_",
                                     ignore_index=ignore_label))

    return updates["val"], updates["test"]


def _mean_over_folds(fold_dicts):
    """Mean of every numeric key over the folds that actually hold metrics."""
    sums, counts = {}, {}
    for fold in fold_dicts:
        if not isinstance(fold, dict):
            continue
        for k, v in fold.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def compute_experiment_threshold_metrics(exp_dir, threshold_level="stitched"):
    """Reads an experiment's metadata + predictions and returns the per-fold updates without writing.

    Returns a dict: {"val_metrics": [...], "test_metrics": [...], "n_folds": int, "updated_folds": int},
    where each list mirrors the metadata's per-fold structure with the new `_opt` keys filled in for
    folds that have predictions (empty dict otherwise). Suitable for the results GUI to call directly.
    """
    meta_path = os.path.join(exp_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    data = meta.get("data", {})
    ignore_label = (data.get("args", {}) or {}).get("ignore_label", -1)
    n_folds = max(len(data.get("val_metrics", [])), len(data.get("test_metrics", [])))
    if n_folds == 0:
        n_folds = len(glob.glob(os.path.join(exp_dir, "fold_*")))

    val_updates = [{} for _ in range(n_folds)]
    test_updates = [{} for _ in range(n_folds)]
    updated = 0

    for fold in range(n_folds):
        fold_dir = os.path.join(exp_dir, f"fold_{fold}")
        if not os.path.isdir(fold_dir):
            continue
        v, t = compute_fold_threshold_metrics(fold_dir, ignore_label, threshold_level)
        if v is None:
            continue
        val_updates[fold] = v
        test_updates[fold] = t
        updated += 1

    return {"val_metrics": val_updates, "test_metrics": test_updates,
            "n_folds": n_folds, "updated_folds": updated}


def update_experiment_metadata(exp_dir, threshold_level="stitched", dry_run=False):
    """Computes optimal-threshold metrics and merges them into the experiment's metadata.json.

    Existing per-fold metric dicts are updated in place (new keys added, non-`_opt` keys untouched),
    and the `*_metric_avg` summaries are recomputed to include the new keys. Returns the number of
    folds updated.
    """
    meta_path = os.path.join(exp_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    data = meta.setdefault("data", {})

    result = compute_experiment_threshold_metrics(exp_dir, threshold_level)
    if result["updated_folds"] == 0:
        return 0

    for key in ("val_metrics", "test_metrics"):
        folds = data.setdefault(key, [{} for _ in range(result["n_folds"])])
        # Grow the list if metadata somehow tracked fewer folds than exist on disk.
        while len(folds) < result["n_folds"]:
            folds.append({})
        for fold, update in enumerate(result[key]):
            if update and isinstance(folds[fold], dict):
                folds[fold].update(update)

    # Keep the stored averages consistent with the freshly added per-fold keys.
    data["val_metric_avg"] = _mean_over_folds(data.get("val_metrics", []))
    data["test_metric_avg"] = _mean_over_folds(data.get("test_metrics", []))

    if not dry_run:
        with open(meta_path, "w") as f:
            json.dump(meta, f, sort_keys=True, indent=4)

    return result["updated_folds"]


def find_experiment_dirs(logs_root):
    """Experiment directories under logs_root that have both a metadata.json and fold predictions."""
    dirs = []
    for meta_path in glob.glob(os.path.join(logs_root, "**", "metadata.json"), recursive=True):
        exp_dir = os.path.dirname(meta_path)
        if glob.glob(os.path.join(exp_dir, "fold_*", "val_preds_fold_*.json")):
            dirs.append(exp_dir)
    return sorted(dirs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*",
                        help="Experiment directories to process. If omitted, --logs-root is scanned.")
    parser.add_argument("--logs-root", default="logs",
                        help="Directory scanned for experiments when none are given explicitly.")
    parser.add_argument("--threshold-level", choices=["stitched", "chunk"], default="stitched",
                        help="Prediction level the threshold is tuned on (default: stitched/per-protein).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report but do not write metadata.json.")
    args = parser.parse_args()

    exp_dirs = args.experiments or find_experiment_dirs(args.logs_root)
    if not exp_dirs:
        print(f"No experiments with predictions found under {args.logs_root!r}.")
        return

    for exp_dir in exp_dirs:
        try:
            n = update_experiment_metadata(exp_dir, args.threshold_level, dry_run=args.dry_run)
        except Exception as e:  # keep going over a batch even if one run is malformed
            print(f"[skip] {exp_dir}: {e}")
            continue
        tag = " (dry-run)" if args.dry_run else ""
        print(f"[{'ok' if n else '--'}] {exp_dir}: updated {n} fold(s){tag}")


if __name__ == "__main__":
    main()
