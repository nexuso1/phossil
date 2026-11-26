# Metrics of a combination of residue-specific models.
#
# Takes runs trained on disjoint residue sets (typically one S, one T and one Y model) and evaluates
# them as a single predictor that routes every candidate residue to the model trained on it: the
# saved predictions of the components are pooled per fold and split, and the usual metrics are
# computed over the pooled set. A model trained with --residues S only ever scores S positions, so
# pooling never counts a position twice.
#
# Each component is evaluated on its own held-out data. The residue-specific splits are built
# independently, so fold i of an S run and fold i of a T run contain different proteins, and the
# pooled test set is not the test set of a jointly trained ST run on the same dataset. Fold-averaged
# metrics of the two are both cross-validation estimates, but not paired per protein.
#
# The result is written as a regular run directory (`metadata.json` only, no checkpoints or
# predictions) whose metric keys mirror the ones training writes, so the results GUI lists it next to
# the real runs:
#   - `val_*` / `test_*` (chunk level) and `stitched_val_*` / `stitched_test_*` (per protein), all
#     computed by training.py's own compute_stitched_metrics at the 0.5 cutoff;
#   - `*_opt` threshold-dependent metrics, where every component is cut at its own validation
#     MCC-optimal threshold (see threshold_metrics.py), stored as `opt_threshold_<residues>`.
#
# The component args are merged into the combined args: values shared by all components are kept,
# differing ones become a {residues: value} dict, `residues` is the union and `combined_from` points
# at the component runs.
#
# CLI (from model/):
#   python combine_residue_models.py --auto --list                 # what would be combined
#   python combine_residue_models.py --auto                        # ST and STY for every S/T/Y group
#   python combine_residue_models.py --auto --filter 'deeppsp|dbptm_chunked_max$'
#   python combine_residue_models.py logs/x_S_650M_dbptm_chunked_max logs/x_T_650M_dbptm_chunked_max
#
# --test-split pools other test prediction files than the runs' own, e.g. the ones
# evaluate_checkpoints.py writes for a residue-specific run evaluated on a joint test set.
#
# Auto mode groups runs whose names differ only in the residue token of the naming convention
# (`<prefix>_<residues>_<type>_<dataset><variant>`), e.g. finetuning_drop_{S,T,Y}_2_650M_deeppsp_chunked.
# Output goes to <logs-root>/combined/<name with the residue token replaced by e.g. S+T>.

import argparse
import ast
import glob
import json
import os
import re

import numpy as np
import pandas as pd

import threshold_metrics as tm

RESIDUE_ORDER = "STY"
COMBINED_KEY = "combined_from"
# Args that differ between the components of a valid combination. A difference in anything else
# (learning rate, backbone, ...) means the runs are probably not meant to be combined, and is warned about.
EXPECTED_ARG_DIFFS = {"residues", "dataset_path", "prot_info_path", "o", "logdir", "checkpoint_path"}
LEVEL_PREFIX = {"chunk": "", "stitched": "stitched_"}


def parse_residues(value):
    """Residues as a list. Metadata stores them either as 'ST' or as "['S', 'T']"."""
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    return list(value)


def residue_token(residues):
    return "".join(sorted(residues, key=RESIDUE_ORDER.index))


def load_run(run_dir):
    run_dir = os.path.normpath(run_dir)
    with open(os.path.join(run_dir, "metadata.json")) as f:
        data = json.load(f).get("data", {})
    args = data.get("args", {}) or {}
    n_folds = max(len(data.get("val_metrics") or []), len(data.get("test_metrics") or []),
                  len(glob.glob(os.path.join(run_dir, "fold_*"))))

    return {
        "dir": run_dir,
        "name": os.path.basename(run_dir),
        "args": args,
        # Same default as the training parser
        "residues": parse_residues(args.get("residues", "['S', 'T', 'Y']")),
        "n_folds": n_folds,
        "is_combined": COMBINED_KEY in args,
    }


def name_template(name, token):
    """Run name with its residue token replaced by '{}', or None if the token is not exactly once in it."""
    tokens = name.split("_")
    hits = [i for i, t in enumerate(tokens) if t == token]
    if len(hits) != 1:
        return None
    tokens[hits[0]] = "{}"
    return "_".join(tokens)


def combined_name(runs):
    """Name of the combination, e.g. x_S+T_650M_dbptm, if all components share a name template."""
    templates = {name_template(r["name"], residue_token(r["residues"])) for r in runs}
    if len(templates) != 1 or None in templates:
        return None
    return templates.pop().replace("{}", "+".join(residue_token(r["residues"]) for r in runs))


def merge_args(runs, out_dir):
    """Combined args, plus the names of args that differ between components unexpectedly."""
    keys = sorted(set().union(*(r["args"].keys() for r in runs)))
    merged, unexpected = {}, []
    for key in keys:
        values = [r["args"].get(key) for r in runs]
        if all(v == values[0] for v in values):
            merged[key] = values[0]
        else:
            merged[key] = {residue_token(r["residues"]): v for r, v in zip(runs, values)}
            if key not in EXPECTED_ARG_DIFFS:
                unexpected.append(key)

    merged["residues"] = residue_token([res for r in runs for res in r["residues"]])
    # Relative to the combined run, so the reference survives moving the whole logs tree
    merged[COMBINED_KEY] = {residue_token(r["residues"]): os.path.relpath(r["dir"], out_dir) for r in runs}
    return merged, unexpected


def _fold_dirs(runs, fold):
    return [os.path.join(r["dir"], f"fold_{fold}") for r in runs]


def _read_all(fold_dirs, split, level):
    """Prediction frames of every component, or None if any of them is missing."""
    frames = [tm.read_preds(d, split, level) if os.path.isdir(d) else None for d in fold_dirs]
    return None if any(f is None for f in frames) else frames


def _threshold_shifted(frames, thresholds, ignore_label):
    """Pooled logits shifted so that every component's own threshold lands on probability 0.5.

    prob >= t is the same as logit >= logit(t), so subtracting logit(t) per component lets a single
    0.5 cutoff apply each component's threshold. The shift changes how components rank against each
    other, which is why only threshold-dependent metrics are computed from it.
    """
    logits, labels = [], []
    for frame, threshold in zip(frames, thresholds):
        component_logits, component_labels = tm.flatten_predictions(frame, ignore_label)
        threshold = np.clip(threshold, 1e-12, 1 - 1e-12)
        logits.append(component_logits - np.log(threshold / (1 - threshold)))
        labels.append(component_labels)
    return np.concatenate(logits), np.concatenate(labels)


def combine_fold(runs, fold, ignore_label=-1, threshold_level="stitched", test_split="test"):
    """Pooled (val_metrics, test_metrics) of one fold. A split/level is left out unless every
    component has predictions for it, and a fold without test metrics is left out entirely.

    test_split names the prediction files of the test split (<test_split>_preds_fold_N*.json), e.g.
    the ones evaluate_checkpoints.py writes; the metrics keep their usual test_* names."""
    from training import compute_stitched_metrics

    fold_dirs = _fold_dirs(runs, fold)
    thresholds = [tm.tune_fold_threshold(d, ignore_label, threshold_level) if os.path.isdir(d) else None
                  for d in fold_dirs]
    tuned = all(t is not None for t in thresholds)

    results = {}
    for split in ("val", "test"):
        metrics = {}
        for level, level_prefix in LEVEL_PREFIX.items():
            frames = _read_all(fold_dirs, test_split if split == "test" else split, level)
            if frames is None:
                continue
            prefix = f"{level_prefix}{split}_"
            metrics.update(compute_stitched_metrics(pd.concat(frames, ignore_index=True), ignore_label,
                                                    prefix=prefix))
            if tuned:
                logits, labels = _threshold_shifted(frames, thresholds, ignore_label)
                metrics.update(tm.metrics_at_threshold(logits, labels, 0.5, prefix=prefix,
                                                       ignore_index=ignore_label))

        if metrics and tuned:
            for run, threshold in zip(runs, thresholds):
                metrics[f"{tm.THRESHOLD_KEY}_{residue_token(run['residues'])}"] = threshold
        results[split] = metrics

    if not results["test"]:
        return {}, {}
    return results["val"], results["test"]


def combine_runs(runs, out_dir, threshold_level="stitched", test_split="test"):
    """Combined metadata of the given component runs (not written). Returns (metadata, unexpected_args)."""
    runs = sorted(runs, key=lambda r: RESIDUE_ORDER.index(residue_token(r["residues"])[0]))

    seen = set()
    for run in runs:
        if run["is_combined"]:
            raise ValueError(f"{run['name']} is itself a combination")
        overlap = seen & set(run["residues"])
        if overlap:
            raise ValueError(f"residues {sorted(overlap)} are predicted by more than one component")
        seen |= set(run["residues"])

    ignore_label = runs[0]["args"].get("ignore_label", -1)
    n_folds = max(r["n_folds"] for r in runs)

    val_folds, test_folds = [], []
    for fold in range(n_folds):
        val, test = combine_fold(runs, fold, ignore_label, threshold_level, test_split)
        val_folds.append(val)
        test_folds.append(test)

    args, unexpected = merge_args(runs, out_dir)
    if test_split != "test":
        args["test_split"] = test_split
    metadata = {"data": {
        "args": args,
        "val_metrics": val_folds,
        "test_metrics": test_folds,
        "fold_finished": [bool(v or t) for v, t in zip(val_folds, test_folds)],
        "val_metric_avg": tm.mean_over_folds(val_folds),
        "test_metric_avg": tm.mean_over_folds(test_folds),
    }}
    return metadata, unexpected


def write_combination(runs, out_dir, threshold_level="stitched", dry_run=False, test_split="test"):
    meta_path = os.path.join(out_dir, "metadata.json")
    # Never overwrite a real run that happens to sit at the output path
    if os.path.exists(meta_path) and not load_run(out_dir)["is_combined"]:
        raise FileExistsError(f"{meta_path} exists and is not a combined run")

    metadata, unexpected = combine_runs(runs, out_dir, threshold_level, test_split)
    if not any(metadata["data"]["fold_finished"]):
        raise ValueError("no fold has saved predictions from every component")
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=4)
    return metadata, unexpected


def find_single_residue_groups(logs_root, name_filter=None):
    """{name template: {residue: run}} over the single-residue runs directly under logs_root."""
    groups = {}
    for meta_path in sorted(glob.glob(os.path.join(logs_root, "*", "metadata.json"))):
        run_dir = os.path.dirname(meta_path)
        # Filter on the name before loading, so runs outside the filter are never read
        if name_filter and not re.search(name_filter, os.path.basename(run_dir)):
            continue
        try:
            run = load_run(run_dir)
        except (OSError, ValueError, SyntaxError, TypeError) as e:
            print(f"[skip] {meta_path}: {e}")
            continue
        if run["is_combined"] or len(run["residues"]) != 1:
            continue
        template = name_template(run["name"], run["residues"][0])
        if template is not None:
            groups.setdefault(template, {})[run["residues"][0]] = run
    return groups


def _summary(metadata):
    data = metadata["data"]
    done = sum(data["fold_finished"])
    avg = data["test_metric_avg"]
    parts = [f"{done}/{len(data['fold_finished'])} folds"]
    for key in ("stitched_test_mcc", "stitched_test_mcc_opt", "stitched_test_auprc"):
        if key in avg:
            parts.append(f"{key}={avg[key]:.4f}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Pools the predictions of residue-specific runs and computes the metrics of their combination.")
    parser.add_argument("runs", nargs="*", help="Run directories to combine (disjoint residue sets).")
    parser.add_argument("--auto", action="store_true",
                        help="Group the single-residue runs under --logs-root by name and combine each group.")
    parser.add_argument("--logs-root", default="logs", help="Directory scanned in --auto mode.")
    parser.add_argument("--out-root", default=None, help="Where combined runs are written (default: <logs-root>/combined).")
    parser.add_argument("-o", default=None, help="Output run name when combining explicit runs.")
    parser.add_argument("--combos", default="ST,STY",
                        help="Comma-separated residue combinations built in --auto mode (default: ST,STY).")
    parser.add_argument("--filter", default=None, help="Regex a run name must match to be used in --auto mode.")
    parser.add_argument("--threshold-level", choices=["stitched", "chunk"], default="stitched",
                        help="Prediction level the per-component thresholds are tuned on.")
    parser.add_argument("--test-split", default="test",
                        help="Prefix of the test prediction files to pool, e.g. test_ST_chunked for the ones "
                             "evaluate_checkpoints.py writes. A non-default split is appended to the output name.")
    parser.add_argument("--list", action="store_true", help="Only print what would be combined.")
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write metadata.json.")
    args = parser.parse_args()

    out_root = args.out_root or os.path.join(args.logs_root, "combined")

    # (output name, component runs) pairs
    jobs = []
    if args.auto:
        combos = [residue_token(c.strip()) for c in args.combos.split(",") if c.strip()]
        for template, by_residue in find_single_residue_groups(args.logs_root, args.filter).items():
            for combo in combos:
                if all(res in by_residue for res in combo):
                    runs = [by_residue[res] for res in combo]
                    jobs.append((template.replace("{}", "+".join(combo)), runs))
    elif args.runs:
        runs = [load_run(d) for d in args.runs]
        name = args.o or combined_name(runs)
        if name is None:
            parser.error("the run names do not share a template, pass the output name with -o")
        jobs.append((name, runs))
    else:
        parser.error("pass run directories to combine, or --auto")

    if not jobs:
        print("Nothing to combine.")
        return

    for name, runs in jobs:
        if args.test_split != "test":
            name = f"{name}_{args.test_split}"
        components = ", ".join(r["name"] for r in runs)
        if args.list:
            print(f"{name} <- {components}")
            continue

        out_dir = os.path.join(out_root, name)
        try:
            metadata, unexpected = write_combination(runs, out_dir, args.threshold_level, args.dry_run,
                                                     args.test_split)
        except (OSError, ValueError) as e:  # keep going over a batch even if one group is malformed
            print(f"[skip] {name}: {e}")
            continue

        tag = " (dry-run)" if args.dry_run else ""
        print(f"[ok] {out_dir}{tag}: {_summary(metadata)}")
        if unexpected:
            print(f"     warning: components differ in {', '.join(unexpected)} ({components})")


if __name__ == "__main__":
    main()
