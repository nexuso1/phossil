# Statistical significance of the difference between two runs on the same cross-validation splits.
#
# Both runs must have been trained on the same splits file, so that every protein lands in the same
# test fold of both and every candidate site is scored by both models. Their stitched (per-protein)
# test predictions are then paired site by site, and three tests are reported for every metric:
#
#   - Paired bootstrap over proteins (scipy.stats.bootstrap). Sites within a protein are correlated,
#     proteins are close to independent after homology reduction, so the resampled unit is the
#     protein. Every fold is passed as its own sample, so proteins are drawn with replacement within
#     each fold (keeping the fold sizes); the draw becomes per-site sample weights, the same for both
#     runs. Gives a percentile CI and a two-sided bootstrap p-value.
#   - Paired permutation test over proteins (scipy.stats.permutation_test, permutation_type="samples").
#     Under the null hypothesis the two runs' predictions of one protein are exchangeable, so each
#     protein's predictions are swapped between the runs with probability 1/2. For AUROC/AUPRC the
#     scores are first replaced by within-fold quantile ranks of each run: that leaves each run's own
#     metric unchanged (they are rank-based) but puts the two runs on one scale, so a swap does not mix
#     two differently calibrated logit scales.
#   - Corrected resampled t-test (Nadeau & Bengio, 2003) on the per-fold differences. CV folds share
#     most of their training data, so the plain paired t-test (scipy.stats.ttest_rel, kept in the JSON
#     output for reference) underestimates the variance; the correction inflates it by
#     (1/k + n_test/n_train). n_test/n_train is 1/(K - 1) for K-fold CV: the proteins outside the
#     test fold, train and dev together, count as training data (dev picks the checkpoint and the
#     threshold). With 5 folds it has little power, report it next to the bootstrap rather than instead
#     of it. --no-t-test leaves it out, for "folds" that are not CV folds (the cells of
#     scripts/cross_representative_significance.py).
#
# Metrics are sklearn's average_precision_score / roc_auc_score / matthews_corrcoef. The statistic is
# the mean over folds of the per-fold metric, i.e. the same `stitched_test_*` fold average the
# metadata reports, not a metric over the pooled predictions of all folds (which would rank sites of
# differently calibrated fold models against each other). The per-fold values are checked against the
# metadata where it has them.
#
# MCC needs a cutoff: --threshold 0.5 matches the training loop, --threshold opt uses each fold's
# validation MCC-optimal threshold (threshold_metrics.py), tuned per run. The test labels never pick it.
# The false positive / negative rates (--metrics fpr fnr, not tested by default) use the same cutoff.
#
# --residues restricts both runs to the sites on those residues, looked up in the unchunked --info
# file, e.g. to compare a joint ST model on its S sites with a model trained on S alone. To be
# paired, that model must have been trained on splits with the joint model's folds and proteins (the
# _from_ST / _from_STY splits of scripts/derive_residue_splits.py). With --threshold opt a joint
# model keeps its own validation threshold, tuned over all its residues together.
#
# The bootstrap and the permutation test condition on the trained models: they measure how much the
# difference would vary over another sample of proteins, not over another training seed.
#
# Results are printed and written to ../data/statistical_tests/<name>.txt (the printed table) and
# <name>.json (everything, including the per-fold values), <name> derived from the runs and options.
#
# CLI (from model/):
#   python significance_test.py logs/run_a logs/run_b                     # difference is b - a
#   python significance_test.py logs/run_a logs/run_b --threshold opt --n-boot 5000 --name my_test
#   python significance_test.py logs/run_a logs/run_b --folds 0           # one fold, no t-test
#   python significance_test.py logs/joint_ST logs/single_S_from_ST --residues S
#   python significance_test.py logs/run_a logs/run_b --split other_test  # evaluate_checkpoints.py files

import argparse
import glob
import json
import multiprocessing
import os

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

import threshold_metrics as tm
from combine_residue_models import parse_residues, residue_token

METRICS = ("auprc", "auroc", "mcc")
# Error rates at the MCC cutoff, only computed when asked for with --metrics
RATE_METRICS = ("fpr", "fnr")


def load_sequences(info_path):
    """{protein id: whole-protein sequence} from an unchunked info file. The dbPTM info file repeats
    a few hundred ids, always with the same sequence, so the first one is kept."""
    info = pd.read_json(info_path).dropna(subset=["id", "sequence"]).drop_duplicates("id")
    return dict(zip(info["id"].astype(str), info["sequence"]))


def load_fold(fold_dir, split, ignore_label, residues=None, sequences=None):
    """Per-site arrays of one fold's stitched predictions, sorted by (protein, position).

    With `residues`, only the sites on those residues are kept; `sequences` (see load_sequences)
    tells which residue a whole-protein position is on.

    Returns a dict with `keys` (protein id, position) and the per-site `protein` index, `y`, `logit`,
    or None if the fold has no stitched prediction file for this split.
    """
    df = tm.read_preds(fold_dir, split, "stitched")
    if df is None:
        return None

    rows = []
    for pid, logits, labels, positions in zip(df["id"], df["logits"], df["labels"],
                                              df["sequence_indices"]):
        pid = str(pid)
        if residues is not None and pid not in sequences:
            raise ValueError(f"{fold_dir}: protein {pid} is not in the --info file")
        for logit, label, pos in zip(logits, labels, positions):
            if label == ignore_label:
                continue
            if residues is not None:
                residue = sequences[pid][int(pos)] if int(pos) < len(sequences[pid]) else None
                if residue not in ("S", "T", "Y"):
                    raise ValueError(f"{fold_dir}: {pid} position {pos} is on {residue!r}, not "
                                     f"S/T/Y - the --info file does not match these predictions")
                if residue not in residues:
                    continue
            rows.append((pid, int(pos), float(logit), int(label)))
    rows.sort(key=lambda r: (r[0], r[1]))

    keys = [(r[0], r[1]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{fold_dir}: duplicate (protein, position) in the stitched predictions")

    _, protein = np.unique([r[0] for r in rows], return_inverse=True)
    return {"keys": keys, "protein": protein,
            "logit": np.array([r[2] for r in rows]), "y": np.array([r[3] for r in rows])}


def load_run(run_dir):
    with open(os.path.join(run_dir, "metadata.json")) as f:
        data = json.load(f).get("data", {})
    return data


def fold_numbers(run_dir):
    return {int(os.path.basename(d).replace("fold_", ""))
            for d in glob.glob(os.path.join(run_dir, "fold_*")) if os.path.isdir(d)}


def fold_threshold(fold_dir, threshold, ignore_label):
    """Probability cutoff of one fold: a fixed float, or the validation MCC-optimal one for "opt"."""
    if threshold != "opt":
        return float(threshold)
    tuned = tm.tune_fold_threshold(fold_dir, ignore_label)
    if tuned is None:
        raise ValueError(f"{fold_dir}: --threshold opt needs validation predictions to tune on")
    return tuned


# Label and prediction of the four cells of a binary confusion matrix. MCC depends on the sites only
# through how many fall in each cell, so sklearn's MCC of these four pseudo-sites, weighted by the
# cell counts, is exactly the MCC of the sites, without a confusion matrix of every site per resample.
CELL_LABEL = np.array([0, 0, 1, 1])
CELL_PRED = np.array([0, 1, 0, 1])


def score_metrics(metrics, y, score, cells, w=None):
    """{metric: value} of one run.

    `y` and `score` are the sites (weighted by `w` for AUROC/AUPRC) in ascending score order, which
    sklearn's stable sort then has next to nothing to do on; `cells` the weighted site counts per
    confusion matrix cell (CELL_LABEL, CELL_PRED) for MCC. The inputs are checked when loaded, so
    sklearn's per-call validation is skipped.
    """
    funcs = {"auprc": lambda: average_precision_score(y, score, sample_weight=w),
             "auroc": lambda: roc_auc_score(y, score, sample_weight=w),
             "mcc": lambda: matthews_corrcoef(CELL_LABEL, CELL_PRED, sample_weight=cells),
             # cells are (TN, FP, FN, TP), see CELL_LABEL / CELL_PRED
             "fpr": lambda: cells[1] / (cells[0] + cells[1]),
             "fnr": lambda: cells[2] / (cells[2] + cells[3])}
    with sklearn.config_context(assume_finite=True, skip_parameter_validation=True):
        return {m: funcs[m]() for m in metrics}


class PairedFold:
    """One fold of two runs aligned site by site."""

    def __init__(self, fold_a, fold_b, threshold_a, threshold_b, metrics):
        self.metrics_names = metrics
        self.protein = fold_a["protein"]
        self.n_proteins = self.protein.max() + 1
        self.y = fold_a["y"]

        # Quantile ranks keep each run's rank metrics as they are and put both on the same scale for
        # the permutation test.
        rank = [stats.rankdata(f["logit"]) / len(self.y) for f in (fold_a, fold_b)]
        # torchmetrics, which training uses, predicts positive for probabilities above the cutoff.
        pred = [tm.sigmoid(f["logit"]) > t for f, t in ((fold_a, threshold_a), (fold_b, threshold_b))]

        # Each run's sites sorted by its score once. A bootstrap only reweights them and a permutation
        # concatenates two such sorted runs, so sklearn's stable sort (timsort) only has to merge.
        order = [np.argsort(r, kind="stable") for r in rank]
        self.sorted_rank = [r[o] for r, o in zip(rank, order)]
        self.sorted_y = [self.y[o] for o in order]
        self.sorted_protein = [self.protein[o] for o in order]
        # (protein, confusion matrix cell) site counts of each run, see CELL_LABEL.
        self.cells = [np.bincount(self.protein * 4 + 2 * self.y + p.astype(int),
                                  minlength=4 * self.n_proteins).reshape(-1, 4) for p in pred]

    def metrics(self, run, protein_weights=None):
        """Metrics of one run (0 or 1), optionally under per-protein bootstrap weights."""
        if protein_weights is None:
            return score_metrics(self.metrics_names, self.sorted_y[run], self.sorted_rank[run],
                                 self.cells[run].sum(0))
        w = protein_weights[self.sorted_protein[run]]
        # sklearn drops zero-weight sites itself, but only after checking all of them
        drawn = w > 0
        return score_metrics(self.metrics_names, self.sorted_y[run][drawn],
                             self.sorted_rank[run][drawn], protein_weights @ self.cells[run], w[drawn])

    def protein_weights(self, protein_sample):
        """Bootstrap weight of every protein: how often a draw of protein indices picked it."""
        return np.bincount(protein_sample, minlength=self.n_proteins)

    def swapped_metrics(self, swap):
        """Metrics of both runs after swapping their predictions on the proteins where `swap` is set."""
        out = []
        for own, other in ((0, 1), (1, 0)):
            kept = ~swap[self.sorted_protein[own]]
            taken = swap[self.sorted_protein[other]]
            score = np.concatenate((self.sorted_rank[own][kept], self.sorted_rank[other][taken]))
            y = np.concatenate((self.sorted_y[own][kept], self.sorted_y[other][taken]))
            cells = self.cells[own][~swap].sum(0) + self.cells[other][swap].sum(0)
            out.append(score_metrics(self.metrics_names, y, score, cells))
        return out


def pair_folds(run_a, run_b, split, threshold, metrics, only_folds=None, residues=None,
               sequences=None):
    """Loads and aligns the folds both runs have predictions for, restricted to the sites on
    `residues` if given. Fails if the runs' sites differ."""
    meta_a, meta_b = load_run(run_a), load_run(run_b)
    ignore_label = (meta_a.get("args") or {}).get("ignore_label", -1)

    folds = sorted(only_folds if only_folds is not None else fold_numbers(run_a) | fold_numbers(run_b))
    paired, used = [], []
    for fold in folds:
        dir_a, dir_b = (os.path.join(r, f"fold_{fold}") for r in (run_a, run_b))
        fa, fb = (load_fold(d, split, ignore_label, residues, sequences) if os.path.isdir(d) else None
                  for d in (dir_a, dir_b))
        if fa is None or fb is None:
            print(f"[warn] fold {fold}: no stitched {split} predictions in "
                  f"{' and '.join(r for r, f in ((run_a, fa), (run_b, fb)) if f is None)}, skipped")
            continue
        if not fa["keys"] or not fb["keys"]:
            raise ValueError(f"fold {fold}: no {split} sites on residues {residue_token(residues)} "
                             f"in {run_a if not fa['keys'] else run_b}")

        if fa["keys"] != fb["keys"]:
            only_a = len(set(fa["keys"]) - set(fb["keys"]))
            only_b = len(set(fb["keys"]) - set(fa["keys"]))
            raise ValueError(
                f"fold {fold}: the runs score different sites ({only_a} only in A, {only_b} only in B). "
                f"Were they trained on the same splits file?\n"
                f"  A: {(meta_a.get('args') or {}).get('dataset_path')}\n"
                f"  B: {(meta_b.get('args') or {}).get('dataset_path')}")
        if not np.array_equal(fa["y"], fb["y"]):
            raise ValueError(f"fold {fold}: the runs disagree on the labels of the same sites")

        paired.append(PairedFold(fa, fb, fold_threshold(dir_a, threshold, ignore_label),
                                 fold_threshold(dir_b, threshold, ignore_label), metrics))
        used.append(fold)

    if not paired:
        raise ValueError(f"no fold has stitched {split} predictions in both runs")
    return paired, used, (meta_a, meta_b)


def check_against_metadata(fold_metrics, folds, metas, split, threshold, residues=None):
    """Warnings where the recomputed per-fold metrics disagree with the ones training stored.

    A run trained on residues outside `residues` is not checked: its stored metrics cover sites
    the residue filter removed.
    """
    warnings = []
    if split != "test":
        return warnings
    # The metadata only holds MCC at the 0.5 cutoff and at the validation-optimal one.
    if threshold == "opt":
        mcc_key = f"stitched_test_mcc{tm.OPT_SUFFIX}"
    else:
        mcc_key = "stitched_test_mcc" if float(threshold) == 0.5 else None
    for run, (per_fold, meta) in enumerate(zip(fold_metrics, metas)):
        trained = parse_residues((meta.get("args") or {}).get("residues", "STY"))
        if residues is not None and not set(trained) <= set(residues):
            continue
        stored = meta.get("test_metrics") or []
        for values, fold in zip(per_fold, folds):
            fold_meta = stored[fold] if fold < len(stored) and isinstance(stored[fold], dict) else {}
            for m in values:
                key = mcc_key if m == "mcc" else f"stitched_test_{m}"
                if key in fold_meta and abs(fold_meta[key] - values[m]) > 1e-3:
                    warnings.append(f"[warn] run {'AB'[run]} fold {fold}: {key} is {values[m]:.4f}, "
                                    f"metadata has {fold_meta[key]:.4f}")
    return warnings


def corrected_t_test(diffs, test_ratio):
    """Nadeau-Bengio corrected resampled t-test of per-fold differences. Returns (t, two-sided p),
    both None for a single fold, which has no fold-to-fold variance to test against.
    """
    k = len(diffs)
    if k < 2:
        return None, None
    mean, var = np.mean(diffs), np.var(diffs, ddof=1)
    if var == 0:
        return (np.inf if mean != 0 else 0.0), (0.0 if mean != 0 else 1.0)
    t = mean / np.sqrt((1 / k + test_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


# State of the resampling statistics. Worker processes are forked after it is set, so they inherit the
# paired folds and only the resamples themselves (protein indices, swap masks) are sent to them.
_STATE = {}


def _boot_row(protein_samples):
    """Mean over folds of B - A under one bootstrap draw: one array of protein indices per fold."""
    paired, metrics = _STATE["paired"], _STATE["metrics"]
    diff = np.zeros(len(metrics))
    for f, sample in zip(paired, protein_samples):
        w = f.protein_weights(sample)
        va, vb = f.metrics(0, w), f.metrics(1, w)
        diff += [vb[m] - va[m] for m in metrics]
    return diff / len(paired)


def _perm_row(row):
    """Mean over folds of B - A under one permutation, see run_tests for the encoding of `row`."""
    paired, metrics, offsets = _STATE["paired"], _STATE["metrics"], _STATE["offsets"]
    swap = row >= offsets[-1]
    diff = np.zeros(len(metrics))
    for f, lo, hi in zip(paired, offsets[:-1], offsets[1:]):
        va, vb = f.swapped_metrics(swap[lo:hi])
        diff += [vb[m] - va[m] for m in metrics]
    return diff / len(paired)


def run_tests(paired, metrics, n_boot, n_perm, ci, seed, n_cv_folds, workers=1, save_boot_dist=False,
              t_test=True):
    rng = np.random.default_rng(seed)
    k = len(paired)

    observed = [[f.metrics(run) for f in paired] for run in (0, 1)]
    mean_a = np.array([np.mean([fm[m] for fm in observed[0]]) for m in metrics])
    mean_b = np.array([np.mean([fm[m] for fm in observed[1]]) for m in metrics])
    delta = mean_b - mean_a

    # Permutation: observation i is protein i of all folds concatenated. It holds the value i in the
    # A sample and i + n in the B sample, so after scipy swaps a pair, `x >= n` marks the proteins whose
    # predictions changed sides.
    offsets = np.cumsum([0] + [f.n_proteins for f in paired])
    n = offsets[-1]

    # scipy draws every resample in this process and passes them in batches (vectorized=True), so the
    # results do not depend on how many workers evaluate the rows of a batch.
    _STATE.update(paired=paired, metrics=metrics, offsets=offsets)
    pool = multiprocessing.get_context("fork").Pool(workers) if workers > 1 else None
    rows_map = pool.map if pool else lambda fn, rows: list(map(fn, rows))

    # Bootstrap: one sample of protein indices per fold, so scipy resamples within folds. The same
    # draw weights both runs, which makes the bootstrap paired. A batch holds one (resample, protein)
    # array per fold; scipy expects the metric axis first and the resample axis last.
    def boot_statistic(*protein_samples, axis=-1):
        if np.ndim(protein_samples[0]) == 1:
            return _boot_row(protein_samples)
        return np.array(rows_map(_boot_row, list(zip(*protein_samples)))).T

    # scipy's permutation_test expects the resample axis first and the metric axis last.
    def perm_statistic(x, y, axis=-1):
        if np.ndim(x) == 1:
            return _perm_row(x)
        return np.array(rows_map(_perm_row, list(x)))

    try:
        boot = stats.bootstrap(tuple(np.arange(f.n_proteins) for f in paired), boot_statistic,
                               n_resamples=n_boot, confidence_level=ci, method="percentile",
                               vectorized=True, rng=rng)
        perm = stats.permutation_test((np.arange(n), np.arange(n) + n), perm_statistic,
                                      permutation_type="samples", n_resamples=n_perm,
                                      vectorized=True, alternative="two-sided", rng=rng)
    finally:
        if pool:
            pool.close()
            pool.join()
    boot_dist = boot.bootstrap_distribution  # (metric, resample)

    # Every fold trains on the other K - 1 folds (train and dev together), which are of equal size,
    # also when only some folds are tested.
    n_test = np.array([f.n_proteins for f in paired])
    test_ratio = 1 / (n_cv_folds - 1) if t_test else None

    results = {}
    for i, m in enumerate(metrics):
        fold_diffs = np.array([b[m] - a[m] for a, b in zip(*observed)])
        t, t_p = corrected_t_test(fold_diffs, test_ratio) if t_test else (None, None)
        dist = boot_dist[i]
        results[m] = {
            "a": float(mean_a[i]), "b": float(mean_b[i]), "delta": float(delta[i]),
            "ci": [float(boot.confidence_interval.low[i]), float(boot.confidence_interval.high[i])],
            "bootstrap_p": float(min(1.0, 2 * min(np.mean(dist <= 0), np.mean(dist >= 0)))),
            "permutation_p": float(perm.pvalue[i]),
            "fold_a": [float(fm[m]) for fm in observed[0]],
            "fold_b": [float(fm[m]) for fm in observed[1]],
            "fold_delta": fold_diffs.tolist(),
            "folds_b_better": int(np.sum(fold_diffs > 0)),
            "corrected_t": t, "corrected_t_p": t_p,
            "paired_t_p": float(stats.ttest_rel([fm[m] for fm in observed[1]],
                                                [fm[m] for fm in observed[0]]).pvalue)
            if t_test and k > 1 else None,
        }
        if save_boot_dist:
            results[m]["bootstrap_distribution"] = dist.tolist()
    return results, observed, {"test_ratio": test_ratio,
                               "n_proteins": n_test.tolist(),
                               "n_sites": [len(f.y) for f in paired]}


def format_results(run_a, run_b, results, folds, info, args, warnings):
    """The results table as text, as printed and saved to the .txt file."""
    lines = list(warnings)
    lines.append(f"A: {run_a}\nB: {run_b}")
    residues = "" if args.residues is None else f"residues {residue_token(args.residues)}, "
    lines.append(f"folds {folds}, {residues}{sum(info['n_proteins'])} proteins, "
                 f"{sum(info['n_sites'])} sites, "
                 f"MCC cutoff {args.threshold}, {args.n_boot} bootstrap / {args.n_perm} permutation "
                 f"rounds\n")
    pct = f"{args.ci:.0%}"
    lines.append(f"{'metric':<7} {'A':>7} {'B':>7} {'B-A':>8}  {pct + ' CI':>19} {'boot p':>8} "
                 f"{'perm p':>8}  {'B>A':>4} {'corr. t p':>9}")
    for m, r in results.items():
        t_p = "n/a" if r["corrected_t_p"] is None else f"{r['corrected_t_p']:.4f}"
        lines.append(f"{m:<7} {r['a']:7.4f} {r['b']:7.4f} {r['delta']:+8.4f}  "
                     f"[{r['ci'][0]:+8.4f}, {r['ci'][1]:+8.4f}] {r['bootstrap_p']:8.4f} "
                     f"{r['permutation_p']:8.4f}  {r['folds_b_better']:>2}/{len(folds)} {t_p:>9}")
    lines.append("\nper-fold B-A:")
    for m, r in results.items():
        lines.append(f"  {m:<7} " + " ".join(f"{d:+.4f}" for d in r["fold_delta"]))
    return "\n".join(lines) + "\n"


def default_name(args):
    """`<run B>_vs_<run A>_<cutoff>`, plus the folds, metrics and split where they are not the defaults,
    so that differently configured tests of the same pair do not overwrite each other.
    """
    name = (f"{os.path.basename(os.path.normpath(args.run_b))}_vs_"
            f"{os.path.basename(os.path.normpath(args.run_a))}_"
            f"{'opt' if args.threshold == 'opt' else 'thr' + args.threshold}")
    if args.folds is not None:
        name += "_fold" + "-".join(str(f) for f in sorted(args.folds))
    if sorted(args.metrics) != sorted(METRICS):
        name += "_" + "-".join(args.metrics)
    if args.residues is not None:
        name += f"_res{residue_token(args.residues)}"
    if args.split != "test":
        name += f"_{args.split}"
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", help="Run directory of method A (the baseline).")
    parser.add_argument("run_b", help="Run directory of method B. Differences are B - A.")
    parser.add_argument("--metrics", nargs="+", choices=METRICS + RATE_METRICS, default=list(METRICS),
                        help="Metrics to test (default: auprc auroc mcc). fpr / fnr are the false "
                             "positive / negative rates at the --threshold cutoff.")
    parser.add_argument("--threshold", default="0.5",
                        help="MCC probability cutoff: a float, or 'opt' for each fold's validation "
                             "MCC-optimal threshold (default: 0.5, as in training).")
    parser.add_argument("--folds", type=int, nargs="+",
                        help="Only test these folds (default: every fold both runs have predictions "
                             "for). With a single fold there is no corrected t-test.")
    parser.add_argument("--residues", type=parse_residues,
                        help="Only test the sites on these residues, e.g. S or ST (default: every "
                             "site both runs score). Needs --info.")
    parser.add_argument("--info", default="../data/dbptm/dbptm_info.json",
                        help="Unchunked info file, for the residue at each whole-protein position "
                             "(default: ../data/dbptm/dbptm_info.json).")
    parser.add_argument("--split", default="test",
                        help="Prediction files to pair, <split>_preds_fold_N_stitched.json "
                             "(default: test).")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--save-boot-dist", action="store_true",
                        help="Also write every metric's bootstrap distribution of B - A to the JSON, "
                             "for intervals at other levels, e.g. adjusted over a whole table.")
    parser.add_argument("--no-t-test", action="store_true",
                        help="Leave out the corrected and the plain t-test, for runs whose folds are "
                             "not cross-validation folds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Processes evaluating the resamples (default: all CPUs). The results do "
                             "not depend on it; use 1 when running several tests side by side.")
    parser.add_argument("--out-dir", default="../data/statistical_tests",
                        help="Directory the results are written to, as <name>.txt (the printed "
                             "table) and <name>.json (default: ../data/statistical_tests).")
    parser.add_argument("--name",
                        help="File name of the results, without extension (default: "
                             "<run B>_vs_<run A>_<cutoff>, plus folds/metrics/split if given).")
    args = parser.parse_args()

    if args.threshold != "opt":
        float(args.threshold)  # fail early on a malformed cutoff

    if args.residues is not None:
        if not args.residues or not set(args.residues) <= set("STY"):
            parser.error(f"--residues must be a subset of STY, got {args.residues}")
        args.residues = sorted(set(args.residues), key="STY".index)
    sequences = load_sequences(args.info) if args.residues is not None else None

    paired, folds, metas = pair_folds(args.run_a, args.run_b, args.split, args.threshold,
                                      args.metrics, args.folds, args.residues, sequences)
    n_cv_folds = len((metas[0].get("fold_finished") or [])) or len(fold_numbers(args.run_a))
    results, observed, info = run_tests(paired, args.metrics, args.n_boot, args.n_perm, args.ci,
                                        args.seed, n_cv_folds, args.workers, args.save_boot_dist,
                                        t_test=not args.no_t_test)
    warnings = check_against_metadata(observed, folds, metas, args.split, args.threshold,
                                      args.residues)
    text = format_results(args.run_a, args.run_b, results, folds, info, args, warnings)
    print(text, end="")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, args.name or default_name(args))
    with open(f"{stem}.txt", "w") as f:
        f.write(text)
    with open(f"{stem}.json", "w") as f:
        json.dump({"run_a": args.run_a, "run_b": args.run_b, "folds": folds,
                   "args": vars(args), **info, "metrics": results}, f, indent=4)
    print(f"\nwritten to {stem}.txt / .json")


if __name__ == "__main__":
    main()
