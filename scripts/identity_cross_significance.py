"""Paired significance of one run family's 30/50/70 % identity versions on common test sets.

The cross-evaluated counterpart of identity_significance_test.py, from the predictions of
evaluate_checkpoints.py on the identity_cross_splits.py files. Two families of three comparisons,
differences always the higher-identity run minus the lower one:

  leakfree  50 - 30, 70 - 30 and 70 - 50 on the common leak-free subset of the 30 % test set
            (splits_<res>_chunked_max_leakfree.json): the 30 % run's own test predictions restricted
            to it, the 50 % and 70 % runs' predictions on it. No model has a 30 % homolog of a test
            protein in its training data, so the difference is that of the training data alone.
  hi        50 - 30 on the 50 % test set, 70 - 30 and 70 - 50 on the 70 % test set: the higher run's
            own test predictions, the lower run's predictions on them. The lower run is at least as
            separated from the test proteins as the higher one, so the difference includes the
            benefit of the higher run's closer homologs in training.

As in cross_representative_significance.py, a protein is scored by fold j of one run and fold k of
the other; the predictions are regrouped into the 25 (j, k) cells, written as two temporary runs
with one "fold" per cell, and model/significance_test.py is run on them. Every metric is thus still
computed over the predictions of one model per run, and the statistic is the mean over the cells
of the per-cell metric, not the 5-fold average of the metadata. Read the bootstrap and permutation
columns: the corrected resampled t-test does not apply to cells, which are not CV folds. The
cross-evaluated predictions cover whole proteins and are narrowed to the positions of the test set
owner's own predictions (the chunks its splits list), so both runs score exactly the same sites.

The bootstrap and permutation p-values are then adjusted with Benjamini-Hochberg over the three
comparisons of each family, per metric and cutoff, and collected in one summary per residue.

    cd scripts
    python identity_cross_significance.py                          # S T Y, both families, 0.5 and opt
    python identity_cross_significance.py --residues S --families leakfree --thresholds opt

Output in ../data/statistical_tests: significance_test.py's
<higher run>_vs_<lower run>_<test>_test_cells_<cutoff>.txt / .json per comparison, with <test>
leakfree, chunked_50_max or chunked_70_max, and the summary
<run with the variant replaced by "identity">_cross_cells.txt / .json.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

from cross_representative_significance import MODEL_DIR, N_FOLDS, narrow, stitched, write_cell_run
from identity_cross_splits import LEAKFREE, VERSIONS

METRICS = ['auprc', 'auroc', 'mcc']


def own_source(run, proteins=None):
    """{protein id: (fold, stitched row)} of a run's own test predictions, optionally restricted."""
    out = {}
    for fold in range(N_FOLDS):
        for row in stitched(run, fold, 'test').to_dict('records'):
            if proteins is None or str(row['id']) in proteins:
                out[str(row['id'])] = (fold, row)
    return out


def cross_source(run, split):
    """{protein id: (fold, stitched row)} of a run's evaluate_checkpoints.py predictions."""
    return {str(row['id']): (fold, row) for fold in range(N_FOLDS)
            for row in stitched(run, fold, split).to_dict('records')}


def comparisons(family, res, runs, info_path):
    """(test label, lower identity, higher identity, {identity: source}) of each comparison, the
    first source of a test set being its owner, whose positions the others are narrowed to."""
    if family == 'leakfree':
        info = pd.read_json(info_path)
        with open(os.path.join(os.path.dirname(info_path), f'splits_{res}_{LEAKFREE}.json')) as f:
            subset = {str(info.loc[r, 'parent_id']) for fold in json.load(f) for r in fold['test']}
        sources = {30: own_source(runs[30], subset)}
        if len(sources[30]) != len(subset):
            raise SystemExit(f'{runs[30]}: {len(subset) - len(sources[30])} proteins of the leak-free '
                             f'subset are not in its own test predictions')
        for hi in (50, 70):
            sources[hi] = cross_source(runs[hi], f'test_{res}_{LEAKFREE}_test_in_{VERSIONS[hi][0]}_folds_whole')
        return [('leakfree', lo, hi, {30: sources[30], lo: sources[lo], hi: sources[hi]})
                for lo, hi in ((30, 50), (30, 70), (50, 70))]
    out = []
    for lo, hi in ((30, 50), (30, 70), (50, 70)):
        split = f'test_{res}_{VERSIONS[hi][0]}_test_in_{VERSIONS[lo][0]}_folds_whole'
        out.append((VERSIONS[hi][0], lo, hi, {hi: own_source(runs[hi]), lo: cross_source(runs[lo], split)}))
    return out


def build_cells(tmp, runs, owner, lo, hi, sources):
    """Writes the cell runs of the lower and the higher run, returns their directories."""
    reference = sources[owner]
    positions = {pid: {int(p) for p in row['sequence_indices']} for pid, (_, row) in reference.items()}
    for t in (lo, hi):
        if set(sources[t]) != set(reference):
            raise SystemExit(f'{runs[t]}: {len(set(sources[t]) ^ set(reference))} proteins differ from '
                             f'the test proteins of {runs[owner]}')
    cell_of = {pid: N_FOLDS * sources[lo][pid][0] + sources[hi][pid][0] for pid in reference}

    def cells(rows):
        out = {}
        for row in rows:
            out.setdefault(cell_of[str(row['id'])], []).append(row)
        return out

    dirs = []
    for name, t, val_fold in (('lo', lo, lambda c: c // N_FOLDS), ('hi', hi, lambda c: c % N_FOLDS)):
        rows = [narrow(row, positions[pid]) for pid, (_, row) in sources[t].items()]
        out_dir = os.path.join(tmp, f'{name}_{os.path.basename(os.path.normpath(runs[t]))}')
        write_cell_run(out_dir, runs[t], cells, rows, val_fold)
        dirs.append(out_dir)
    return dirs


def adjusted_summary(rows, out_stem):
    """BH-adjusts the p-values of every family's three comparisons, writes and returns the table."""
    df = pd.DataFrame(rows)
    for p in ('bootstrap_p', 'permutation_p'):
        df[f'bh_{p}'] = df.groupby(['family', 'threshold', 'metric'])[p].transform(
            lambda ps: stats.false_discovery_control(ps.to_numpy(), method='bh'))
    lines = []
    for (family, threshold), group in df.groupby(['family', 'threshold'], sort=False):
        lines.append(f'=== {family}, MCC cutoff {threshold} ===')
        lines.append(f'  {"test":<16}{"B - A":<9}{"metric":<7}{"A":>8}{"B":>8}{"B-A":>9}  {"95% CI":>19}'
                     f'{"boot p":>8}{"BH":>8}{"perm p":>8}{"BH":>8}  {"sites":>9}')
        for _, r in group.iterrows():
            lines.append(f'  {r["test"]:<16}{f"{r.hi}-{r.lo}":<9}{r.metric:<7}{r.a:8.4f}{r.b:8.4f}'
                         f'{r.delta:+9.4f}  [{r.ci_low:+.4f}, {r.ci_high:+.4f}]{r.bootstrap_p:8.4f}'
                         f'{r.bh_bootstrap_p:8.4f}{r.permutation_p:8.4f}{r.bh_permutation_p:8.4f}  {r.sites:>9,}')
        lines.append('')
    text = '\n'.join(lines)
    with open(f'{out_stem}.txt', 'w') as f:
        f.write(text + '\n')
    df.to_json(f'{out_stem}.json', orient='records', indent=2)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--residues', nargs='+', default=['S', 'T', 'Y'])
    parser.add_argument('--run', default='finetuning_drop_{res}_2_650M_dbptm_{variant}',
                        help='Run directory name, with {res} and {variant} placeholders.')
    parser.add_argument('--families', nargs='+', choices=['leakfree', 'hi'], default=['leakfree', 'hi'])
    parser.add_argument('--thresholds', nargs='+', default=['0.5', 'opt'])
    parser.add_argument('--logs', default=os.path.join(MODEL_DIR, 'logs'))
    parser.add_argument('--info', default='../data/dbptm/dbptm_info_chunked.json',
                        help='Info file the splits index, next to the leak-free splits file.')
    parser.add_argument('--out_dir', default='../data/statistical_tests')
    parser.add_argument('--n_boot', default='1000')
    parser.add_argument('--n_perm', default='1000')
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)

    for res in args.residues:
        runs = {t: os.path.join(args.logs, args.run.format(res=res, variant=variant))
                for t, (variant, _) in VERSIONS.items()}
        summary = []
        for family in args.families:
            for test, lo, hi, sources in comparisons(family, res, runs, args.info):
                owner = 30 if family == 'leakfree' else hi
                with tempfile.TemporaryDirectory() as tmp:
                    run_a, run_b = build_cells(tmp, runs, owner, lo, hi, sources)
                    for threshold in args.thresholds:
                        name = (f'{os.path.basename(runs[hi])}_vs_{os.path.basename(runs[lo])}_{test}_test_cells_'
                                f'{"opt" if threshold == "opt" else "thr" + threshold}')
                        print(f'\n=== {res}, {hi} - {lo} on the {test} test set, cutoff {threshold}', flush=True)
                        subprocess.run([sys.executable, 'significance_test.py', run_a, run_b,
                                        '--threshold', threshold, '--n-boot', args.n_boot,
                                        '--n-perm', args.n_perm, '--out-dir', out_dir, '--name', name],
                                       cwd=MODEL_DIR, check=True)
                        with open(os.path.join(out_dir, f'{name}.json')) as f:
                            result = json.load(f)
                        for metric in METRICS:
                            r = result['metrics'][metric]
                            summary.append(dict(family=family, threshold=threshold, test=test, lo=lo, hi=hi,
                                                metric=metric, a=r['a'], b=r['b'], delta=r['delta'],
                                                ci_low=r['ci'][0], ci_high=r['ci'][1],
                                                bootstrap_p=r['bootstrap_p'], permutation_p=r['permutation_p'],
                                                sites=int(np.sum(result['n_sites']))))
        stem = os.path.join(out_dir, args.run.format(res=res, variant='identity') + '_cross_cells')
        print(f'\n{adjusted_summary(summary, stem)}\nwritten to {stem}.txt / .json')
