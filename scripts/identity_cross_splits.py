"""Cross-evaluation splits between the 30/50/70 % identity versions of the dbPTM chunked_max data.

The versions (see table_identity_versions.py) are split from three MMseqs2 clusterings of the same
info file, at --min-seq-id 0.3, 0.5 and 0.7. These nest: every 70 % cluster lies inside one 50 %
cluster, and every 50 % cluster inside one 30 % cluster (checked on load). Each version has its own
StratifiedKFold folds, so, as with the representative variants (cross_representative_splits.py), a
model of one version can only score the test proteins of another after they are laid out over its
own folds by cluster. Two kinds of files are written per residue set:

  Higher-identity test set over lower-identity folds (lo < hi):
      splits_<res>_<hi>_test_in_<lo>_folds.json
  The hi version's test proteins over the lo version's folds by their lo cluster, exactly as
  cross_representative_splits.py does it. Every protein is scored by a lo model that never saw its
  lo cluster, so it is as separated from that model's training data as from the hi model's, or more:
  pairs (30, 50), (30, 70) and (50, 70).

  Common leak-free subset of the 30 % test set over the 50 % and 70 % folds:
      splits_<res>_chunked_max_leakfree.json
      splits_<res>_chunked_max_leakfree_test_in_<hi>_folds.json
  A 30 % test protein goes to the hi fold that tests its hi cluster. That fold may still have
  trained on other members of the protein's 30 % cluster (homologs at 30-hi % identity), so only
  the proteins where neither the 50 % nor the 70 % fold has seen their 30 % cluster are kept: the
  three models then score the same proteins, each without a 30 % homolog in its training data. The
  first file is the 30 % folds with only that test subset, for restricting the 30 % run's own
  predictions to it. The subset is not a random sample: it drops proteins of 30 % families that
  split into several hi clusters, i.e. large families.

where <lo>/<hi> are the variants chunked_max, chunked_50_max, chunked_70_max. Every fold keeps the
folds file's train and dev partitions, so each file is a valid splits file of the run it is
evaluated with. The names do not end in "_chunked_max.json", so the analysis scripts selecting
runs by that suffix do not pick them up.

    cd scripts
    python identity_cross_splits.py                     # S T Y
    python identity_cross_splits.py --residues S --dry_run

Then evaluate each run on the files laid out over its folds, e.g. for S:

    cd model
    python evaluate_checkpoints.py logs/<30 % run> --entry ft_selective \\
        --splits ../data/dbptm/splits_S_chunked_50_max_test_in_chunked_max_folds.json
    python evaluate_checkpoints.py logs/<30 % run> --entry ft_selective \\
        --splits ../data/dbptm/splits_S_chunked_70_max_test_in_chunked_max_folds.json
    python evaluate_checkpoints.py logs/<50 % run> --entry ft_selective \\
        --splits ../data/dbptm/splits_S_chunked_70_max_test_in_chunked_50_max_folds.json
    python evaluate_checkpoints.py logs/<50 % run> --entry ft_selective \\
        --splits ../data/dbptm/splits_S_chunked_max_leakfree_test_in_chunked_50_max_folds.json
    python evaluate_checkpoints.py logs/<70 % run> --entry ft_selective \\
        --splits ../data/dbptm/splits_S_chunked_max_leakfree_test_in_chunked_70_max_folds.json

and compare them with identity_cross_significance.py.
"""
import argparse
import json
import os

from cross_representative_splits import (check_redistributed, check_splits, output_path, redistribute,
                                         test_fold_of_cluster)
from dataset_creation import count_unique_sites, load_clustered_data

# identity threshold -> (splits variant, clustering file in --data_dir)
VERSIONS = {30: ('chunked_max', 'dbptm_clusters.tsv'),
            50: ('chunked_50_max', 'dbptm_50.tsv'),
            70: ('chunked_70_max', 'dbptm_70.tsv')}
LEAKFREE = 'chunked_max_leakfree'
UP_PAIRS = [(30, 50), (30, 70), (50, 70)]


def check_nesting(joined):
    """Fails unless every cluster of a higher threshold lies inside one cluster of each lower one."""
    for lo, hi in UP_PAIRS:
        spans = joined[lo]['cluster_rep'].groupby(joined[hi]['cluster_rep']).nunique()
        if (spans > 1).any():
            raise SystemExit(f'{(spans > 1).sum()} clusters at {hi} % span several {lo} % clusters: '
                             f'the clusterings do not nest')


def leakfree_subset(joined, splits):
    """The 30 % test splits restricted to the proteins whose 30 % cluster neither the 50 % nor the
    70 % fold that would score them (the one testing their own cluster) has trained on."""
    keep = None
    for hi in (50, 70):
        fold_of = test_fold_of_cluster(joined[hi], splits[hi])
        seen = [set(joined[30].loc[fold['train'] + fold['dev'], 'cluster_rep']) for fold in splits[hi]]
        clean = set()
        for i, fold in enumerate(splits[30]):
            rows = joined[30].loc[fold['test']]
            for parent, group in rows.groupby('parent_id'):
                # the fold redistribute() assigns the protein to, see cross_representative_splits.py
                k = fold_of.get(joined[hi].loc[group.index[0], 'cluster_rep'], i)
                if group['cluster_rep'].iloc[0] not in seen[k]:
                    clean.add(parent)
        keep = clean if keep is None else keep & clean

    subset = []
    for fold in splits[30]:
        test = [r for r in fold['test'] if joined[30].loc[r, 'parent_id'] in keep]
        new_fold = {'train': list(fold['train']), 'dev': list(fold['dev']), 'test': test}
        new_fold['total'] = sorted(new_fold['train'] + new_fold['dev'] + new_fold['test'])
        subset.append(new_fold)
    return subset


def summary(joined, rows, sites_column):
    test = joined.loc[rows]
    return f'{test["parent_id"].nunique():,} proteins, {count_unique_sites(test, sites_column):,} sites'


def write(path, splits, dry_run):
    if dry_run:
        print(f'    (dry run) would write {path}')
        return
    with open(path, 'w') as f:
        json.dump(splits, f)
    print(f'    wrote {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', default='../data/dbptm')
    parser.add_argument('--residues', nargs='+', default=['S', 'T', 'Y'],
                        help='Residue sets, as in the splits file names.')
    parser.add_argument('--prot_info', default=None,
                        help='Defaults to <data_dir>/dbptm_info_chunked.json, which all three versions index.')
    parser.add_argument('--dry_run', action='store_true', help='Report and check, write nothing.')
    args = parser.parse_args()

    info_path = args.prot_info or os.path.join(args.data_dir, 'dbptm_info_chunked.json')
    joined = {t: load_clustered_data(info_path, os.path.join(args.data_dir, clusters),
                                     [set(r) for r in args.residues])
              for t, (_, clusters) in VERSIONS.items()}
    check_nesting(joined)
    print(f'info {info_path}; the 30/50/70 % clusterings nest')

    for res in args.residues:
        splits = {}
        for t, (variant, _) in VERSIONS.items():
            with open(os.path.join(args.data_dir, f'splits_{res}_{variant}.json')) as f:
                splits[t] = json.load(f)
        problems = [p for t, (v, _) in VERSIONS.items() for p in check_splits(joined[t], splits[t], f'{res}_{v}')]
        if problems:
            raise SystemExit('splits failed the checks - not generated from this info and clusters file?\n  '
                             + '\n  '.join(problems))
        sites = f'{res}_sites'
        print(f'\n{res}:')

        for lo, hi in UP_PAIRS:
            new = redistribute(joined[lo], splits[lo], splits[hi])
            problems = check_redistributed(joined[lo], splits[lo], splits[hi], new)
            if problems:
                raise SystemExit('redistributed splits failed the checks:\n  ' + '\n  '.join(problems))
            print(f'  {hi} % test over {lo} % folds: '
                  f'{summary(joined[hi], [r for f in new for r in f["test"]], sites)}')
            write(output_path(args.data_dir, res, VERSIONS[hi][0], VERSIONS[lo][0]), new, args.dry_run)

        subset = leakfree_subset(joined, splits)
        full = [r for f in splits[30] for r in f['test']]
        print(f'  leak-free 30 % test subset: {summary(joined[30], [r for f in subset for r in f["test"]], sites)} '
              f'of {summary(joined[30], full, sites)}')
        write(os.path.join(args.data_dir, f'splits_{res}_{LEAKFREE}.json'), subset, args.dry_run)
        for hi in (50, 70):
            new = redistribute(joined[hi], splits[hi], subset)
            # no hi cluster, and no 30 % cluster either, in the train or dev partition of the model
            problems = (check_redistributed(joined[hi], splits[hi], subset, new)
                        + check_redistributed(joined[30], splits[hi], subset, new))
            if problems:
                raise SystemExit('leak-free splits failed the checks:\n  ' + '\n  '.join(problems))
            write(output_path(args.data_dir, res, LEAKFREE, VERSIONS[hi][0]), new, args.dry_run)
