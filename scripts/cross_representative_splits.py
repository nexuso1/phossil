"""Test sets of one representative strategy, laid out over the folds of the other.

dataset_creation.py writes two splits files per residue set from one clustering: splits_<res>_chunked
(the representative the clustering assigned) and splits_<res>_chunked_max (the member with the most
sites of the residue set). Their clusters correspond: every cluster of the default file is in the
_max file, represented by one protein in each, and the _max file adds the clusters whose assigned
representative has no site of the residue set. Their FOLDS do not: StratifiedKFold runs on each
file's own proteins and length classes, so a cluster tested in fold 0 of one file is usually trained
on in fold 0 of the other (only ~1/5 of the test clusters coincide, as by chance). Evaluating fold N
of a default model on fold N of the _max test set would score it on clusters it trained on.

This script therefore takes the test proteins of one file (--test variant) and redistributes them
over the folds of the other (--folds variant) by cluster: a protein goes to the fold whose test
partition holds its cluster, so the fold's model never saw the cluster in train or dev. A cluster the
--folds file lacks altogether (a _max-only cluster) was never seen by any of its models and stays in
the fold that tests it in its own file. Over the five folds every test protein of the --test file is
scored exactly once, by a model blind to its cluster.

    cd scripts
    python cross_representative_splits.py                           # both directions, every residue set
    python cross_representative_splits.py --residues S T
    python cross_representative_splits.py --variants chunked_50 chunked_50_max \\
        --clusters ../data/dbptm/dbptm_50.tsv
    python cross_representative_splits.py --dry_run                 # report and check, write nothing

Output, per residue set and direction, next to the inputs:

    splits_<res>_<test>_test_in_<folds>_folds.json

e.g. splits_S_chunked_max_test_in_chunked_folds.json, the _max test proteins over the default folds,
for evaluating a model trained on splits_S_chunked.json. Every fold holds the --folds file's train
and dev partitions unchanged and the redistributed test partition, so the file is still a valid
splits file (training on it is training the --folds run). Both files index the same info file. The
name deliberately does not end in "_chunked_max.json", so the analysis scripts selecting runs by that
suffix do not pick it up. Evaluate a finished run on it with:

    cd model
    python evaluate_checkpoints.py logs/<S chunked run> --entry ft_selective \\
        --splits ../data/dbptm/splits_S_chunked_max_test_in_chunked_folds.json
    python evaluate_checkpoints.py logs/<S chunked_max run> --entry ft_selective \\
        --splits ../data/dbptm/splits_S_chunked_test_in_chunked_max_folds.json

which writes test_S_chunked_max_test_in_chunked_folds_whole_{preds,metrics}_fold_N* into the run's
fold dirs. Proteins that represent their cluster in both files are in the run's own test set of the
same fold, so evaluate_checkpoints.py's comparison with the saved predictions covers them.

The cluster information is rebuilt from the info and clusters files the way dataset_creation.py
builds it, and both files are checked against it: one protein per cluster, every cluster tested in
exactly one fold, no cluster in two partitions of a fold.
"""
import argparse
import json
import os

from dataset_creation import count_unique_sites, load_clustered_data

PARTITIONS = ['train', 'dev', 'test']


def check_splits(joined, splits, name):
    """Problems of a splits file against the clustering, as a list of messages."""
    problems = []
    total = splits[0]['total']
    if not set(total) <= set(joined.index):
        return [f'{name}: rows that are not in the info file']
    per_cluster = joined.loc[total].groupby('cluster_rep')['parent_id'].nunique()
    if (per_cluster > 1).any():
        problems.append(f'{name}: {(per_cluster > 1).sum()} clusters with several proteins')

    tested = {}
    for i, fold in enumerate(splits):
        if sorted(fold['total']) != sorted(total):
            problems.append(f'{name} fold {i}: total differs from fold 0')
        if sorted(r for part in PARTITIONS for r in fold[part]) != sorted(total):
            problems.append(f'{name} fold {i}: partitions do not add up to total')
        clusters = {part: set(joined.loc[fold[part], 'cluster_rep']) for part in PARTITIONS}
        for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
            if clusters[a] & clusters[b]:
                problems.append(f'{name} fold {i}: {len(clusters[a] & clusters[b])} clusters in both {a} and {b}')
        for cluster in clusters['test']:
            tested.setdefault(cluster, []).append(i)

    if set(tested) != set(per_cluster.index):
        problems.append(f'{name}: {len(set(per_cluster.index) - set(tested))} clusters never tested')
    if any(len(folds) > 1 for folds in tested.values()):
        problems.append(f'{name}: {sum(len(f) > 1 for f in tested.values())} clusters tested in several folds')
    return problems


def test_fold_of_cluster(joined, splits):
    """{cluster_rep: fold} of the fold whose test partition holds each cluster."""
    return {cluster: i for i, fold in enumerate(splits)
            for cluster in joined.loc[fold['test'], 'cluster_rep'].unique()}


def redistribute(joined, fold_splits, test_splits):
    """The --test file's test rows over the --folds file's folds, train and dev kept as they are."""
    fold_of = test_fold_of_cluster(joined, fold_splits)
    tests = [[] for _ in fold_splits]
    for i, fold in enumerate(test_splits):
        rows = joined.loc[fold['test']]
        for cluster, group in rows.groupby('cluster_rep'):
            # A cluster the folds file does not have was never seen by any of its models
            tests[fold_of.get(cluster, i)].extend(group.index)

    new_splits = []
    for fold, test in zip(fold_splits, tests):
        new_fold = {'train': list(fold['train']), 'dev': list(fold['dev']), 'test': sorted(test)}
        new_fold['total'] = sorted(new_fold['train'] + new_fold['dev'] + new_fold['test'])
        new_splits.append(new_fold)
    return new_splits


def check_redistributed(joined, fold_splits, test_splits, new_splits):
    problems = []
    all_test = sorted(r for fold in test_splits for r in fold['test'])
    if sorted(r for fold in new_splits for r in fold['test']) != all_test:
        problems.append('the redistributed test partitions are not the --test file\'s test rows, each once')
    for i, (fold, new_fold) in enumerate(zip(fold_splits, new_splits)):
        seen = set(joined.loc[fold['train'] + fold['dev'], 'cluster_rep'])
        leaked = seen & set(joined.loc[new_fold['test'], 'cluster_rep'])
        if leaked:
            problems.append(f'fold {i}: {len(leaked)} test clusters in the train or dev partition of the model')
        seen_parents = set(joined.loc[fold['train'] + fold['dev'], 'parent_id'])
        if seen_parents & set(joined.loc[new_fold['test'], 'parent_id']):
            problems.append(f'fold {i}: test proteins in the train or dev partition of the model')
    return problems


def fold_report(joined, fold, new_fold, known_clusters, sites_column):
    """Row of the summary table: where the fold's new test proteins come from."""
    own_parents = set(joined.loc[fold['test'], 'parent_id'])
    test = joined.loc[new_fold['test']]
    per_protein = test.groupby('parent_id')['cluster_rep'].first()
    same = sum(p in own_parents for p in per_protein.index)
    absent = sum(c not in known_clusters for c in per_protein)
    return (len(per_protein), same, len(per_protein) - same - absent, absent,
            count_unique_sites(joined.loc[fold['test']], sites_column), count_unique_sites(test, sites_column))


def output_path(data_dir, residues, test_variant, folds_variant):
    return os.path.join(data_dir, f'splits_{residues}_{test_variant}_test_in_{folds_variant}_folds.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', default='../data/dbptm')
    parser.add_argument('--residues', nargs='+', default=['S', 'T', 'Y', 'ST', 'STY'],
                        help='Residue sets, as in the splits file names.')
    parser.add_argument('--variants', nargs=2, default=['chunked', 'chunked_max'], metavar=('A', 'B'),
                        help='The two splits file variants (splits_<res>_<variant>.json) generated together. '
                             'Both directions are written: B\'s test over A\'s folds and A\'s test over B\'s.')
    parser.add_argument('--prot_info', default=None, help='Defaults to <dataset>_info_chunked.json.')
    parser.add_argument('--clusters', default=None,
                        help='Clustering both files were split by. Defaults to <dataset>_clusters.tsv.')
    parser.add_argument('--dry_run', action='store_true', help='Report and check, write nothing.')
    args = parser.parse_args()

    dataset = os.path.basename(os.path.normpath(args.data_dir))
    info_path = args.prot_info or os.path.join(args.data_dir, f'{dataset}_info_chunked.json')
    clusters_path = args.clusters or os.path.join(args.data_dir, f'{dataset}_clusters.tsv')
    joined = load_clustered_data(info_path, clusters_path, [set(r) for r in args.residues])
    print(f'info {info_path}, clusters {clusters_path}')

    for residues in args.residues:
        splits = {}
        for variant in args.variants:
            with open(os.path.join(args.data_dir, f'splits_{residues}_{variant}.json')) as f:
                splits[variant] = json.load(f)
        problems = [p for v in args.variants for p in check_splits(joined, splits[v], f'{residues}_{v}')]
        if problems:
            raise SystemExit('splits failed the checks - not generated from this info and clusters file?\n  '
                             + '\n  '.join(problems))

        clusters = {v: set(joined.loc[splits[v][0]['total'], 'cluster_rep']) for v in args.variants}
        a, b = args.variants
        print(f'\n{residues}: {len(clusters[a]):,} clusters in {a}, {len(clusters[b]):,} in {b}, '
              f'{len(clusters[a] & clusters[b]):,} shared, {len(clusters[a] - clusters[b]):,} only in {a}, '
              f'{len(clusters[b] - clusters[a]):,} only in {b}')

        for folds_variant, test_variant in [(a, b), (b, a)]:
            fold_splits, test_splits = splits[folds_variant], splits[test_variant]
            new_splits = redistribute(joined, fold_splits, test_splits)
            problems = check_redistributed(joined, fold_splits, test_splits, new_splits)
            if problems:
                raise SystemExit('redistributed splits failed the checks:\n  ' + '\n  '.join(problems))

            print(f'  {test_variant} test over {folds_variant} folds (proteins: same = the fold\'s own test '
                  f'protein, other rep = another member of one of its test clusters, new cluster = absent '
                  f'from {folds_variant}):')
            print(f'    {"fold":<5}{"proteins":>9}{"same":>7}{"other rep":>10}{"new cluster":>12}{"sites":>18}')
            for i, (fold, new_fold) in enumerate(zip(fold_splits, new_splits)):
                n, same, other, absent, own_sites, sites = fold_report(joined, fold, new_fold,
                                                                       clusters[folds_variant], f'{residues}_sites')
                print(f'    {i:<5}{n:>9,}{same:>7,}{other:>10,}{absent:>12,}{f"{own_sites:,} -> {sites:,}":>18}')

            out = output_path(args.data_dir, residues, test_variant, folds_variant)
            if args.dry_run:
                print(f'    (dry run) would write {out}')
                continue
            with open(out, 'w') as f:
                json.dump(new_splits, f, indent='\t')
            print(f'    wrote {out}')

    print('\nchecks passed: one protein per cluster, every test protein redistributed exactly once, and no '
          'test cluster or protein in the train or dev partition of the fold that tests it')
