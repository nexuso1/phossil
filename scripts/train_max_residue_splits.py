"""Single-residue splits over a joint (ST / STY) splits file, with residue-specific train representatives.

derive_residue_splits.py narrows a joint _max splits file to one residue, so every cluster stays
represented by the member with the most sites of the JOINT residue set. A native single-residue
_max file instead picks the member with the most sites of that residue. This script isolates that
difference: per fold, every cluster of the joint file's train partition is re-represented by its
member with the most sites of the residue (the rule of dataset_creation.py's
extract_max_site_representatives, tie breaks included), while the dev and test partitions are copied
from the joint file unchanged, row for row. A model trained on the output is therefore scored on
exactly the dev/test sets of the joint model and of the _from_<source> derived model, and differs
from the derived model only in which protein represents each train cluster.

    cd scripts
    python train_max_residue_splits.py --source STY                  # S, T and Y from STY
    python train_max_residue_splits.py --source ST --residues S
    python train_max_residue_splits.py --source STY --dry_run        # report only, write nothing

How the train partition is built, per fold and residue:

  - The clusters are those of the joint train partition, nothing is added or dropped.
  - Each cluster's representative is its member with the most sites of the residue, counted over
    the chunks with such a site, as in a native single-residue _max dataset. A cluster none of
    whose members has a site of the residue keeps the joint representative (the native dataset
    drops it; keeping it leaves the cluster set equal to the derived file's).
  - The rows of the chosen protein are those the joint file would hold had it been the
    representative (chunks with a site of the joint set), narrowed by --keep exactly as
    derive_residue_splits.py narrows them. A cluster whose representative does not change thus
    gets exactly the rows it has in the derived file, which the script checks.

The cluster information is rebuilt from the info and clusters files the way dataset_creation.py
builds it, and checked against the joint file: its representatives must be what the max strategy
picks for the joint residue set, otherwise the files were not generated from each other.

Output: splits_<res>_<variant>_from_<source>_trainmax.json next to the source, e.g.
splits_S_chunked_max_from_STY_trainmax.json. Train on it as on any splits file:

    cd model
    python ft_selective.py ... --prot_info_path ../data/dbptm/dbptm_info_chunked.json \\
        --dataset_path ../data/dbptm/splits_S_chunked_max_from_STY_trainmax.json --residues "['S']" \\
        -o finetuning_S_2_650M_dbptm_chunked_max_from_STY_trainmax

Because dev/test are not narrowed, they can hold chunks with no candidate of the residue at all
(mostly Y from STY, ~90 per fold). Those chunks contribute no position to the loss or the metrics.
"""
import argparse
import json
import os

from dataset_creation import extract_max_site_representatives, filter_dataset, load_clustered_data
from derive_residue_splits import chunk_flags, site_count

EVAL_PARTITIONS = ['dev', 'test']


def joint_representatives(joined, source):
    """Rows of the joint _max dataset, as dataset_creation.py selects them."""
    return extract_max_site_representatives(filter_dataset(joined, source),
                                            sites_column=f'{source}_sites')


def train_representatives(joined, train_rows, residue):
    """
    {cluster_rep: parent_id} of the protein representing each train cluster for the residue, and
    the clusters that fell back to the joint representative.
    """
    joint_parents = joined.loc[train_rows].groupby('cluster_rep')['parent_id'].unique()
    multiple = joint_parents[joint_parents.apply(len) > 1]
    if len(multiple):
        raise SystemExit(f'{len(multiple)} train clusters are represented by more than one protein')
    chosen = joint_parents.apply(lambda p: p[0]).to_dict()

    members = joined[joined['cluster_rep'].isin(set(chosen))]
    best = extract_max_site_representatives(filter_dataset(members, residue),
                                            sites_column=f'{residue}_sites')
    best = best.groupby('cluster_rep')['parent_id'].first().to_dict()

    fallback = set(chosen) - set(best)
    chosen.update(best)
    return chosen, fallback


def build_fold(joined, fold, residue, source, keep):
    chosen, fallback = train_representatives(joined, fold['train'], residue)
    # The rows the joint file would hold for the chosen proteins, narrowed like the derived file
    eligible = joined[joined['parent_id'].isin(set(chosen.values()))
                      & joined[f'{source}_sites'].apply(len).gt(0) & keep]
    train = sorted(eligible.index)
    new_fold = {'train': train, **{part: list(fold[part]) for part in EVAL_PARTITIONS}}
    new_fold['total'] = sorted(train + [row for part in EVAL_PARTITIONS for row in fold[part]])
    return new_fold, chosen, fallback


def check(joined, source_fold, new_fold, chosen, keep, i):
    problems = []
    for part in EVAL_PARTITIONS:
        if new_fold[part] != source_fold[part]:
            problems.append(f'fold {i} {part}: differs from the source')

    clusters = {part: set(joined.loc[new_fold[part], 'cluster_rep']) for part in new_fold}
    parents = {part: set(joined.loc[new_fold[part], 'parent_id']) for part in new_fold}
    for part in EVAL_PARTITIONS:
        if clusters['train'] & clusters[part]:
            problems.append(f'fold {i}: {len(clusters["train"] & clusters[part])} clusters in both '
                            f'train and {part}')
        if parents['train'] & parents[part]:
            problems.append(f'fold {i}: {len(parents["train"] & parents[part])} proteins in both '
                            f'train and {part}')
    per_cluster = joined.loc[new_fold['train']].groupby('cluster_rep')['parent_id'].nunique()
    if (per_cluster > 1).any():
        problems.append(f'fold {i}: {(per_cluster > 1).sum()} train clusters with several proteins')
    if not set(new_fold['train']) <= set(keep[keep].index):
        problems.append(f'fold {i}: train rows outside the --keep filter')

    # Clusters keeping their joint representative must keep exactly their derived rows
    source_parents = joined.loc[source_fold['train']].groupby('cluster_rep')['parent_id'].first()
    unchanged = {c for c, p in chosen.items() if source_parents[c] == p}
    derived = {r for r in source_fold['train'] if keep[r]}
    def rows_of(rows):
        return {r for r in rows if joined.at[r, 'cluster_rep'] in unchanged}
    if rows_of(new_fold['train']) != rows_of(derived):
        problems.append(f'fold {i}: unchanged clusters do not keep their derived train rows')
    return problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', default='../data/dbptm')
    parser.add_argument('--source', required=True, help='Joint residue set, e.g. ST or STY.')
    parser.add_argument('--variant', default='chunked_max',
                        help='Splits file variant: splits_<source>_<variant>.json. Must be a _max '
                             'variant, its representatives are checked against the max strategy.')
    parser.add_argument('--residues', nargs='+', default=None,
                        help='Residues to write splits for. Defaults to each letter of --source.')
    parser.add_argument('--keep', choices=['candidates', 'sites'], default='candidates',
                        help='Train chunks kept, as in derive_residue_splits.py. The existing '
                             '_from_<source> files use candidates.')
    parser.add_argument('--prot_info', default=None,
                        help='Defaults to <dataset>_info_chunked.json.')
    parser.add_argument('--clusters', default=None,
                        help='Clustering the source was split by. Defaults to '
                             '<dataset>_clusters.tsv.')
    parser.add_argument('--dry_run', action='store_true', help='Report and check, write nothing.')
    args = parser.parse_args()

    dataset = os.path.basename(os.path.normpath(args.data_dir))
    source_path = os.path.join(args.data_dir, f'splits_{args.source}_{args.variant}.json')
    info_path = args.prot_info or os.path.join(args.data_dir, f'{dataset}_info_chunked.json')
    clusters_path = args.clusters or os.path.join(args.data_dir, f'{dataset}_clusters.tsv')
    residues = args.residues or list(args.source)
    for residue in residues:
        if len(residue) != 1 or residue not in args.source:
            raise SystemExit(f'{residue} is not a single residue of {args.source}')

    with open(source_path) as f:
        source = json.load(f)
    joined = load_clustered_data(info_path, clusters_path, [set(args.source)] + [{r} for r in residues])
    print(f'source {source_path} ({len(source)} folds), info {info_path}, clusters {clusters_path}, '
          f'keep={args.keep}')

    expected = set(joint_representatives(joined, args.source).index)
    for i, fold in enumerate(source):
        if set(fold['total']) != expected:
            raise SystemExit(f'fold {i}: the source does not hold the max representatives of '
                             f'{clusters_path} - the files were not generated from each other')

    for residue in residues:
        has_candidate, has_site = chunk_flags(joined, residue)
        keep = has_candidate if args.keep == 'candidates' else has_site

        print(f'\n{residue} from {args.source}, train partition (derived -> trainmax):')
        print(f'  {"fold":<5}{"clusters":>9}{"changed":>9}{"fallback":>9}'
              f'{"chunks":>16}{"sites":>18}')
        splits, problems = [], []
        for i, fold in enumerate(source):
            new_fold, chosen, fallback = build_fold(joined, fold, residue, args.source, keep)
            problems += check(joined, fold, new_fold, chosen, keep, i)
            splits.append(new_fold)

            derived = [r for r in fold['train'] if keep[r]]
            joint_parents = set(joined.loc[fold['train'], 'parent_id'])
            changed = sum(p not in joint_parents for p in chosen.values())
            chunks = f'{len(derived):,} -> {len(new_fold["train"]):,}'
            sites = (f'{site_count(joined, derived, residue):,} -> '
                     f'{site_count(joined, new_fold["train"], residue):,}')
            print(f'  {i:<5}{len(chosen):>9,}{changed:>9,}{len(fallback):>9,}{chunks:>16}{sites:>18}')

        if problems:
            raise SystemExit('splits failed the checks:\n  ' + '\n  '.join(problems))

        out = os.path.join(args.data_dir,
                           f'splits_{residue}_{args.variant}_from_{args.source}_trainmax.json')
        if args.dry_run:
            print(f'  (dry run) would write {out}')
            continue
        with open(out, 'w') as f:
            json.dump(splits, f, indent='\t')
        print(f'  wrote {out}')

    print('\nchecks passed: dev and test are the source\'s row for row, no cluster or protein is in '
          'train and dev/test, one protein per train cluster, and unchanged clusters keep their '
          'derived train rows')
