import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import os
import json
import argparse

def plot_cluster_sizes(ax,  clusters, cmap='viridis', title=None):
    cmap = colormaps['viridis']
    counts = np.bincount(clusters.groupby('cluster_rep').count()['id'])
    colors = np.zeros_like(counts, np.float32)
    mask = counts > 0
    nonzero = counts[mask]
    nonzero = np.log(nonzero) / np.log(nonzero).max()
    colors[mask] = nonzero
    colors = cmap(colors)
    ax.bar(np.arange(len(counts)), counts, color=colors)
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_yscale('log')
    ax.set_title(title)

def plot_binned_cluster_dist(ax,  clusters, cmap='viridis', title=None):
    cmap = colormaps['viridis']
    group_sizes = clusters.groupby('cluster_rep').count().to_numpy().flatten()
    bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, 1000]
    hist = np.histogram(group_sizes, bins=bins)
    labels = [str(i) for i in bins[:-1]]
    for i in range(4, len(labels) - 1):
        labels[i] = labels[i] + '-' + labels[i + 1]

    labels[-1] = labels[-1] + '+'
    colors = cmap(hist[0] * 2 / np.sum(hist[0]))
    ax.bar(labels, hist[0] / np.sum(hist[0]), color=colors)
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Percentage of data')
    ax.set_title(title)

def plot_info(data, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    plot_cluster_sizes(ax[0], data, title='Final cluster size distribution')
    plot_binned_cluster_dist(ax[1], data, title='Final cluster size relative distribution')

    plt.savefig(path)

def prune_long_seqs(data : pd.DataFrame, max_length : int = 1023):
    return data[data['sequence'].apply(lambda x: len(x) < max_length)]

def add_chunk_columns(data : pd.DataFrame):
    """
    Fills in the columns added by chunk_proteins.py, so that datasets of whole proteins
    (where every protein is its own parent) go through the same code path.
    """
    if 'parent_id' not in data:
        data['parent_id'] = data['id']

    if 'parent_length' not in data:
        data['parent_length'] = data['sequence'].apply(len)

    return data

def collapse_fn(group : pd.DataFrame):
    #print(group['sequence'])
    #print(group)
    lens = group['sequence'].apply(lambda x: len(x))
    new_rep = lens.argmax()
    group['new_rep'] = group.iloc[new_rep].parent_id
    return group

def reassign_representatives(data : pd.DataFrame):
    reps = set(data.cluster_rep.unique())
    members = set(data.parent_id.unique())

    # Representatives that are not in the filtered dataset
    missing = set.difference(reps, members)
    # Chunked datasets keep every protein, so there may be nothing to reassign
    if not missing:
        return data

    grouped_to_fix = data[data['cluster_rep'].apply(lambda x: x in missing)].groupby('cluster_rep')
    fixed = grouped_to_fix.apply(collapse_fn)
    fixed = fixed.reset_index(level='cluster_rep', drop=True)
    data.loc[fixed.index, 'cluster_rep'] = fixed['new_rep']
    return data

def compute_length_labels(data):
    # Stratify by the length of the whole protein, chunks of one protein are not independent
    data['length'] = data['parent_length']
    _, length_bins = np.histogram(data['length'], 10, range=(0, 1000))
    length_labels = np.digitize(data['length'], length_bins)
    data['length_class'] = length_labels

    return data

def extract_per_residue_sites(row : pd.Series, res_sets, set_labels):
    results = row.copy()
    for j, res_set in enumerate(res_sets):
        # Indices are 1-based
        results[f'{set_labels[j]}_sites'] = [i for i in row['sites'] if row['sequence'][int(i)-1] in res_set ]
    return results

def compute_site_labels(data : pd.DataFrame, res_sets, set_labels):

    return data.apply(lambda x: extract_per_residue_sites(x, res_sets, set_labels), axis=1, result_type='expand')

def filter_dataset(df : pd.DataFrame, set_label):
    return df[df[f'{set_label}_sites'].apply(lambda x: len(x) > 0)].copy()

def write_dataset_info(all_splits, out_folder, suffix=''):
    res = {}
    for label, splits in all_splits.items():
        res[label] = {}
        for i, fold in enumerate(splits):
            for data_type in fold:
                res[label][f'fold_{i}_{data_type}_size'] = len(fold[data_type])

    pd.DataFrame.from_dict(res).T.to_csv(os.path.join(out_folder, f'dataset_info{suffix}.csv'), sep=',')

def write_site_increase(data : pd.DataFrame, set_labels : list[str], out_folder, suffix=''):
    """
    Records how many sites picking the best annotated member of every cluster as its representative
    gains over the representatives assigned by the clustering, for each residue set.
    """
    res = {}
    for label in set_labels:
        filtered = filter_dataset(data, label)
        sites_column = f'{label}_sites'
        default = get_representatives(filtered, label)
        max_site = get_representatives(filtered, label, max_site_reps=True)

        default_sites = count_unique_sites(default, sites_column)
        max_sites = count_unique_sites(max_site, sites_column)
        res[label] = {
            # Clusters whose assigned representative has no site of this residue set are lost
            # entirely by the default selection, the max site one keeps them
            'default_proteins' : default['parent_id'].nunique(),
            'max_proteins' : max_site['parent_id'].nunique(),
            'default_sites' : default_sites,
            'max_sites' : max_sites,
            'site_increase' : max_sites - default_sites,
            'relative_increase' : (max_sites - default_sites) / default_sites if default_sites else 0,
        }

    pd.DataFrame.from_dict(res).T.to_csv(os.path.join(out_folder, f'site_increase{suffix}.csv'), sep=',')

def get_parent_length_classes(data : pd.DataFrame):
    """
    Length class per protein instead of per row. Chunks of one protein are near duplicates of
    each other, so they have to be split as a unit, otherwise the folds leak into each other.
    """
    return data.groupby('parent_id', sort=False)['length_class'].first()

def get_parent_indices(data : pd.DataFrame, parents):
    """
    Row indices of all chunks belonging to the given proteins.
    """
    return data.index[data['parent_id'].isin(set(parents))].tolist()

def get_kfold_splits(data : pd.DataFrame, seed=42, train_size=0.8):
    cv = StratifiedKFold(random_state=seed, shuffle=True)
    parent_classes = get_parent_length_classes(data)
    splits = []
    for train, test in cv.split(parent_classes.index, parent_classes):
        # Hold out a dev partition of the train split, over proteins like the rest of the splits.
        # Same arguments FullProteinDataset.get_fold used when creating the dev set on the fly
        train_parents, dev_parents = train_test_split(parent_classes.index[train], train_size=train_size,
                                                      random_state=seed)
        orig_indices_train = get_parent_indices(data, train_parents)
        orig_indices_dev = get_parent_indices(data, dev_parents)
        orig_indices_test = get_parent_indices(data, parent_classes.index[test])
        splits.append({'train' : orig_indices_train, 'dev' : orig_indices_dev,
                       'test' : orig_indices_test, 'total' : list(data.index)})
    return splits

def get_release_split(data, dev_size=None, seed=42):
    if not dev_size:
                raise AssertionError('Dev size not set for release dataset preparation.')
    parent_classes = get_parent_length_classes(data)
    try:
        train, dev = train_test_split(parent_classes.index, test_size=dev_size, random_state=seed, stratify=parent_classes)

    except ValueError:
        largest_class = parent_classes.max()
        print(largest_class)
        parent_classes[parent_classes == largest_class] = largest_class - 1
        train, dev = train_test_split(parent_classes.index, test_size=dev_size, random_state=seed, stratify=parent_classes)

    return [{'train' : get_parent_indices(data, train), 'dev' : get_parent_indices(data, dev),
             'total' : data.index.to_list()}]

def split_data_per_residue(data : pd.DataFrame, set_labels : list[str], seed=42, release=False,
                           dev_size : float|None =None, train_size=0.8, max_site_reps=False):
    total_splits : dict[str, list[dict[str, list[int]]]] = {}
    for label in set_labels:
        filtered = filter_dataset(data, label)

        # Use only cluster representatives going forward
        filtered = get_representatives(filtered, label, max_site_reps=max_site_reps)
        if release:
            total_splits[label] = get_release_split(data, dev_size, seed)
        else:
            total_splits[label] = get_kfold_splits(filtered, seed=seed, train_size=train_size)
    
    return total_splits

def save_dataset(all_splits, out_folder, suffix=''):
    for label, splits in all_splits.items():
        with open(os.path.join(out_folder, f"splits_{label}{suffix}.json"), 'w') as f:
            json.dump(splits, f, indent='\t')
        
def extract_representatives(data):
    reps = set(data['cluster_rep'].unique())
    rep_mask = data['parent_id'].apply(lambda x : x in reps)
    return data[rep_mask]

def get_representatives(data : pd.DataFrame, set_label : str, max_site_reps=False):
    """
    Cluster representatives of the dataset, either the ones assigned by the clustering, or the
    member of each cluster carrying the most sites of the given residue set.
    """
    if max_site_reps:
        return extract_max_site_representatives(data, sites_column=f'{set_label}_sites')

    return extract_representatives(data)

def get_protein_sites(data : pd.DataFrame, sites_column='sites'):
    """
    Sites of every row, shifted onto the protein the row came from. Chunk sites are chunk local
    and overlapping chunks repeat the sites they share, shifting is what tells those apart.
    """
    offsets = data['offset'] if 'offset' in data else pd.Series(0, index=data.index)
    return [{site + offset for site in sites} for sites, offset in zip(data[sites_column], offsets)]

def count_unique_sites(data : pd.DataFrame, sites_column='sites'):
    """
    Number of distinct sites in the data, counting a site shared by overlapping chunks once.
    """
    per_protein = {}
    for parent_id, sites in zip(data['parent_id'], get_protein_sites(data, sites_column)):
        per_protein.setdefault(parent_id, set()).update(sites)

    return sum(len(sites) for sites in per_protein.values())

def extract_max_site_representatives(data : pd.DataFrame, sites_column='sites'):
    """
    Picks the protein with the most sites in each cluster as its representative, and returns the
    rows of the chosen proteins.

    Expects "sites_column" to hold only the sites relevant for the dataset being created. Sites of
    a chunked protein are counted over all of its chunks, without counting the ones its overlapping
    chunks have in common twice.
    """
    per_protein = pd.DataFrame({
        'cluster_rep' : data['cluster_rep'].to_numpy(),
        'parent_id' : data['parent_id'].to_numpy(),
        'parent_length' : data['parent_length'].to_numpy(),
        'sites' : get_protein_sites(data, sites_column),
    }).groupby(['cluster_rep', 'parent_id'], sort=False).agg(
        n_sites=('sites', lambda chunks: len(set().union(*chunks))),
        length=('parent_length', 'first')).reset_index()

    # Sorted, so that ties between equally annotated proteins are broken by the longer sequence,
    # and always the same way no matter the order of the rows
    per_protein = per_protein.sort_values(['n_sites', 'length', 'parent_id'], ascending=[False, False, True])
    representatives = per_protein.groupby('cluster_rep', sort=False)['parent_id'].first()

    return data[data['parent_id'].isin(set(representatives))]

def create_splits(prot_info_path, clusters_path, res_sets, out_folder=None, release=False, release_dev_size=0.2,
                  train_size=0.8, suffix=''):
    prot_info = pd.read_json(prot_info_path)
    clusters = pd.read_csv(clusters_path, sep='\t', names=['cluster_rep', 'cluster_mem'])

    clusters_name = Path(clusters_path).stem
    out_folder = f'../data/dataset_{clusters_name}' if not out_folder else out_folder
    os.makedirs(out_folder, exist_ok=True)

    # Chunked datasets are clustered and split by the protein a chunk came from
    prot_info = add_chunk_columns(prot_info)

    # Join the cluster information with proteins that we have in the prot info
    # Some proteins from prot_info may have representatives that are not inside prot_info,
    # or they might be longer than max length for the base PLM
    joined = prot_info.join(clusters.set_index('cluster_mem'), on='parent_id', how='left').drop_duplicates('id')
    joined = prune_long_seqs(joined)

    # Find the longest member of the cluster that is inside prot_info, and assign it as the representative
    joined = reassign_representatives(joined)

    # Plot cluster size distribution
    # plot_name = 'cluster_dist.png'
    # plot_info(joined, os.path.join(out_folder, plot_name))


    # Split sites according to residues in res_sets
    set_labels = ["".join(sorted(list(res_set))) for res_set in res_sets]
    joined = compute_site_labels(joined, res_sets, set_labels)

    # Compute length labels for stratification
    joined = compute_length_labels(joined)

    # Split data into folds using stratified k-fold cross-validation, or just train-dev if release dataset
    all_splits = split_data_per_residue(joined, set_labels, release=release, dev_size=release_dev_size,
                                        train_size=train_size)


    write_dataset_info(all_splits, out_folder, suffix=suffix)
    save_dataset(all_splits, out_folder, suffix=suffix)

    # The same splits, but built from the best annotated member of each cluster instead of the
    # representative the clustering assigned
    max_splits = split_data_per_residue(joined, set_labels, release=release, dev_size=release_dev_size,
                                        train_size=train_size, max_site_reps=True)

    write_dataset_info(max_splits, out_folder, suffix=f'{suffix}_max')
    save_dataset(max_splits, out_folder, suffix=f'{suffix}_max')

    write_site_increase(joined, set_labels, out_folder, suffix=suffix)

def main(args):
    res_sets = eval(args.res_sets)
    create_splits(args.prot_info, args.clusters, res_sets, release=args.release, out_folder=args.out_folder,
                  release_dev_size=args.release_dev_size, train_size=args.train_size, suffix=args.suffix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot_info', type=str, default='../data/dbptm/dbptm_info_chunked.json')
    parser.add_argument('--clusters', type=str, default='../data/dbptm/dbptm_clusters.tsv')
    parser.add_argument('--res_sets', type=str, default="[{'S'}, {'T'}, {'Y'}, {'S', 'T'}, {'S', 'T', 'Y'}]")
    parser.add_argument('--release', action='store_true')
    parser.add_argument('--out_folder', type=str, help='Output folder', default='../data/dbptm')
    parser.add_argument('--release_dev_size', type=float, help='Release dataset dev size', default=0.2)
    parser.add_argument('--suffix', type=str, help='Suffix for the splits filename.', default='')
    parser.add_argument('--train_size', type=float, default=0.8,
                        help='Fraction of the train partition of each fold kept for training, the rest becomes the dev partition.')
    args = parser.parse_args()
    main(args)