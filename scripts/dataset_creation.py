import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
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

def collapse_fn(group : pd.DataFrame):
    #print(group['sequence'])
    #print(group)
    lens = group['sequence'].apply(lambda x: len(x))
    new_rep = lens.argmax()
    group['new_rep'] = group.iloc[new_rep].id
    return group

def reassign_representatives(data : pd.DataFrame):
    reps = set(data.cluster_rep.unique())
    members = set(data.id.unique())

    # Representatives that are not in the filtered dataset
    missing = set.difference(reps, members)
    grouped_to_fix = data[data['cluster_rep'].apply(lambda x: x in missing)].groupby('cluster_rep')
    fixed = grouped_to_fix.apply(collapse_fn)
    fixed = fixed.reset_index(level='cluster_rep', drop=True)
    data.loc[fixed.index, 'cluster_rep'] = fixed['new_rep']
    return data

def extract_representatives(data):
    reps = data['cluster_rep'].unique()
    rep_mask = data['id'].apply(lambda x : x in reps)
    return data[rep_mask]

def compute_length_labels(data):
    data['length'] = data['sequence'].apply(lambda x: len(x))
    _, length_bins = np.histogram(data['length'], 10)
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

def write_dataset_info(all_splits, out_folder):
    res = {}
    for label, splits in all_splits.items():
        res[label] = {}
        for i, fold in enumerate(splits):
            for data_type in fold:
                res[label][f'fold_{i}_{data_type}_size'] = len(fold[data_type])

    pd.DataFrame.from_dict(res).T.to_csv(os.path.join(out_folder, 'dataset_info.csv'), sep=',')

def split_data_per_residue(data, set_labels, seed=42):
    total_splits = {}
    for label in set_labels:
        filtered = filter_dataset(data, label)
        cv = StratifiedKFold(random_state=seed, shuffle=True)
        splits = []
        for train, test in cv.split(filtered.index, filtered['length_class']):
            orig_indices_train = filtered.index[train].tolist()
            orig_indices_test = filtered.index[test].tolist()
            splits.append({'train' : orig_indices_train, 'test' : orig_indices_test, 'total' : list(filtered.index)})

        total_splits[label] = splits
    return total_splits

def save_dataset(all_splits, out_folder):
    for label, splits in all_splits.items():
        with open(os.path.join(out_folder, f"splits_{label}.json"), 'w') as f:
            json.dump(splits, f)
        
def extract_representatives(data):
    reps = data['cluster_rep'].unique()
    rep_mask = data['id'].apply(lambda x : x in reps)
    return data[rep_mask]

def create_splits(prot_info_path, clusters_path, res_sets):
    prot_info = pd.read_json(prot_info_path)
    clusters = pd.read_csv(clusters_path, sep='\t', names=['cluster_rep', 'cluster_mem'])

    clusters_name = Path(clusters_path).stem
    out_folder = f'../data/dataset_{clusters_name}'
    os.makedirs(out_folder, exist_ok=True)

    # Join the cluster information with proteins that we have in the prot info
    # Some proteins from prot_info may have representatives that are not inside prot_info,
    # or they might be longer than max length for the base PLM
    joined = prot_info.join(clusters.set_index('cluster_mem'), on='id', how='left').drop_duplicates('id')
    joined = prune_long_seqs(joined)

    # Find the longest member of the cluster that is inside prot_info, and assign it as the representative
    joined = reassign_representatives(joined)

    # Plot cluster size distribution
    plot_name = 'cluster_dist.png'
    plot_info(joined, os.path.join(out_folder, plot_name))

    # Use only cluster representatives going forward
    joined = extract_representatives(joined)

    # Split sites according to residues in res_sets
    set_labels = ["".join(sorted(list(res_set))) for res_set in res_sets]
    joined = compute_site_labels(joined, res_sets, set_labels)

    # Compute length labels for stratification
    joined = compute_length_labels(joined)

    # Split data into folds using stratified k-fold cross-validation
    all_splits = split_data_per_residue(joined, set_labels)
    write_dataset_info(all_splits, out_folder)
    save_dataset(all_splits, out_folder)

def main(args):
    res_sets = eval(args.res_sets)
    create_splits(args.prot_info, args.clusters, res_sets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot_info', type=str, default='../data/phosphosite_sequences/phosphosite_df.json')
    parser.add_argument('--clusters', type=str, default='../data/clusters_cov1_c05.tsv')
    parser.add_argument('--res_sets', type=str, default="[{'S'}, {'T'}, {'Y'}, {'S', 'T'}, {'S', 'T', 'Y'}]")

    args = parser.parse_args()
    main(args)