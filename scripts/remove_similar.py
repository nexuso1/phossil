import pandas as pd
import json
import os
from pathlib import Path
import argparse

def find_similar_prots(similarity_df : pd.DataFrame, prot_info, threshold=0.3):
    # Read the similarity file into a DataFrame
    df = similarity_df[similarity_df['query_id'] != similarity_df['target_id']]
    to_remove_ids = set()
    for _, row in df.iterrows():
        if row['similarity'] > threshold:
            to_remove_ids.add(row['target_id'])
            
    to_remove_indices = set(prot_info[prot_info['id'].apply(lambda x: x in to_remove_ids)].index)
    print(f"Number of target (train) proteins removed: {len(to_remove_ids)}")

    return to_remove_ids, to_remove_indices

def remove_similar_prots_splits(similarity_path, prot_info_path, splits_path, threshold=0.3):
    similarity_df = pd.read_csv(similarity_path, sep="\t", names=['query_id', 'target_id', 'similarity', 'alnlen', 'qlen', 'tlen'])
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    prot_info = pd.read_json(prot_info_path)

    if 'total' in splits[0]:
        indices = [int(i) for i in splits[0]['total']]
    else:
        indices = set()
        # In order to ensure that truly all unique indices are captured
        for split in splits:
            for idx in split['test']:
                indices.add(idx)

            for idx in split['train']:
                indices.add(idx)

        indices = list(indices)

    ids  = set(prot_info.loc[indices, 'id'])
    relevant_similarities = similarity_df[similarity_df.apply(lambda row: row['query_id'] in ids and row['target_id'] in ids, axis=1)]
    _, indices_to_remove = find_similar_prots(relevant_similarities, prot_info, threshold)
    kept_indices = list(set(indices) - indices_to_remove)

    
    
    for i in range(len(splits)):
        keep = []
        for j in splits[i]['train']:
            if int(j) in indices_to_remove:
                continue
            else:
                keep.append(j)
        splits[i]['train'] = keep
        splits[i]['total'] = kept_indices
        print(f'Kept {len(keep)} train proteins in fold {i}')
    out_folder = Path(splits_path).parent
    fname = f'{Path(splits_path).stem}_filtered.json'                            
    with open(os.path.join(out_folder, fname), 'w') as f:
        json.dump(splits, f, indent='\t')

def find_all_pairs(grouped_similarity):
    all_known_bad_pairs = {}
    for prot in grouped_similarity.index:
        if not prot in all_known_bad_pairs:
            all_known_bad_pairs[prot] = set()
        
        all_known_bad_pairs[prot] = all_known_bad_pairs[prot] | grouped_similarity.loc[prot]['target_id']
        
        for target in grouped_similarity.loc[prot]['target_id']:
            if target not in all_known_bad_pairs:
                all_known_bad_pairs[target] = set()
            
            all_known_bad_pairs[target].add(prot)

    return all_known_bad_pairs

def remove_prots_sequentially(grouped):
    grouped['n_similar'] = grouped['target_id'].apply(lambda x: len(x))
    grouped = grouped.sort_values('n_similar', ascending=False)
    removed = []

    while grouped['n_similar'].iloc[0] > 0:
        # Remove the protein with most similar pairs
        to_update = list(grouped.iloc[0]['target_id'])
        removed.append(grouped.iloc[0].name)
        grouped = grouped.iloc[1:]

        # Update related counts
        grouped.loc[to_update, 'n_similar'] = grouped.loc[to_update, 'n_similar'] - 1

        # Update related sets
        sets = [grouped.loc[prot]['target_id'] for prot in to_update]
        for s in sets:
            s.discard(removed[-1])

        grouped.loc[to_update, 'target_id'] = sets

        # Reorder the data
        grouped = grouped.sort_values('n_similar', ascending=False)

    return removed

def remove_similar_prots_sequential(similarity_path : pd.DataFrame, clusters_path, out_path, threshold=0.3):
    similarity_df = pd.read_csv(similarity_path, sep="\t", names=['query_id', 'target_id', 'similarity', 'alnlen', 'qlen', 'tlen'])
    clusters = pd.read_csv(clusters_path, sep='\t', names=['cluster_rep', 'cluster_mem'])
    df = similarity_df[similarity_df['similarity'] > threshold]
    grouped = df.groupby('query_id').agg(set)

    # remove references to the same prot
    for prot in grouped.index:
        grouped.loc[prot]['target_id'].remove(prot)

    all_known_bad_pairs = find_all_pairs(grouped)
    grouped.loc[all_known_bad_pairs.keys(), 'target_id'] = list(all_known_bad_pairs.values())
    to_remove = set(remove_prots_sequentially(grouped))
    filtered_clusters = clusters[clusters['cluster_rep'].apply(lambda x: x not in to_remove)]
    filtered_clusters.to_csv(out_path, sep='\t')
    print(f'Removed {len(to_remove)} representatives. New total is {len(filtered_clusters['cluster_rep'].unique())}. Filtered clusters saved to {out_path}')
    return to_remove

def main(args):
    if args.mode == 'splits':
        remove_similar_prots_splits(args.similarity_file, args.prot_info, args.splits, args.threshold)
    else:
        remove_similar_prots_sequential(args.similarity_file, args.clusters, args.out_path, args.threshold)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove similar pairs from a similarity file based on a threshold.')
    parser.add_argument('--similarity_file', type=str, help='Path to the similarity file (tab-separated)')
    parser.add_argument('--prot_info', type=str, help='Protein info file path (contains id column, indices in splits are indices into this file)', default=None)
    parser.add_argument('--splits', type=str, help='Path to the splits file', default=None)
    parser.add_argument('--clusters', type=str, help='Clusters csv file', default=None)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--mode', default='sequential', choices=['sequential', 'splits'])
    parser.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold for filtering pairs (default: 0.3)')
    args = parser.parse_args()

    main(args)