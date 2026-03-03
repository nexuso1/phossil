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
        if row['similarity'] > threshold and row['query_id'] not in to_remove_ids:
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

    print(f'Kept {len(kept_indices)} train proteins')
    
    for i in range(len(splits)):
        keep = []
        for j in splits[i]['train']:
            if int(j) in indices_to_remove:
                continue
            else:
                keep.append(j)
        splits[i]['train'] = keep
        splits[i]['total'] = kept_indices

    out_folder = Path(splits_path).parent
    fname = f'{Path(splits_path).stem}_filtered.json'                            
    with open(os.path.join(out_folder, fname), 'w') as f:
        json.dump(splits, f, indent='\t')

def main(args):
    remove_similar_prots_splits(args.similarity_file, args.prot_info, args.splits, args.threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove similar pairs from a similarity file based on a threshold.')
    parser.add_argument('--similarity_file', type=str, help='Path to the similarity file (tab-separated)')
    parser.add_argument('--prot_info', type=str, help='Protein info file path (contains id column, indices in splits are indices into this file)')
    parser.add_argument('--splits', type=str, help='Path to the splits file')
    parser.add_argument('--threshold', type=float, default=0.3, help='Similarity threshold for filtering pairs (default: 0.3)')
    args = parser.parse_args()

    main(args)