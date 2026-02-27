import json
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import argparse
import pandas as pd
from pathlib import Path
import os

def create_fold_fastas(sequences_json, splits_json, prefix=None):
    # 1. Load the sequence data into a lookup dictionary {id: sequence_string}
    seq_data = pd.read_json(sequences_json)
    #seq_lookup = {i : entry['sequence'] for i, entry in seq_data.iterrows()}

    # 2. Load the split information
    with open(splits_json, 'r') as f:
        folds = json.load(f)

    # 3. Iterate through each fold (e.g., Fold 0, Fold 1, etc.)
    for i, fold in enumerate(folds):
        pardir = f'{prefix}_fastas/fold{i}'
        os.makedirs(pardir)
        for category in ['train', 'test']:
            records = []
            ids_in_split = fold.get(category, [])

            for df_id in ids_in_split:
                if df_id in seq_data.index:
                    # Create SeqRecord
                    record = SeqRecord(
                        Seq(seq_data.loc[df_id]['sequence']),
                        id=seq_data.loc[df_id]['id'],
                        description=""
                    )
                    records.append(record)
                else:
                    print(f"Warning: ID {df_id} found in splits but missing in sequence data.")

            # 4. Write the FASTA file for this specific fold and category
            filename = f"{pardir}/{prefix}_fold_{i}_{category}.fasta"
            if records:
                with open(filename, 'w') as output_handle:
                    SeqIO.write(records, output_handle, "fasta")
                print(f"Created {filename} with {len(records)} sequences.")

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', type=str)
    parser.add_argument('--splits', type=str)
    args = parser.parse_args()
    create_fold_fastas(args.seqs, args.splits, prefix=Path(args.splits).stem)