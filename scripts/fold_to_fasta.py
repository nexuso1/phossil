import json
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import argparse
import pandas as pd
from pathlib import Path
import os
import glob

def add_hashes(sequence, indices, residues):
    buffer = list(sequence)
    for i in indices:
        if sequence[i] in residues:
            buffer[i] = buffer[i] + '#'

    return "".join(buffer)

def create_fold_fastas(prot_info_path, splits_folder, prefix=None, hashes=False):
    # 1. Load the sequence data into a lookup dictionary {id: sequence_string}
    prot_info = pd.read_json(prot_info_path)
    #seq_lookup = {i : entry['sequence'] for i, entry in seq_data.iterrows()}

    split_paths = glob.glob(f'{splits_folder}/splits_*.json')
    splits = {}
    for split in split_paths:
        # 2. Load the split information
        with open(split, 'r') as f:
            folds = json.load(f)
        splits[Path(split).stem] = folds

    # 3. Iterate through each fold (e.g., Fold 0, Fold 1, etc.)
    for res in splits:
        for i, fold in enumerate(splits[res]):
            pardir = f'{splits_folder}/fastas/{res}_fold{i}'
            os.makedirs(pardir, exist_ok=True)
            for category in ['train', 'test']:
                records = []
                if hashes:
                    hash_records = []
                ids_in_split = fold.get(category, [])

                for df_id in ids_in_split:
                    if df_id in prot_info.index:
                        # Create SeqRecord
                        record = SeqRecord(
                            Seq(prot_info.loc[df_id]['sequence']),
                            id=prot_info.loc[df_id]['id'],
                            description=""
                        )
                        records.append(record)
                        if hashes:
                            with_hashes = add_hashes(prot_info.loc[df_id]['sequence'], [int(i) - 1 for i in prot_info.loc[df_id]['sites']], residues=set(list(res)))
                            hash_record = SeqRecord(
                                Seq(with_hashes),
                                id=prot_info.loc[df_id]['id'],
                                description=""
                            )
                            hash_records.append(hash_record)
                    else:
                        print(f"Warning: ID {df_id} found in splits but missing in sequence data.")

                # 4. Write the FASTA file for this specific fold and category
                filename = f"{pardir}/{category}.fasta"
                if records:
                    with open(filename, 'w') as output_handle:
                        SeqIO.write(records, output_handle, "fasta")
                    print(f"Created {filename} with {len(records)} sequences.")

                if hashes:
                    filename = f"{pardir}/{category}###.fasta"
                    with open(filename, 'w') as output_handle:
                        SeqIO.write(hash_records, output_handle, "fasta")
                    print(f"Created {filename} with {len(hash_records)} sequences.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', type=str)
    parser.add_argument('--splits', type=str, default=None)
    parser.add_argument('--hash', action='store_true', help='Also creates fastas with hashes after sites.')
    args = parser.parse_args()
    create_fold_fastas(args.seqs, args.splits, prefix=Path(args.splits).stem, hashes=args.hash)