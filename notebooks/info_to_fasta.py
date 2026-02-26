import json
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import argparse
import pandas as pd
from pathlib import Path
 
def json_to_fasta(json_file, fasta_output):
    data = pd.read_json(json_file)

    # 2. Create a list to hold SeqRecord objects
    protein_records = []

    # 3. Iterate through your JSON entries
    # Adjust 'id' and 'sequence' keys based on your specific JSON structure
    for i, entry in data.iterrows():
        record = SeqRecord(
            Seq(entry['sequence']),
            id=entry.get('id', 'unknown_id'),
            description=entry.get('description', '')
        )
        protein_records.append(record)

    # 4. Write to FASTA file
    with open(fasta_output, 'w') as output_handle:
        SeqIO.write(protein_records, output_handle, "fasta")
    
    print(f"Successfully converted {len(protein_records)} sequences to {fasta_output}")

# Usage
if __name__ == "__main__":
    # Change these filenames to match your files
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str)
    args = parser.parse_args()
    json_to_fasta(args.i, f"{Path(args.i).stem}.fasta")