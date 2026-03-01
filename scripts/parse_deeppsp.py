import pandas as pd
from Bio import SeqIO
from io import StringIO
import argparse

def parse_phospho_with_biopython(fasta_file):
    results = []
    
    # Use StringIO to make the string behave like a file
    records = SeqIO.parse(fasta_file, 'fasta')
    
    for record in records:
        raw_seq = str(record.seq)
        clean_seq_chars = []
        sites = []
        
        # current_pos tracks the 1-based index of the last added residue
        current_pos = 0
        
        for char in raw_seq:
            if char == '#':
                # The site is the residue immediately preceding the '#'
                sites.append(current_pos)
            else:
                clean_seq_chars.append(char)
                current_pos += 1
        
        results.append({
            'id': record.id,
            'sites': sites,
            'sequence': "".join(clean_seq_chars)
        })
        
    return pd.DataFrame(results)

def save_to_csv(df, filename):
    df.to_csv(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse phosphoprotein sequences and extract phospho sites.')
    parser.add_argument('--input_file', type=str, help='Path to the input FASTA file')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file')
    args = parser.parse_args()
    df = parse_phospho_with_biopython(args.input_file)
    save_to_csv(df, args.output_file)