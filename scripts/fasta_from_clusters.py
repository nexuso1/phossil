import argparse
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def export_cluster_fastas(clusters_path: str,
                          fasta_path: str,
                          output_path: str) -> None:
    """
    write a fasta containing only the cluster representatives found in
    *clusters_path*.

    *clusters_path* should be a tab‑separated file with two columns
    `cluster_rep` and `cluster_mem` (no header).
    *fasta_path* is the input FASTA; *output_path* is where the filtered
    FASTA will be written.
    """
    clusters = pd.read_csv(clusters_path,
                           sep='\t',
                           names=['cluster_rep', 'cluster_mem'])
    reps = set(clusters['cluster_rep'].astype(str))

    with open(output_path, 'w') as out_handle:
        for record in SeqIO.parse(fasta_path, 'fasta'):
            if record.id in reps:
                SeqIO.write(record, out_handle, 'fasta')

    print(f"Output saved in {output_path}")

def export_cluster_prot_info(clusters_path: str,
                          prot_info_path: str,
                          output_path: str) -> None:
    """
    write a fasta containing only the cluster representatives found in
    *clusters_path*.

    *clusters_path* should be a tab‑separated file with two columns
    `cluster_rep` and `cluster_mem` (no header).
    *fasta_path* is the input FASTA; *output_path* is where the filtered
    FASTA will be written.
    """
    clusters = pd.read_csv(clusters_path,
                           sep='\t',
                           names=['cluster_rep', 'cluster_mem'])
    
    prot_info = pd.read_json(prot_info_path)
    reps = set(clusters['cluster_rep'].astype(str))
    filtered = prot_info[prot_info['id'].apply(lambda x: x in reps)]
    with open(output_path, 'w') as out_handle:        
        for _, row in filtered.iterrows():
            record = SeqRecord(seq=Seq(row['sequence']), id=row['id'], name='', description='')
            SeqIO.write(record, out_handle, 'fasta')

    print(f"Output saved in {output_path}")

def main(args):
    if args.fasta:
        export_cluster_fastas(args.clusters, args.fasta, args.output)
    elif args.prot_info:
        export_cluster_prot_info(args.clusters, args.prot_info, args.output)

    else:
        print('No input file supplied.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='keep only cluster representatives in a FASTA file')
    parser.add_argument('--clusters', help='tsv file with cluster_rep,cluster_mem')
    parser.add_argument('--fasta', help='input FASTA', default=None)
    parser.add_argument('--prot_info', help='input prot info file, containing cols id, sequence', default=None)
    parser.add_argument('--output', help='filtered FASTA output path')
    args = parser.parse_args()

    main(args)