
import pandas as pd
import numpy as np
import argparse
import os
import json
import glob

from multiprocessing import Pool
from pathlib import Path
from Bio.Align import PairwiseAligner
from Bio import SeqIO
from Bio.Seq import Seq

def load_sequences(path_or_seq, num=None):
    recs = []
    try:
        with open(path_or_seq, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                recs.append((record.id, record.seq))

        return pd.DataFrame.from_records(recs, columns=['id', 'sequence'])
    
    except OSError or FileNotFoundError:
        id = id=f'Sequence {num}'
        seq = Seq(path_or_seq)
        return pd.DataFrame.from_records([(id, seq)], columns=['id', 'sequence'])

def task_func(args : tuple[int, int, Seq, Seq]):
    i, j, seqA, seqB= args
    return i, j, compute_identity((seqA, seqB))

def compute_similarity_matrix(sequencesA, sequencesB, out_path, max_workers=None, chunk_size=None):
    res_global = np.zeros(shape=(len(sequencesA), len(sequencesB)))
    res_shortest, res_shortest_gapped = np.zeros_like(res_global), np.zeros_like(res_global)
    records = []

    tasks = [(i, j, sequencesA[i], sequencesB[j]) for j in range(len(sequencesB)) for i in range(len(sequencesA))]
    count = 0
    with Pool(max_workers) as pool:
        for i, j, result in pool.imap_unordered( task_func, tasks, chunksize=chunk_size):
        
            result['idxA'] = i
            result['idxB'] = j
            
            res_global[i, j] = result['identity_global']
            res_shortest[i, j] = result['identity_shortest']
            res_shortest_gapped[i, j] = result['identity_shortest_gapped']
            records.append(result)
            count += 1

            if count % 10000 == 0:
                with open(f'{str(out_path)}/identity_chunk_{count}.json', 'w') as f:
                    json.dump(records, f)
                    records.clear()

    with open(f'{str(out_path)}/identity_chunk_{count}.json', 'w') as f:
        json.dump(records, f)
        records.clear()

    return res_global, res_shortest, res_shortest_gapped

def get_aligner():
        aligner = PairwiseAligner()
        aligner.mode = 'global'

        return aligner
def compute_identity(sequences : tuple[str, str], verbose=0):
    # 1. Set up the aligner and perform the alignment
    seqA, seqB = sequences

    aligner = get_aligner()
    alignments = aligner.align(seqA, seqB)

    # 2. Grab the highest scoring alignment (the first one returned)
    best_alignment = alignments[0]

    # 3. Extract the aligned strings with gaps
    # Index 0 is the target (seqA), Index 1 is the query (seqB).
    # The ':' slice gets the entire sequence.
    seqA_aligned = best_alignment[0, :]
    seqB_aligned = best_alignment[1, :]

    # 4. Count the identical positions
    matches = sum(1 for a, b in zip(seqA_aligned, seqB_aligned) if a == b and a != '-')

    # 5. Calculate Sequence Identity

    # Method A: Divide by the total alignment length
    alignment_length = len(seqA_aligned)
    identity_by_alignment = matches / alignment_length

    # Method B: Divide by the length of the shortest sequence
    min_length = min(len(seqA), len(seqB))
    identity_by_shortest = matches / min_length

    # Method C: Divide by the length of the gapped alignment of the shorter sequence
    shorter_aligned_seq = seqA_aligned if len(seqA) < len(seqB) else seqB_aligned
    stripped = str(shorter_aligned_seq).strip('-')
    identity_by_shortest_gapped_length = matches / len(stripped)

    if verbose > 0:
        # Display Results
        print(f"Aligned Sequence 1: {seqA_aligned}")
        print(f"Aligned Sequence 2: {seqB_aligned}")
        print(f"Total Matches:      {matches}")
        print(f"Identity (Alignment Length): {identity_by_alignment:.1%}")
        print(f"Identity (Shortest Length):  {identity_by_shortest:.1%}")
        print(f"Identity (Shortest Aligned Gapped Length):  {identity_by_shortest_gapped_length:.1%}")

    return {
        'score' : best_alignment.score,
        'identity_global' : identity_by_alignment,
        'identity_shortest' : identity_by_shortest,
        'identity_shortest_gapped' : identity_by_shortest_gapped_length,
        'seqA_aligned' : str(seqA_aligned),
        'seqB_aligned' : str(seqB_aligned) 
    }

def save_detailed_results(result_df : pd.DataFrame, idxA, idxB, idsA, idsB, out_path):
    result_df['id_seqA'] = [idsA[i] for i in idxA]
    result_df['id_seqB'] = [idsB[i] for i in idxB]
    result_df.to_parquet(out_path.with_suffix('.parquet.zst'), compression='zstd')

def compute_matrix_info(matrix):
    metrics = {
        'mean' : np.mean,
        'std' : np.std,
        'median' : np.median,
        'max' : np.max
    }

    return { name : metric(matrix) for name, metric in metrics.items()}

def save_matrix(matrix, idsA, idsB, out_path : Path):
    df = pd.DataFrame(matrix, index=idsA, columns=idsB)
    df.to_parquet(out_path.with_suffix('.parquet.zst'), compression='zstd')

def save_result_summary(glob_matrix, loc_matrix, align_matrix, out_path : Path):
    pd.DataFrame.from_dict({
        'global' : compute_matrix_info(glob_matrix),
        'local' : compute_matrix_info(loc_matrix),
        'local_gapped' : compute_matrix_info(align_matrix),
    }).T.to_csv(out_path.with_suffix('.csv'), float_format='%.4f', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seqsA', type=str, help='Path to a .fasta file or a single sequence')
    parser.add_argument('seqsB', type=str, help='Path to a .fasta file or a single sequence')
    parser.add_argument('--save_matrices', action='store_true', help='Save individual identity matrices as separate files. Optional, because pairwise identity information is already included in the detailed summary file.')
    parser.add_argument('--max_workers', type=int, default=15)
    parser.add_argument('--chunk_size', type=int, default=256, help='Number of sequence pairs passed to a worker subprocess')
    parser.add_argument('-o', type=str, help='Output folder')
    parser.add_argument('-n', type=str, help='Output base name')
    args = parser.parse_args()
    seqsA = load_sequences(args.seqsA, 'A')
    seqsB = load_sequences(args.seqsB, 'B')
    out_dir = Path(args.o)
    os.makedirs(out_dir, exist_ok=True)
    
    glob_matrix, loc_matrix, local_gapped = compute_similarity_matrix(seqsA.sequence.to_list(), seqsB.sequence.to_list(), max_workers=args.max_workers, chunk_size=args.chunk_size, out_path=out_dir)

    name = args.n

    
    idsA=seqsA.id.to_list()
    idsB=seqsB.id.to_list()
    if args.save_matrices:
        for matrix, suffix in zip((glob_matrix, loc_matrix, local_gapped), ('_global_sim_matrix', '_local_sim_matrix', '_local_gapped_sim_matrix')):
            save_matrix(matrix, idsA=idsA, idsB=idsB, out_path=out_dir / f'{name}{suffix}')

    save_result_summary(glob_matrix, loc_matrix, local_gapped, out_dir / f'{name}_sim_summary')

    for path in glob.glob(f'{str(out_dir)}/identity_chunk*'):
        count = Path(path).name.removesuffix('.json').split('_')[-1]
        with open(path, 'r') as f:
            records = json.load(f)

        detailed_results = pd.DataFrame.from_records(records)
        save_detailed_results(
            detailed_results.drop(columns=['idxA', 'idxB']),
            idxA=detailed_results.idxA.to_list(),
            idxB=detailed_results.idxB.to_list(),
            idsA=seqsA.id,
            idsB=seqsB.id, 
            out_path=out_dir / f'{name}_detailed_results_{count}'
        )