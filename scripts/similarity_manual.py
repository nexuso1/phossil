from Bio.Align import PairwiseAligner
from Bio import SeqIO
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm.contrib.concurrent import process_map


def load_sequences(path):
    recs = []
    with open(path, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            recs.append((record.id, record.seq))

    return pd.DataFrame.from_records(recs, columns=['id', 'sequence'])


def compute_similarity_matrix(sequencesA, sequencesB,max_workers=None, chunk_size=None):
    res_global = np.zeros(shape=(len(sequencesA), len(sequencesB)), dtype=float)
    res_shorter = np.zeros(shape=(len(sequencesA), len(sequencesB)), dtype=float)

    tasks = [
        (i, j, sequencesA[i], sequencesB[j])
        for i in range(len(sequencesA))
        for j in range(len(sequencesB))
    ]

    task_pairs = [(seqA, seqB) for _, _, seqA, seqB in tasks]
    print(len(task_pairs))
    results = process_map(
        compute_identity,
        task_pairs,
        desc='Computing similarities',
        smoothing=0.05,
        max_workers=max_workers,
        chunksize=chunk_size,
    )

    for (i, j, _, _), (sim_global, sim_shorter) in zip(tasks, results):
        res_global[i, j] = sim_global
        res_shorter[i, j] = sim_shorter

    return res_global, res_shorter

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
    seq1_aligned = best_alignment[0, :]
    seq2_aligned = best_alignment[1, :]

    # 4. Count the identical positions
    matches = sum(1 for a, b in zip(seq1_aligned, seq2_aligned) if a == b and a != '-')

    # 5. Calculate Sequence Identity

    # Method A: Divide by the total alignment length
    alignment_length = len(seq1_aligned)
    identity_by_alignment = matches / alignment_length

    # Method B: Divide by the length of the shortest sequence
    min_length = min(len(seqA), len(seqB))
    identity_by_shortest = matches / min_length

    if verbose > 0:
        # Display Results
        print(f"Aligned Sequence 1: {seq1_aligned}")
        print(f"Aligned Sequence 2: {seq2_aligned}")
        print(f"Total Matches:      {matches}")
        print(f"Identity (Alignment Length): {identity_by_alignment:.1%}")
        print(f"Identity (Shortest Length):  {identity_by_shortest:.1%}")

    return identity_by_alignment, identity_by_shortest

def save_matrix(matrix, idsA, idsB, out_path):
    df = pd.DataFrame(matrix, index=idsA, columns=idsB)
    df.to_csv(out_path)

def compute_matrix_info(matrix):
    metrics = {
        'mean' : np.mean,
        'std' : np.std,
        'median' : np.median,
        'max' : np.max
    }

    return { name : metric(matrix) for name, metric in metrics.items()}

def save_result_summary(glob_matrix, loc_matrix, out_path):
    pd.DataFrame.from_dict({
        'global' : compute_matrix_info(glob_matrix),
        'local' : compute_matrix_info(loc_matrix)
    }).T.to_csv(out_path, float_format='%.4f')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seqsA', type=str)
    parser.add_argument('seqsB', type=str)
    parser.add_argument('--max_workers', type=int, default=15)
    parser.add_argument('--chunk_size', type=int, default=8)
    parser.add_argument('-o', type=str, help='Output path')
    args = parser.parse_args()
    seqsA = load_sequences(args.seqsA)
    seqsB = load_sequences(args.seqsB)

    glob_matrix, loc_matrix = compute_similarity_matrix(seqsA.sequence.to_list(), seqsB.sequence.to_list(), max_workers=args.max_workers, chunk_size=args.chunk_size)
    out_path_obj = Path(args.o)

    for matrix, suffix in zip((glob_matrix, loc_matrix), ('_global_sim_matrix', '_local_sim_matrix')):
        save_matrix(matrix, idsA=seqsA.id.to_list(), idsB=seqsB.id.to_list(), out_path=out_path_obj.with_stem(out_path_obj.stem + suffix))

    save_result_summary(glob_matrix, loc_matrix, out_path_obj.with_stem(out_path_obj.stem + '_summary'))