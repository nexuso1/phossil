from Bio.Align import substitution_matrices
import numpy as np

SUB_MATRIX_NAME = 'BLOSUM62'

def compute_substitution_probs(sub_matrix):
    # Last column is *
    weights = 2**sub_matrix[:-1, :-1]
    probs = weights / np.sum(weights, -1)
    return probs

sub_matrix = substitution_matrices.load('BLOSUM62')
sub_probs = compute_substitution_probs(sub_matrix)