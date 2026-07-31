from Bio.Align import substitution_matrices
import numpy as np
import torch 

SUB_MATRIX_NAME = 'BLOSUM62'

def compute_substitution_probs(sub_matrix) -> torch.Tensor:
    # Last column is *
    weights = 2**sub_matrix.select(sub_matrix.alphabet[:-1])
    # Keepdims needed for correct division
    probs = weights / np.sum(weights, -1, keepdims=True)
    return torch.as_tensor(probs, dtype=torch.float32)

def compute_esm_to_sub_matrix(sm_to_esm : torch.Tensor) -> torch.Tensor:
    # 32 is the <mask> token ID
    esm_to_blosum_id = torch.zeros(sm_to_esm.max() + 1, dtype=torch.int32) + 32
    for i in range(sm_to_esm.shape[0]):
        # print(f"{sm_to_esm[i]} -> {i}")
        esm_to_blosum_id[sm_to_esm[i]] = i

    return esm_to_blosum_id

sub_matrix = substitution_matrices.load('BLOSUM62')
sub_probs = compute_substitution_probs(sub_matrix)

id_to_res = ['G', 'A', 'V', 'L', 'I', 'T', 'S', 'M', 'C', 'P', 'F', 'Y', 'W', 'H', 'K', 'R', 'D', 'E', 'N', 'Q']

# List of valid ESM residue IDs
esm_valid_res_ids = list(range(4, 29))
# ESM token ID -> Residue mapping
esm_id_to_res_mapping = {4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C', 24: 'X', 25: 'B', 26: 'U', 27: 'Z', 28: 'O'}

# Residue -> ESM token ID mapping
esm_res_to_id_mapping = {v : k for k, v in esm_id_to_res_mapping.items()}

# Sub Matrix index -> ESM token ID mapping
sm_to_esm_id_mapping = torch.as_tensor([esm_res_to_id_mapping[r] for r in sub_matrix.alphabet if r in esm_res_to_id_mapping], dtype=torch.int32)

# ESM token ID mapping -> Sub Matrix index
esm_to_sm_id_mapping = compute_esm_to_sub_matrix(sm_to_esm_id_mapping)
