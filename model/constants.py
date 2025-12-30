from Bio.Align import substitution_matrices
import torch

id_to_res = ['G', 'A', 'V', 'L', 'I', 'T', 'S', 'M', 'C', 'P', 'F', 'Y', 'W', 'H', 'K', 'R', 'D', 'E', 'N', 'Q']
esm_valid_res_ids = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
esm_id_to_res = {4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C', 24: 'X', 25: 'B', 26: 'U', 27: 'Z', 28: 'O'}
esm_res_to_id = {v : k for k, v in esm_id_to_res.items()}
blosum = substitution_matrices.load('BLOSUM62')
blosum_to_esm_id = torch.as_tensor([esm_res_to_id[r] for r in blosum.alphabet if r in esm_res_to_id])
esm_to_blosum_id = torch.zeros(blosum_to_esm_id.max())

def prepare_blosum_mat():
    mat = as_tensor(blosum)
    mat = 2**mat
    mat = torch.s