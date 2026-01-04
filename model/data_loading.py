# This file contains functions pertaining to preparing and loading data during training

import pandas as pd
import torch
import numpy as np
import json

from prot_dataset import FullProteinDataset, ProteinDataset
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from functools import partial
from ast import literal_eval
from constants import id_to_res, esm_valid_res_ids, esm_to_sm_id_mapping, sub_probs

def remove_long_sequences(df, max_length):
    mask = df['sequence'].apply(lambda x: len(x) < max_length)
    return df[mask]

def labeling_fn(row, residues={'S', 'T', 'Y'}, ignore_index=-1):
    res = np.zeros(len(row['sequence']), dtype=np.int32) + ignore_index
    mask = [s in residues for s in row['sequence']] # Only relevant prots are not ignored
    res[mask] = 0
    valid_sites = [i for i in row['sites'] if row['sequence'][i] in residues]
    res[valid_sites] = 1

    return res

def load_prot_data(dataset_path, residues={'S', 'T', 'Y'}, ignore_index=-1):
    """
    Loads the protein dataset and creates label vectors according to the 'sites' column, 
    stored in a new column 'label'. Returns a dataframe with columns 'id', 'sequence' and 'label'
    """
    df = pd.read_json(dataset_path)
    df = df.dropna()
    df['sites'] = df['sites'].apply(lambda x: [int(i) - 1 for i in x])
    labels = df.apply(partial(labeling_fn, residues=residues, ignore_index=ignore_index), axis=1)
    df['label'] = labels
    
    return df[['id', 'sequence', 'label']]

def perturb_seq(seq : str, mode=1, mask_token = '<mask>', mask_prob=0.15):
    # generate mask probabilites
    split_seq = seq.split()
    probs = np.random.rand(len(seq)) < mask_prob
    # Mask or perturb residue
    for i, p in enumerate(probs):
        if p < mask_prob:
            if mode == 1:
                split_seq[i] = np.random.randint(0, len(id_to_res))
            else:
                t = np.random.rand()
                if t < 0.3:
                    split_seq[i] = np.random.randint(0, len(id_to_res))
                    
                else:
                    split_seq[i] = mask_token

    return ''.join(split_seq)

def generate_random_residues(num_samples, residue_id_mapping):
    if num_samples == 0:
        return torch.Tensor([])

    choices = torch.randint(len(residue_id_mapping), (num_samples,))
    return torch.as_tensor(residue_id_mapping)[choices]

def generate_sm_residues(orig_res_ids : torch.Tensor, esm_to_sm_mapping):
    if orig_res_ids.numel() == 0:
        return orig_res_ids
    
    sm_indices = esm_to_sm_mapping[orig_res_ids]
    # By default, use a uniform distribution
    buffer = torch.zeros((orig_res_ids.numel(), sub_probs.shape[-1])) + 1

    # Some rare residues may not be included in the substitution matrix, get weights only for valid residues
    valid = sm_indices < sub_probs.shape[0]
    weights = sub_probs[sm_indices[valid]]
    buffer[valid] = weights
 
    return torch.multinomial(buffer, 1).squeeze()
    
def perturb_batch(
    input_ids: torch.Tensor,
    mask_token_id: int = 32,
    bos_token_id: int = 0,
    eos_token_id: int = 2,
    pad_token_id : int = 1,
    modify_prob: float = 0.15,
    mask_prob : float = 0.7,
    random_prob : float = 0.15,
    sub_prob : float = 0.15
) -> torch.Tensor:
    """
    Perturbs a batch of input token IDs for Masked Language Modeling (MLM).

    This function creates a copy of the input batch and modifies a percentage of 
    tokens based on the `modify_prob`. Special tokens (BOS, EOS, PAD) are excluded 
    from modification.
    
    Selected tokens are perturbed using one of three strategies, determined by the 
    weights provided in `mask_prob`, `random_prob`, and `blosum_prob`:
    1. Replaced with a mask token.
    2. Replaced with a random valid residue.
    3. Replaced with a similar residue based on the BLOSUM62 substitution matrix.

    Args:
        input_ids (torch.Tensor): The input tensor containing batches of token IDs.
        mask_token_id (int, optional): The ID used for masking. Defaults to 32.
        bos_token_id (int, optional): Beginning-of-sequence token ID. Defaults to 0.
        eos_token_id (int, optional): End-of-sequence token ID. Defaults to 2.
        pad_token_id (int, optional): Padding token ID. Defaults to 1.
        modify_prob (float, optional): The probability (0.0 to 1.0) of selecting a 
            token for modification. Defaults to 0.15.
        mask_prob (float, optional): The relative probability that a selected token 
            is replaced with `mask_token_id`. Defaults to 0.7.
        random_prob (float, optional): The relative probability that a selected token 
            is replaced with a random residue. Defaults to 0.15.
        sub_prob (float, optional): The relative probability that a selected token 
            is replaced via sub. matrix substitution. Defaults to 0.15.

    Returns:
        torch.Tensor: A new tensor of the same shape and dtype as `input_ids` 
        containing the perturbed sequences.
    """
    # Create a deep copy of the input_ids to modify
    masked_input_ids = input_ids.clone()

    modifiable = (input_ids != bos_token_id) & (input_ids != eos_token_id) & (input_ids != pad_token_id)
    
    # Create a tensor of random numbers between 0 and 1
    probability_matrix = torch.rand_like(input_ids, dtype=torch.float)
    
    # Only consider tokens that are actually maskable
    probability_matrix[~modifiable] = 1.0

    modified = probability_matrix < modify_prob

    choices = torch.multinomial(torch.Tensor([mask_prob, random_prob, sub_prob]), modified.numel(), replacement=True).reshape_as(modified)

    # Modify tokens according to generated choices

    # Mask tokens
    masked_input_ids[modified & (choices == 0)] = mask_token_id

    # Randomly perturb tokens
    masked_input_ids[modified & (choices == 1)] = generate_random_residues(torch.sum(modified & (choices == 1)), esm_valid_res_ids).to(masked_input_ids.dtype)

    # Perturb tokens according to a substitution matrix (in our case BLOSUM62)
    masked_input_ids[modified & (choices == 2)] = generate_sm_residues(masked_input_ids[modified & (choices == 2)], esm_to_sm_id_mapping).to(masked_input_ids.dtype)

    return masked_input_ids

def prep_batch(data, tokenizer, modify_prob=0, mask_prob=0.7, rand_prob=0.15, sub_prob=0.15, ignore_label=-1):
    """
    Collate function for a dataloader. "data" is a list of inputs.

    Return a dictionary with keys [input_ids, labels, batch_lens, indices]
    """
    # Indices are for the protein dataframe
    indices, sequences, labels = zip(*data)
    batch = tokenizer(sequences, padding='longest', return_tensors="pt")
    sequence_length = batch["input_ids"].shape[1]

    # Keep original ids as targets for MLM
    if modify_prob > 0 and mask_prob > 0:
        batch['orig_ids'] = batch['input_ids']

    batch['input_ids'] = perturb_batch(batch['input_ids'], modify_prob=modify_prob, random_prob=rand_prob,
                                        sub_prob=sub_prob, mask_prob=mask_prob)
    # Pad the labels correctly
    batch['labels'] = np.array([[ignore_label] + list(label) + [ignore_label] * (sequence_length - len(label) - 1) for label in labels])
    batch['labels'] = torch.as_tensor(batch['labels'], dtype=torch.float32)
    batch['batch_lens'] = torch.as_tensor(np.array([len(x) for x in labels]))
    batch['indices'] = torch.as_tensor(np.array(indices, dtype=np.int32))

    return batch

def load_clusters(path):
    return pd.read_csv(path, sep='\t', names=['cluster_rep', 'cluster_mem'])

def split_train_test_clusters(args, clusters : pd.DataFrame, test_size : float):
    reps = clusters['cluster_rep'].unique() # Unique cluster representatives
    train, test = train_test_split(reps, test_size=test_size, random_state=args.seed)
    return set(train), set(test)

def get_train_test_prots(clusters, train_clusters, test_clusters):
    train_mask = [x in train_clusters for x in clusters['cluster_rep']]
    test_mask = [x in test_clusters for x in clusters['cluster_rep']]
    train_prots = clusters['cluster_mem'][train_mask]
    test_prots = clusters['cluster_mem'][test_mask]
    return set(train_prots), set(test_prots)

def preprocess_data(df : pd.DataFrame):
    """
    Preprocessing for Pbert/ProtT5. Replaces rare residues with 'X' and adds spaces between residues
    """
    df['sequence'] = df['sequence'].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    df['sequence'] = df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    return df

def split_dataset(data : pd.DataFrame, train_clusters, test_clusters):
    """
    Splits data into train and test data according to train and test clusters.
    """
    train_mask = data['id'].apply(lambda x: x in train_clusters)
    test_mask = data['id'].apply(lambda x: x in test_clusters)
    return data[train_mask], data[test_mask]

def load_fasta(path : str):
    seq_iterator = SeqIO.parse(open(path), 'fasta')
    seq_dict = {}
    for seq in seq_iterator:
        # extract sequence id
        try:
            seq_id = seq.id.split('|')[0]
        except IndexError:
            # For some reason, some sequences do not contain uniprot ids, so skip them
            continue
        seq_dict[seq_id] = str(seq.seq)

    return seq_dict

def load_phospho(path : str):
    """
    Extracts phosphoryllation site indices from the dataset. 
    Locations expected in the column 'MOD_RSD'.
    
    Returns a dictionary in format {ACC_ID : [list of phosphoryllation site indices]}
    """
    dataset = pd.read_csv(path, sep='\t', skiprows=3)
    dataset['position'] = dataset['MOD_RSD'].str.extract(r'[\w]([\d]+)-p')
    grouped = dataset.groupby(dataset['ACC_ID'])
    res = {}
    for id, group in grouped:
        res[id] = group['position'].to_list()
    
    return res

def load_phospho_epsd(path : str):
    data = pd.read_csv(path, sep='\t')
    data.index = data['EPSD ID']
    grouped = data.groupby(data['EPSD ID'])

    res = {}
    for id, group in grouped:
        res[id] = group['Position'].to_list()

    return res

def prepare_datasets(args, ignore_label):
    prot_info = load_prot_data(args.prot_info_path, residues=literal_eval(args.residues), ignore_index=ignore_label)
    with open(args.dataset_path, 'r') as f:
        split_info = json.load(f)
    
    return FullProteinDataset(prot_info, split_info)


def prepare_full_dataset(args, ignore_label):
    prot_info = load_prot_data(args.prot_info_path, residues=literal_eval(args.residues), ignore_index=ignore_label)
    with open(args.dataset_path, 'r') as f:
        split_info = json.load(f)

    # gets the full dataset
    subset_ds = prot_info.loc[split_info[0]['train'] + split_info[0]['test']]
    return ProteinDataset(subset_ds)