# This file contains functions pertaining to preparing and loading data during training

import pandas as pd
import torch
import numpy as np
import json

from prot_dataset import FullProteinDataset, ProteinDataset
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from Bio.Align import substitution_matrices
from functools import partial
from ast import literal_eval
from constants import sub_matrix, sub_probs

id_to_res = ['G', 'A', 'V', 'L', 'I', 'T', 'S', 'M', 'C', 'P', 'F', 'Y', 'W', 'H', 'K', 'R', 'D', 'E', 'N', 'Q']

def remove_long_sequences(df, max_length):
    mask = df['sequence'].apply(lambda x: len(x) < max_length)
    return df[mask]

def labeling_fn(row, residues={'S', 'T', 'Y'}, ignore_index=-1):
    res = np.zeros(len(row['sequence']), dtype=np.uint8) + ignore_index
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

def mask_batch_for_mlm(
    input_ids: torch.Tensor,
    mask_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id : int,
    modify_prob: float = 0.15,
    masking_prob : float = 0.7,
    random_residue_prob = 0.15,
    blosum_residue_prob = 0.15
) -> torch.Tensor:
    """
    Randomly replaces certain tokens in a batch of input IDs with a mask token.

    Excludes padding, BOS, and EOS tokens from the masking process.

    Args:
        input_ids (torch.Tensor): The batch of token IDs (B x L).
        lengths (torch.Tensor): The actual length of each sequence (B).
        mask_token_id (int): The ID of the mask token.
        bos_token_id (int): The ID of the Beginning-of-Sentence token.
        eos_token_id (int): The ID of the End-of-Sentence token.
        masking_probability (float): The probability of masking a token.

    Returns:
        torch.Tensor: The input_ids tensor with selected tokens replaced by mask_token_id.
    """
    # Create a deep copy of the input_ids to modify
    masked_input_ids = input_ids.clone()
    mask_thresh = masking_prob
    random_thresh = mask_thresh + random_residue_prob

    modifiable = input_ids != bos_token_id & input_ids != eos_token_id & input_ids != pad_token_id
    
    # Create a tensor of random numbers between 0 and 1
    probability_matrix = torch.full(input_ids.shape, modify_prob, dtype=torch.float)
    
    # Only consider tokens that are actually maskable
    probability_matrix[~modifiable] = 0.0

    # Draw a mask based on the probability matrix
    to_modify = torch.bernoulli(probability_matrix).bool()

    modifications = torch.multinomial([masking_prob, random_residue_prob, blosum_residue_prob], torch.sum(to_modify), replacement=True)

    return masked_input_ids

def perturb_residues(orig_ids, masked_ids, mask_token_id, random_prob=0.3, blos_prob=0):
    rand_vals = torch.multinomial([1 - (random_prob + blos_prob), random_prob, blos_prob], )
    
def prep_batch(data, tokenizer, perturb_mode=0, ignore_label=-1):
    """
    Collate function for a dataloader. "data" is a list of inputs.

    Return a dictionary with keys [input_ids, labels, batch_lens, indices]
    """
    # Indices are for the protein dataframe
    indices, sequences, labels = zip(*data)
    if perturb_mode > 0:
        sequences = [perturb_seq(seq, perturb_mode) for seq in sequences]
    batch = tokenizer(sequences, padding='longest', return_tensors="pt")
    sequence_length = batch["input_ids"].shape[1]
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