# This file contains various utility functions and classes used throughout other parts of the project.

import pandas as pd
import numpy as np
import re
import os
import json
import torch

from torch.nn.functional import binary_cross_entropy_with_logits
from itertools import chain
from Bio import SeqIO
from transformers import EsmModel, AutoTokenizer, EsmForMaskedLM, AutoModel, AutoModelForMaskedLM


def get_esm(type, masked_lm=False):
    if type == '600M-C':
        repo_id = 'EvolutionaryScale/esmc-600m-2024-12'
        model_class = AutoModelForMaskedLM if masked_lm else AutoModel
        model = model_class.from_pretrained(repo_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        return model, tokenizer

    if masked_lm:
        model_class = EsmForMaskedLM
    else:
        model_class = EsmModel

    if type == '3B':
        model, tokenizer = model_class.from_pretrained('facebook/esm2_t36_3B_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')
    elif type == '15B':
        model, tokenizer = model_class.from_pretrained('facebook/esm2_t48_15B_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')
    elif type == '35M':
        model, tokenizer = model_class.from_pretrained('facebook/esm2_t12_35M_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    else:
        model, tokenizer = model_class.from_pretrained('facebook/esm2_t33_650M_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    return model, tokenizer


def load_torch_model(path):
    import torch
    with open(path, 'rb') as f:
        model = torch.load(f)

    return model

def flatten_list(lst):
    return list(chain(*lst))

def preprocess_data(df : pd.DataFrame):
    """
    Preprocessing for Pbert/ProtT5
    """
    df['sequence'] = df['sequence'].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    df['sequence'] = df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    return df

def save_as_string(obj, path):
    """
    Saves the given object as a JSON string.
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, 'w') as f:
        json.dump(obj, f)

class SimpleNamespace:
    def __init__(self, **kwargs) -> None:
        if len(kwargs.keys()) > 0:
            for k, v in kwargs.items():
                self.__setattr__(k, v)

class Metadata(SimpleNamespace):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if "data" not in self.__dict__.keys():
            self.data = {}

    def jsonify_fn(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return obj.__dict__
    
    def save(self, dir : str):
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'metadata.json'), 'w') as f:
            json.dump(self,f, default=self.jsonify_fn,  sort_keys=True, indent=4 )
            
# Taken from torchvision source
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    p = torch.sigmoid(inputs)
    ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss
