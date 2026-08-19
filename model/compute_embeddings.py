# Computes last hidden layer embeddings of a protein dataset, one compressed file per protein.
# Meant for comparing the representations of a trained model against the pretrained ones.
# --residues restricts what is written to the candidate residues of the task (S/T/Y), which is most
# of the size of the output for a fraction of the information.

import argparse
import numpy as np
import os
import pandas as pd
import torch

from collections import defaultdict
from data_loading import parse_residues
from token_classifier_base import TokenClassifier
from utils import get_esm

def add_chunk_columns(data : pd.DataFrame):
    """
    Fills in the columns added by scripts/chunk_proteins.py, so that datasets of whole proteins
    (where every protein is its own single chunk) go through the same code path.
    """
    if 'parent_id' not in data:
        data['parent_id'] = data['id']

    if 'offset' not in data:
        data['offset'] = 0

    if 'parent_length' not in data:
        data['parent_length'] = data['sequence'].apply(len)

    return data

def load_base_model(args, device):
    """
    Returns the embedding model whose last hidden layer is saved, together with its tokenizer.

    Without a model path this is the pretrained ESM of the given type. With one it is the base model
    of a saved TokenClassifier, either pickled directly or saved via TokenClassifier.save().
    """
    if not args.model_path:
        base, tokenizer = get_esm(args.type)
        return base.to(device), tokenizer

    # weights_only=False, the saved config is a pickled dataclass
    saved = torch.load(args.model_path, weights_only=False, map_location=device)
    if isinstance(saved, TokenClassifier):
        _, tokenizer = get_esm(saved.config.base_type)
        return saved.base.to(device), tokenizer

    config, state_dict = saved['config'], saved['state_dict']
    base, tokenizer = get_esm(config.base_type)

    if any('lora_' in key for key in state_dict):
        # peft wraps the base model, the adapters have to be in place before the weights are loaded
        from lora_model import LoRAClassifier
        model = LoRAClassifier(config=config, base_model=base)
        model.load_state_dict(state_dict)
        return model.base.to(device), tokenizer

    # Only the base model is needed, the classification head is not part of the embedding
    base_state = {key.removeprefix('base.') : value for key, value in state_dict.items()
                  if key.startswith('base.')}
    base.load_state_dict(base_state)
    return base.to(device), tokenizer

def embed_batch(base, tokenizer, sequences : list[str], device):
    """
    Last hidden layer embeddings of a batch of sequences, without the [CLS] and [EOS] tokens.
    Returns one (sequence length, hidden size) array per sequence.
    """
    batch = tokenizer(sequences, padding='longest', return_tensors='pt').to(device)
    with torch.no_grad():
        hidden_states = base(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[0]

    # Token 0 is always the beginning-of-sequence token, so the first residue is token 1
    return [hidden_states[i, 1:len(sequence) + 1].cpu().numpy() for i, sequence in enumerate(sequences)]

def stitch_chunk_embeddings(chunks : list, parent_length : int):
    """
    Lays the embeddings of the chunks of one protein onto whole protein positions. A position
    covered by several overlapping chunks keeps the embedding of the chunk that saw it closest to
    its center, the same rule stitch_chunk_predictions uses.

    "chunks" is a list of (offset, embeddings, sequence, chunk id). Returns (embeddings, sequence).
    """
    hidden_size = chunks[0][1].shape[-1]
    embeddings = np.zeros((parent_length, hidden_size), dtype=chunks[0][1].dtype)
    sequence = np.empty(parent_length, dtype='<U1')
    # Distance of the position from the center of the chunk it currently comes from
    best_distance = np.full(parent_length, np.inf)

    for offset, chunk_embeddings, chunk_sequence, _ in chunks:
        positions = np.arange(len(chunk_embeddings)) + offset
        distance = np.abs(np.arange(len(chunk_embeddings)) - (len(chunk_embeddings) - 1) / 2)
        closer = distance < best_distance[positions]

        embeddings[positions[closer]] = chunk_embeddings[closer]
        best_distance[positions[closer]] = distance[closer]
        sequence[offset:offset + len(chunk_sequence)] = list(chunk_sequence)

    return embeddings, ''.join(sequence)

def select_positions(sequence : str, residues : set|None, offset=0):
    """
    Positions of the residues that are kept, in whole protein coordinates. Without a residue set
    every position is kept, so the rest of the code has one case to handle.
    """
    if residues is None:
        return np.arange(len(sequence)) + offset

    return np.array([i + offset for i, res in enumerate(sequence) if res in residues], dtype=int)

def save_protein(protein_id, chunks, parent_length, out_folder, dtype, residues=None, save_chunks=False):
    """
    Writes the embeddings of one protein. Returns the number of positions saved, zero meaning the
    protein has no residue of the selected types and no file is written.
    """
    chunks = sorted(chunks, key=lambda chunk: chunk[0])
    embeddings, sequence = stitch_chunk_embeddings(chunks, parent_length)
    # Whole protein positions of the saved rows, "embeddings"[j] is the embedding of residue
    # "sequence"[positions[j]]. The full sequence is kept either way, it costs nothing.
    positions = select_positions(sequence, residues)
    if len(positions) == 0:
        return 0

    arrays = {
        'embeddings' : embeddings[positions].astype(dtype),
        'positions' : positions,
        'sequence' : sequence,
        'id' : protein_id,
    }

    if save_chunks:
        # Chunks as they were embedded, keeping the copies stitching drops. Overlapping chunks see
        # a residue with different context around it, so "chunk_<i>" is how chunk i embedded the
        # positions listed in "chunk_positions_<i>", again in whole protein coordinates.
        arrays['chunk_offsets'] = np.array([offset for offset, _, _, _ in chunks])
        arrays['chunk_ids'] = np.array([chunk_id for _, _, _, chunk_id in chunks])
        for i, (offset, chunk_embeddings, chunk_sequence, _) in enumerate(chunks):
            chunk_positions = select_positions(chunk_sequence, residues, offset=offset)
            arrays[f'chunk_{i}'] = chunk_embeddings[chunk_positions - offset].astype(dtype)
            arrays[f'chunk_positions_{i}'] = chunk_positions

    np.savez_compressed(os.path.join(out_folder, f'{protein_id}.npz'), **arrays)

    return len(positions)

def compute_embeddings(args):
    device = torch.device(args.device)
    base, tokenizer = load_base_model(args, device)
    base.eval()

    data = add_chunk_columns(pd.read_json(args.prot_info).dropna())
    os.makedirs(args.out_folder, exist_ok=True)

    residues = set(parse_residues(args.residues)) if args.residues else None
    if residues is not None:
        print(f'Saving the embeddings of {"".join(sorted(residues))} residues only')

    # A protein is saved once all of its chunks have been embedded
    n_chunks = data.groupby('parent_id').size().to_dict()
    parent_lengths = data.groupby('parent_id')['parent_length'].first().to_dict()
    pending = defaultdict(list)
    done, empty = 0, 0

    for start in range(0, len(data), args.batch_size):
        rows = data.iloc[start:start + args.batch_size]
        embeddings = embed_batch(base, tokenizer, rows['sequence'].to_list(), device)

        for row, chunk_embeddings in zip(rows.itertuples(), embeddings):
            pending[row.parent_id].append((int(row.offset), chunk_embeddings, row.sequence, row.id))

        for protein_id in [p for p in pending if len(pending[p]) == n_chunks[p]]:
            saved = save_protein(protein_id, pending.pop(protein_id), int(parent_lengths[protein_id]),
                                 args.out_folder, args.dtype, residues=residues,
                                 save_chunks=args.save_chunks)
            done += saved > 0
            empty += saved == 0

        print(f'{min(start + args.batch_size, len(data))}/{len(data)} chunks, {done} proteins saved', end='\r')

    print(f'\nSaved embeddings of {done} proteins to {args.out_folder}')
    if empty:
        print(f'Skipped {empty} proteins without a single selected residue')

def main(args):
    compute_embeddings(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute last hidden layer embeddings of a protein dataset.')
    parser.add_argument('--prot_info', type=str, default='../data/dbptm/dbptm_info_chunked.json',
                        help='Path to the protein info file, a dataframe with columns ("id", "sequence").')
    parser.add_argument('--out_folder', type=str, required=True,
                        help='Folder the per protein embedding files are written to.')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Saved TokenClassifier to take the base model from. Without it the pretrained model of --type is used.')
    parser.add_argument('--type', type=str, default='650M',
                        help='ESM model type, used when no --model_path is given.')
    parser.add_argument('--residues', type=str, default=None,
                        help='Residues to save the embeddings of, e.g. "STY" or "[\'S\', \'T\', \'Y\']". Every position is saved by default.')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of chunks embedded at once')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'],
                        help='Data type the embeddings are stored in. Half precision halves the size of the output.')
    parser.add_argument('--save_chunks', action='store_true', default=False,
                        help='Also store the embeddings of every chunk separately, keeping the copies of the shared positions that stitching drops.')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(args)
