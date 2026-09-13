# Computes last hidden layer embeddings of a protein dataset, one compressed file per protein.
# Meant for comparing the representations of a trained model against the pretrained ones.
# --residues restricts what is written to the candidate residues of the task (S/T/Y), which is most
# of the size of the output for a fraction of the information.
# --attention_layers additionally saves the attention weights of the listed layers, for attention
# visualization and score distribution analysis. Those maps are quadratic in the sequence length,
# see save_protein for what exactly is written.

import argparse
import numpy as np
import os
import pandas as pd
import torch

from ast import literal_eval
from collections import defaultdict
from data_loading import parse_residues
from token_classifier_base import TokenClassifier
from typing import NamedTuple
from utils import get_esm

class Chunk(NamedTuple):
    """
    One embedded chunk. "attention" is (layers, heads, chunk length, chunk length) and
    "attention_special" (layers, heads, chunk length, 2) when attention layers were asked for, both
    None otherwise; every other field is in chunk coordinates too, the chunk covering whole protein
    positions "offset" to "offset" + len(sequence).
    """
    offset : int
    embeddings : np.ndarray
    sequence : str
    id : str
    attention : np.ndarray|None = None
    attention_special : np.ndarray|None = None

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

def parse_layers(layer_string : str):
    """
    Parses --attention_layers, accepting "[0, 16, -1]" as well as "0,16,-1" and a single "-1".
    Negative indices are kept as they are here and resolved against the backbone later.
    """
    layers = literal_eval(layer_string)
    if isinstance(layers, int):
        layers = [layers]

    return [int(layer) for layer in layers]

def resolve_attention_layers(base, layers : list[int]):
    """
    Turns the requested layer indices into non negative ones, so that what ends up in the output
    files does not depend on the backbone. Negative indices count from the end, -1 being the last
    layer, which is the one the saved embeddings come from.
    """
    # ESM2 calls it num_hidden_layers, ESM-C n_layers
    n_layers = next((getattr(base.config, name) for name in ['num_hidden_layers', 'n_layers']
                     if getattr(base.config, name, None)), None)
    if n_layers is None:
        raise ValueError(f'Cannot tell how many layers {type(base).__name__} has, '
                         'pass non negative --attention_layers')

    resolved = [layer + n_layers if layer < 0 else layer for layer in layers]
    out_of_range = [layer for layer in resolved if not 0 <= layer < n_layers]
    if out_of_range:
        raise ValueError(f'Attention layers {out_of_range} are out of range, the model has {n_layers} layers')

    return resolved

def enable_attention_output(base):
    """
    output_attentions=True only returns anything on an attention implementation that materializes
    the weights. Transformers hands back nothing but a warning on the sdpa path of ESM2, and ESM-C
    raises on flash attention; eager works for both.
    """
    if getattr(base.config, '_attn_implementation', None) in [None, 'eager']:
        return

    if hasattr(base, 'set_attn_implementation'):
        base.set_attn_implementation('eager')
    else:
        base.config._attn_implementation = 'eager'

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

def embed_batch(base, tokenizer, sequences : list[str], device, attention_layers=None):
    """
    Last hidden layer embeddings of a batch of sequences, without the [CLS] and [EOS] tokens, and
    the attention weights of "attention_layers", cut to the same positions on both axes. Returns,
    per sequence, a (sequence length, hidden size) array, a
    (layers, heads, sequence length, sequence length) attention array and a
    (layers, heads, sequence length, 2) array of the attention paid to [CLS] and [EOS], the last two
    None when no attention layers are asked for.

    The special tokens are kept apart rather than dropped because a good part of every row goes to
    them; the residue columns and the two special columns of a row together sum to one.
    """
    batch = tokenizer(sequences, padding='longest', return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = base(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                       output_attentions=attention_layers is not None)

    # Token 0 is always the beginning-of-sequence token, so the first residue is token 1
    hidden_states = outputs[0]
    embeddings = [hidden_states[i, 1:len(sequence) + 1].cpu().numpy() for i, sequence in enumerate(sequences)]

    if attention_layers is None:
        return embeddings, [None] * len(sequences), [None] * len(sequences)

    attentions = getattr(outputs, 'attentions', None)
    if not attentions:
        raise RuntimeError('The model returned no attention weights, its attention implementation '
                           'does not support output_attentions=True')

    attention, attention_special = [], []
    for i, sequence in enumerate(sequences):
        residues = slice(1, len(sequence) + 1)
        # [CLS] is token 0 and [EOS] the token right after the last residue
        special = [0, len(sequence) + 1]
        attention.append(np.stack([attentions[layer][i, :, residues, residues].cpu().numpy()
                                   for layer in attention_layers]))
        attention_special.append(np.stack([attentions[layer][i, :, residues][:, :, special].cpu().numpy()
                                           for layer in attention_layers]))

    return embeddings, attention, attention_special

def chunk_owners(chunks : list[Chunk], parent_length : int):
    """
    Which chunk every whole protein position is taken from, as one boolean mask per chunk over that
    chunk's own positions. A position covered by several overlapping chunks belongs to the chunk
    that saw it closest to its center, the same rule stitch_chunk_predictions uses, so the masks
    together cover every covered position exactly once.
    """
    owner = np.full(parent_length, -1)
    # Distance of the position from the center of the chunk it currently comes from
    best_distance = np.full(parent_length, np.inf)

    for i, chunk in enumerate(chunks):
        positions = np.arange(len(chunk.embeddings)) + chunk.offset
        distance = np.abs(np.arange(len(chunk.embeddings)) - (len(chunk.embeddings) - 1) / 2)
        closer = distance < best_distance[positions]

        owner[positions[closer]] = i
        best_distance[positions[closer]] = distance[closer]

    return [owner[np.arange(len(chunk.embeddings)) + chunk.offset] == i for i, chunk in enumerate(chunks)]

def stitch_chunk_embeddings(chunks : list[Chunk], parent_length : int, owners : list[np.ndarray]):
    """
    Lays the embeddings of the chunks of one protein onto whole protein positions, each position
    taken from the chunk chunk_owners assigned it to. Returns (embeddings, sequence).
    """
    hidden_size = chunks[0].embeddings.shape[-1]
    embeddings = np.zeros((parent_length, hidden_size), dtype=chunks[0].embeddings.dtype)
    sequence = np.empty(parent_length, dtype='<U1')

    for chunk, owned in zip(chunks, owners):
        positions = np.arange(len(chunk.embeddings)) + chunk.offset
        embeddings[positions[owned]] = chunk.embeddings[owned]
        sequence[chunk.offset:chunk.offset + len(chunk.sequence)] = list(chunk.sequence)

    return embeddings, ''.join(sequence)

def select_positions(sequence : str, residues : set|None, offset=0):
    """
    Positions of the residues that are kept, in whole protein coordinates. Without a residue set
    every position is kept, so the rest of the code has one case to handle.
    """
    if residues is None:
        return np.arange(len(sequence)) + offset

    return np.array([i + offset for i, res in enumerate(sequence) if res in residues], dtype=int)

def save_protein(protein_id, chunks : list[Chunk], parent_length, out_folder, dtype, residues=None,
                 save_chunks=False, attention_layers=None):
    """
    Writes the embeddings of one protein. Returns the number of positions saved, zero meaning the
    protein has no residue of the selected types and no file is written.
    """
    chunks = sorted(chunks, key=lambda chunk: chunk.offset)
    owners = chunk_owners(chunks, parent_length)
    embeddings, sequence = stitch_chunk_embeddings(chunks, parent_length, owners)
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

    if save_chunks or attention_layers is not None:
        arrays['chunk_offsets'] = np.array([chunk.offset for chunk in chunks])
        arrays['chunk_ids'] = np.array([chunk.id for chunk in chunks])

    if save_chunks:
        # Chunks as they were embedded, keeping the copies stitching drops. Overlapping chunks see
        # a residue with different context around it, so "chunk_<i>" is how chunk i embedded the
        # positions listed in "chunk_positions_<i>", again in whole protein coordinates.
        for i, chunk in enumerate(chunks):
            chunk_positions = select_positions(chunk.sequence, residues, offset=chunk.offset)
            arrays[f'chunk_{i}'] = chunk.embeddings[chunk_positions - chunk.offset].astype(dtype)
            arrays[f'chunk_positions_{i}'] = chunk_positions

    if attention_layers is not None:
        # Attention stays in chunk coordinates: a chunk only ever attends within itself, so there is
        # no whole protein map to stitch the chunks into. "attention_<i>" is
        # (layers, heads, rows, chunk length), its layer axis is "attention_layers", its rows are
        # the selected positions of chunk i listed in "attention_positions_<i>" and its columns are
        # every position of the chunk, column j being whole protein position j + chunk_offsets[i].
        # Rows are restricted to the positions the chunk owns, the same way stitching picks one
        # chunk per position, so no position is written twice. A row sums to one together with
        # "attention_special_<i>", the two columns of attention to [CLS] and [EOS].
        arrays['attention_layers'] = np.array(attention_layers)
        for i, (chunk, owned) in enumerate(zip(chunks, owners)):
            chunk_positions = select_positions(chunk.sequence, residues, offset=chunk.offset)
            rows = chunk_positions[owned[chunk_positions - chunk.offset]]
            arrays[f'attention_{i}'] = chunk.attention[:, :, rows - chunk.offset].astype(dtype)
            arrays[f'attention_positions_{i}'] = rows
            # The rest of every row, what it attended to [CLS] and [EOS], in that order
            arrays[f'attention_special_{i}'] = chunk.attention_special[:, :, rows - chunk.offset].astype(dtype)

    np.savez_compressed(os.path.join(out_folder, f'{protein_id}.npz'), **arrays)

    return len(positions)

def compute_embeddings(args):
    device = torch.device(args.device)
    base, tokenizer = load_base_model(args, device)
    base.eval()

    attention_layers = None
    if args.attention_layers:
        attention_layers = resolve_attention_layers(base, parse_layers(args.attention_layers))
        enable_attention_output(base)
        print(f'Saving the attention weights of layers {attention_layers}. They are quadratic in the '
              'sequence length, lower --batch_size if the run runs out of memory')

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
        embeddings, attention, attention_special = embed_batch(
            base, tokenizer, rows['sequence'].to_list(), device, attention_layers=attention_layers)

        for row, chunk in zip(rows.itertuples(), zip(embeddings, attention, attention_special)):
            chunk_embeddings, chunk_attention, chunk_special = chunk
            pending[row.parent_id].append(Chunk(int(row.offset), chunk_embeddings, row.sequence,
                                                row.id, chunk_attention, chunk_special))

        for protein_id in [p for p in pending if len(pending[p]) == n_chunks[p]]:
            saved = save_protein(protein_id, pending.pop(protein_id), int(parent_lengths[protein_id]),
                                 args.out_folder, args.dtype, residues=residues,
                                 save_chunks=args.save_chunks, attention_layers=attention_layers)
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
    parser.add_argument('--attention_layers', type=str, default=None,
                        help='Layers to also save the per head attention weights of, e.g. "[0,16,-1]" or "0,16,-1". '
                             'Negative indices count from the last layer. Rows are the --residues positions, columns every position of the chunk. '
                             'No attention is saved by default.')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(args)
