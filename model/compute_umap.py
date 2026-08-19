# Computes a UMAP projection of the residue embeddings written by compute_embeddings.py.
# The embeddings of a whole dataset do not fit in memory, so nothing is ever fully loaded: the
# projection is fitted on a random subsample of residues and every embedding file is then
# transformed in batches, keeping only the (few floats per residue) coordinates.

import argparse
import json
import numpy as np
import os
import pandas as pd

from collections import defaultdict
from umap import UMAP

def list_embedding_files(folder):
    files = sorted(name for name in os.listdir(folder) if name.endswith('.npz'))
    if not files:
        raise SystemExit(f'No .npz embedding files in {folder}')

    return [os.path.join(folder, name) for name in files]

def selected_rows(data, residues):
    """
    Rows of one embedding file that are kept, together with their whole protein positions and their
    residue letters. The embeddings were possibly already filtered by compute_embeddings.py, this is
    for narrowing them down further.
    """
    sequence = np.array(list(str(data['sequence'])))
    positions = data['positions'] if 'positions' in data.files else np.arange(len(sequence))
    letters = sequence[positions]

    if residues is None:
        return np.arange(len(positions)), positions, letters

    rows = np.flatnonzero(np.isin(letters, list(residues)))
    return rows, positions[rows], letters[rows]

def read_metadata(path, residues):
    """
    Protein id, positions and residue letters of the rows of one file, without touching the
    embeddings themselves. Members of an npz are only decompressed when they are accessed, so this
    stays cheap on a folder of gigabytes.
    """
    with np.load(path) as data:
        _, positions, letters = selected_rows(data, residues)
        return str(data['id']), positions, letters

def read_embeddings(path, residues, rows=None):
    """
    Embeddings of one file as float32, optionally only the given rows of the selected ones.
    """
    with np.load(path) as data:
        selected, _, _ = selected_rows(data, residues)
        if rows is not None:
            selected = selected[rows]

        return data['embeddings'][selected].astype(np.float32)

def scan_folder(files, residues):
    """
    Metadata of every residue that goes into the projection, in file order. Protein ids are kept as
    a unique array plus a per row index into it, which is what the coordinates are joined on later.
    """
    protein_ids, protein_index, positions, letters, counts = [], [], [], [], []
    for i, path in enumerate(files):
        protein_id, file_positions, file_letters = read_metadata(path, residues)
        protein_ids.append(protein_id)
        protein_index.append(np.full(len(file_positions), i, dtype=np.int32))
        positions.append(file_positions.astype(np.int32))
        letters.append(file_letters)
        counts.append(len(file_positions))

        print(f'Scanned {i + 1}/{len(files)} files, {sum(counts)} residues', end='\r')

    print()
    return (np.array(protein_ids), np.concatenate(protein_index), np.concatenate(positions),
            np.concatenate(letters), np.array(counts))

def load_fit_sample(files, counts, residues, fit_size, rng):
    """
    A uniform random sample of fit_size residues drawn from all files, loaded one file at a time.
    """
    starts = np.concatenate([[0], np.cumsum(counts)])
    sample = np.sort(rng.choice(starts[-1], size=fit_size, replace=False))
    file_of_row = np.searchsorted(starts, sample, side='right') - 1

    blocks = []
    for i in np.unique(file_of_row):
        rows = sample[file_of_row == i] - starts[i]
        blocks.append(read_embeddings(files[i], residues, rows=rows))
        print(f'Loaded {sum(len(block) for block in blocks)}/{fit_size} residues for the fit', end='\r')

    print()
    return np.concatenate(blocks)

def batched_transform(reducer, files, counts, residues, batch_size, n_components):
    """
    Projects every residue of every file, holding at most batch_size embeddings at a time.
    """
    coords = np.zeros((counts.sum(), n_components), dtype=np.float32)
    buffer, done = [], 0

    def flush(buffer, done):
        if not buffer:
            return done

        block = np.concatenate(buffer)
        coords[done:done + len(block)] = reducer.transform(block)
        return done + len(block)

    for i, path in enumerate(files):
        if counts[i] == 0:
            continue

        buffer.append(read_embeddings(path, residues))
        if sum(len(block) for block in buffer) >= batch_size:
            done = flush(buffer, done)
            buffer = []

        print(f'Projected {done}/{len(coords)} residues, {i + 1}/{len(files)} files', end='\r')

    flush(buffer, done)
    print()
    return coords

def read_all(files, counts, residues):
    """
    Every embedding of the folder in one array, for when the dataset is small enough to fit.
    """
    blocks = [read_embeddings(path, residues) for i, path in enumerate(files) if counts[i]]
    return np.concatenate(blocks)

def load_site_labels(prot_info_path, protein_ids, protein_index, positions):
    """
    Marks the residues that are known phosphorylation sites, for coloring the projection. Sites are
    1-based in the info files and chunk local in a chunked one, both are undone here.
    """
    info = pd.read_json(prot_info_path)
    sites = defaultdict(set)
    for row in info.itertuples():
        parent = getattr(row, 'parent_id', row.id)
        offset = int(getattr(row, 'offset', 0))
        sites[parent].update(int(site) - 1 + offset for site in row.sites)

    return np.fromiter((position in sites[protein_ids[index]]
                        for index, position in zip(protein_index, positions)),
                       dtype=np.int8, count=len(positions))

def compute_umap(args):
    files = list_embedding_files(args.embedding_folder)
    residues = set(args.residues) if args.residues else None
    rng = np.random.default_rng(args.seed)

    protein_ids, protein_index, positions, letters, counts = scan_folder(files, residues)
    total = int(counts.sum())
    if total == 0:
        raise SystemExit(f'No residues left in {args.embedding_folder} after filtering')

    reducer = UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric,
                   n_components=args.n_components, random_state=args.seed if args.seed >= 0 else None,
                   verbose=True)

    fit_size = min(args.fit_size, total) if args.fit_size > 0 else total
    if fit_size < total:
        print(f'Fitting UMAP on {fit_size} of {total} residues, then transforming the rest in batches')
        reducer.fit(load_fit_sample(files, counts, residues, fit_size, rng))
        coords = batched_transform(reducer, files, counts, residues, args.batch_size, args.n_components)
    else:
        # Everything fits, one fit_transform is both faster and better than fitting on a sample
        print(f'Fitting UMAP on all {total} residues')
        coords = reducer.fit_transform(read_all(files, counts, residues)).astype(np.float32)

    arrays = {
        'coords' : coords,
        'protein_ids' : protein_ids,
        'protein_index' : protein_index,
        'positions' : positions,
        'residues' : letters,
        'args' : json.dumps(vars(args)),
    }

    if args.prot_info:
        arrays['is_site'] = load_site_labels(args.prot_info, protein_ids, protein_index, positions)
        print(f'{arrays["is_site"].sum()} of {total} residues are known sites')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f'Saved a {args.n_components}D projection of {total} residues to {args.out}')

def main(args):
    compute_umap(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute a UMAP projection of residue embeddings.')
    parser.add_argument('--embedding_folder', type=str, required=True,
                        help='Folder of per protein .npz files written by compute_embeddings.py.')
    parser.add_argument('--out', type=str, default='umap.npz',
                        help='File the projection and its residue metadata are written to.')
    parser.add_argument('--residues', type=str, default=None,
                        help='Narrow the projection down to these residues, e.g. "STY". By default every saved residue is projected.')
    parser.add_argument('--prot_info', type=str, default=None,
                        help='Protein info file, used to mark which residues are known sites. Optional, only for coloring the projection.')
    parser.add_argument('--fit_size', type=int, default=50000,
                        help='Number of randomly sampled residues the projection is fitted on. 0 fits on everything, which needs the whole dataset in memory.')
    parser.add_argument('--batch_size', type=int, default=50000,
                        help='Number of residues transformed at once after the fit.')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP neighborhood size.')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP minimum distance.')
    parser.add_argument('--n_components', type=int, default=2, help='Dimensions of the projection.')
    parser.add_argument('--metric', type=str, default='cosine', help='UMAP metric.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed. A negative value leaves UMAP unseeded, which lets it use every core and is much faster.')
    args = parser.parse_args()
    main(args)
