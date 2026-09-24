# Computes a UMAP projection of the residue embeddings written by compute_embeddings.py.
# The embeddings of a whole dataset do not fit in memory, so they are never fully loaded as they are.
# Without --pca_dim the projection is fitted on a random subsample of residues and every embedding
# file is then transformed in batches, keeping only the (few floats per residue) coordinates. With
# --pca_dim an incremental PCA is first fitted in batches over the whole folder, and every residue is
# then loaded into memory already reduced to pca_dim dimensions, which UMAP works on from there.

import argparse
import json
import numpy as np
import os
import pandas as pd

from collections import defaultdict
from sklearn.decomposition import IncrementalPCA
from umap import UMAP

def list_embedding_files(folder):
    files = sorted(name for name in os.listdir(folder) if name.endswith('.npz'))
    if not files:
        raise SystemExit(f'No .npz embedding files in {folder}')

    return [os.path.join(folder, name) for name in files]

def split_protein_ids(splits_path, prot_info_path, fold, partition):
    """
    Ids of the proteins in one partition of one fold of a splits file. The splits hold index labels
    into the info file, and a chunked info file is resolved to the parent proteins, which is what
    compute_embeddings.py names its files by.
    """
    with open(splits_path) as f:
        split = json.load(f)[fold]

    if partition not in split:
        raise SystemExit(f'Fold {fold} of {splits_path} has no {partition} partition')

    info = pd.read_json(prot_info_path).loc[split[partition]]
    column = 'parent_id' if 'parent_id' in info else 'id'
    return set(info[column])

def filter_files(files, protein_ids):
    """
    Only the embedding files of the given proteins, matched on the file name compute_embeddings.py
    gives them.
    """
    kept = [path for path in files if os.path.basename(path)[:-len('.npz')] in protein_ids]
    missing = len(protein_ids) - len(kept)
    if missing:
        print(f'{missing} of {len(protein_ids)} proteins of the split have no embedding file')
    if not kept:
        raise SystemExit('No embedding files left after filtering by the split')

    return kept

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

def iter_batches(files, counts, residues, batch_size, label):
    """
    Embeddings of every file in file order, grouped into blocks of at least batch_size residues
    (except possibly the last one), so that only about one batch is held at a time.
    """
    buffer, buffered, done, total = [], 0, 0, int(counts.sum())
    for i, path in enumerate(files):
        if counts[i] == 0:
            continue

        buffer.append(read_embeddings(path, residues))
        buffered += len(buffer[-1])
        if buffered >= batch_size:
            yield np.concatenate(buffer)
            done += buffered
            buffer, buffered = [], 0

        print(f'{label} {done}/{total} residues, {i + 1}/{len(files)} files', end='\r')

    if buffer:
        yield np.concatenate(buffer)
        done += buffered
        print(f'{label} {done}/{total} residues, {len(files)}/{len(files)} files', end='\r')

    print()

def batched_transform(reducer, batches, total, n_components):
    """
    Transforms every block of batches with an already fitted reducer, holding only the results.
    """
    coords = np.zeros((total, n_components), dtype=np.float32)
    done = 0
    for block in batches:
        coords[done:done + len(block)] = reducer.transform(block)
        done += len(block)

    return coords

def fit_pca(files, counts, residues, pca_dim, batch_size):
    """
    An incremental PCA fitted on every residue of the folder, one batch at a time.
    """
    pca = IncrementalPCA(n_components=pca_dim)
    # partial_fit refuses batches with fewer rows than components
    for block in iter_batches(files, counts, residues, max(batch_size, pca_dim), 'Fitted the PCA on'):
        if len(block) < pca_dim:
            # Only the last batch can be this small, its few residues are left out of the fit
            print(f'\nSkipping the last {len(block)} residues for the PCA fit, fewer than --pca_dim')
            continue

        pca.partial_fit(block)

    return pca

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
    if args.splits:
        if not args.prot_info:
            raise SystemExit('--splits needs --prot_info, the info file the splits index into')

        protein_ids = split_protein_ids(args.splits, args.prot_info, args.fold, args.partition)
        files = filter_files(files, protein_ids)
        print(f'Kept {len(files)} proteins of the {args.partition} partition of fold {args.fold}')

    residues = set(args.residues) if args.residues else None
    rng = np.random.default_rng(args.seed)

    protein_ids, protein_index, positions, letters, counts = scan_folder(files, residues)
    total = int(counts.sum())
    if total == 0:
        raise SystemExit(f'No residues left in {args.embedding_folder} after filtering')

    reducer = UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric,
                   n_components=args.n_components, random_state=args.seed if args.seed >= 0 else None,
                   verbose=True)

    embeddings = None
    if args.pca_dim > 0:
        if args.pca_dim > total:
            raise SystemExit(f'--pca_dim {args.pca_dim} is larger than the {total} residues to project')

        print(f'Fitting an incremental PCA down to {args.pca_dim} dimensions on all {total} residues')
        pca = fit_pca(files, counts, residues, args.pca_dim, args.batch_size)
        print(f'The PCA keeps {pca.explained_variance_ratio_.sum():.1%} of the variance')

        print(f'Loading all {total} residues into memory, reduced by the PCA')
        batches = iter_batches(files, counts, residues, args.batch_size, 'Loaded')
        embeddings = batched_transform(pca, batches, total, args.pca_dim)

    fit_size = min(args.fit_size, total) if args.fit_size > 0 else total
    if fit_size < total:
        print(f'Fitting UMAP on {fit_size} of {total} residues, then transforming the rest in batches')
        if embeddings is not None:
            sample = np.sort(rng.choice(total, size=fit_size, replace=False))
            reducer.fit(embeddings[sample])
            batches = (embeddings[start:start + args.batch_size]
                       for start in range(0, total, args.batch_size))
        else:
            reducer.fit(load_fit_sample(files, counts, residues, fit_size, rng))
            batches = iter_batches(files, counts, residues, args.batch_size, 'Projected')

        coords = batched_transform(reducer, batches, total, args.n_components)
    else:
        # Everything fits, one fit_transform is both faster and better than fitting on a sample
        print(f'Fitting UMAP on all {total} residues')
        if embeddings is None:
            embeddings = read_all(files, counts, residues)

        coords = reducer.fit_transform(embeddings).astype(np.float32)

    arrays = {
        'coords' : coords,
        'protein_ids' : protein_ids,
        'protein_index' : protein_index,
        'positions' : positions,
        'residues' : letters,
        'args' : json.dumps(vars(args)),
    }

    if args.pca_dim > 0:
        arrays['pca_explained_variance'] = pca.explained_variance_ratio_.astype(np.float32)

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
                        help='Protein info file, used to mark which residues are known sites and to resolve --splits. Optional otherwise, only for coloring the projection.')
    parser.add_argument('--splits', type=str, default=None,
                        help='Splits file generated from --prot_info. When given, only the proteins of --partition of --fold are projected.')
    parser.add_argument('--fold', type=int, default=0, help='Fold of --splits to take the proteins from.')
    parser.add_argument('--partition', type=str, default='test', choices=['train', 'dev', 'test', 'total'],
                        help='Partition of --fold to take the proteins from.')
    parser.add_argument('--fit_size', type=int, default=50000,
                        help='Number of randomly sampled residues the projection is fitted on. 0 fits on everything, which needs the whole dataset in memory (only its PCA reduction with --pca_dim).')
    parser.add_argument('--batch_size', type=int, default=50000,
                        help='Number of residues the PCA is fitted on and residues are transformed at once.')
    parser.add_argument('--pca_dim', type=int, default=0,
                        help='Reduce the embeddings to this many dimensions with an incremental PCA fitted on every residue, then load all of them into memory reduced. 0 disables the PCA.')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP neighborhood size.')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP minimum distance.')
    parser.add_argument('--n_components', type=int, default=2, help='Dimensions of the projection.')
    parser.add_argument('--metric', type=str, default='cosine', help='UMAP metric.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed. A negative value leaves UMAP unseeded, which lets it use every core and is much faster.')
    args = parser.parse_args()
    main(args)
