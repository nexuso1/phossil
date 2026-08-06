# Splits proteins that do not fit into the base model context into overlapping chunks.
# The resulting info file can be used in place of the original one. Chunks carry a 'parent_id'
# column, which dataset_creation.py uses to keep every chunk of a protein in the same fold.

import argparse
import pandas as pd

from pathlib import Path

# Residues that can be phosphorylated. Kinase labels are aligned with these sites only.
PHOSPHO_RESIDUES = {'S', 'T', 'Y'}

def chunk_bounds(length : int, max_length : int, overlap : int):
    """
    Returns the 0-based (start, end) bounds of the chunks covering a sequence of the given length.
    The last chunk is aligned to the end of the sequence, so it may overlap more than the others.
    """
    if length <= max_length:
        return [(0, length)]

    stride = max_length - overlap
    starts = list(range(0, length - max_length + 1, stride))
    if starts[-1] + max_length < length:
        starts.append(length - max_length)

    return [(start, start + max_length) for start in starts]

def chunk_kinase_labels(row : pd.Series, start : int, end : int):
    """
    Kinase labels are aligned with the phosphorylatable sites of the protein in order, which is
    how load_prot_data pairs them up. Subset them the same way, so the alignment survives chunking.
    """
    labels = []
    site_index = 0
    for site in row['sites']:
        if row['sequence'][int(site) - 1] not in PHOSPHO_RESIDUES:
            continue

        if start <= int(site) - 1 < end and site_index < len(row['kinase_labels']):
            labels.append(row['kinase_labels'][site_index])

        site_index += 1

    return labels

def chunk_protein(row : pd.Series, max_length : int, overlap : int):
    """
    Splits a single protein into chunks. Sites are 1-based, and a site is kept in every chunk
    containing it, otherwise a chunk would label a real site as a negative.
    """
    bounds = chunk_bounds(len(row['sequence']), max_length, overlap)
    chunks = []
    for i, (start, end) in enumerate(bounds):
        chunk = row.copy()
        # Proteins that fit keep their original id, only split ones are suffixed
        chunk['id'] = row['id'] if len(bounds) == 1 else f"{row['id']}_{i}"
        chunk['sequence'] = row['sequence'][start:end]
        chunk['sites'] = [int(site) - start for site in row['sites'] if start <= int(site) - 1 < end]
        if 'kinase_labels' in row:
            chunk['kinase_labels'] = chunk_kinase_labels(row, start, end)

        chunk['parent_id'] = row['id']
        chunk['chunk_index'] = i
        chunk['offset'] = start
        chunk['parent_length'] = len(row['sequence'])
        chunks.append(chunk)

    return chunks

def chunk_dataset(data : pd.DataFrame, max_length : int, overlap : int):
    chunks = [chunk for _, row in data.iterrows() for chunk in chunk_protein(row, max_length, overlap)]
    return pd.DataFrame(chunks).reset_index(drop=True)

def verify_chunks(original : pd.DataFrame, chunked : pd.DataFrame, max_length : int):
    """
    Checks that chunking preserved the data: every chunk fits the length limit, every site sits on
    the same residue it did in the original protein, and no site was lost.
    """
    sequences = original.set_index('id')['sequence'].to_dict()
    sites = original.set_index('id')['sites'].apply(lambda x: {int(site) for site in x}).to_dict()

    too_long = chunked['sequence'].apply(len) > max_length
    if too_long.any():
        raise AssertionError(f'{int(too_long.sum())} chunks are longer than {max_length}')

    recovered = {}
    for _, chunk in chunked.iterrows():
        parent = chunk['parent_id']
        parent_sequence = sequences[parent]
        if chunk['sequence'] != parent_sequence[chunk['offset']:chunk['offset'] + len(chunk['sequence'])]:
            raise AssertionError(f'Chunk {chunk["id"]} does not match the parent sequence')

        for site in chunk['sites']:
            original_site = int(site) + chunk['offset']
            if chunk['sequence'][int(site) - 1] != parent_sequence[original_site - 1]:
                raise AssertionError(f'Site {site} of chunk {chunk["id"]} sits on a different residue')

            recovered.setdefault(parent, set()).add(original_site)

    for parent, parent_sites in sites.items():
        missing = parent_sites - recovered.get(parent, set())
        if missing:
            raise AssertionError(f'Sites {sorted(missing)} of protein {parent} are not in any chunk')

def default_out_path(info_path : str):
    path = Path(info_path)
    return path.with_name(f'{path.stem}_chunked{path.suffix}')

def print_summary(original : pd.DataFrame, chunked : pd.DataFrame, max_length : int):
    split = chunked['parent_length'] > max_length
    n_split_prots = chunked[split]['parent_id'].nunique()
    original_sites = original['sites'].apply(len).sum()
    chunked_sites = chunked['sites'].apply(len).sum()

    print(f'Proteins: {len(original)} -> {len(chunked)} chunks')
    print(f'Proteins split: {n_split_prots} ({100 * n_split_prots / len(original):.1f}%)')
    print(f'Longest protein: {int(original["sequence"].apply(len).max())} residues')
    print(f'Sites: {original_sites} -> {chunked_sites} '
          f'({chunked_sites - original_sites} duplicated across overlapping chunks)')

def main(args):
    if args.overlap >= args.max_length:
        raise ValueError('Overlap has to be smaller than the maximum chunk length')

    original = pd.read_json(args.info_path)
    chunked = chunk_dataset(original, args.max_length, args.overlap)
    verify_chunks(original, chunked, args.max_length)

    out_path = args.out if args.out else default_out_path(args.info_path)
    chunked.to_json(out_path)

    print_summary(original, chunked, args.max_length)
    print(f'Chunked dataset saved to {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split proteins longer than the base model context into overlapping chunks.')
    parser.add_argument('info_path', type=str,
                        help='Path to the protein info file, a dataframe with columns ("id", "sequence", "sites").')
    parser.add_argument('-o', '--out', type=str, default=None,
                        help='Output path. Defaults to "<input>_chunked.json", next to the input file.')
    parser.add_argument('--max_length', type=int, default=1022,
                        help='Maximum chunk length. ESM2 takes 1024 tokens, 2 of which are [CLS] and [EOS].')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Number of residues shared by consecutive chunks. Every site is then seen with at least half of it as context on both sides.')
    args = parser.parse_args()
    main(args)
