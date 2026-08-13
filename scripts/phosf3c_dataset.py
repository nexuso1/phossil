# Converts a phossil dataset (protein info file + splits file) into the format PhosF3C trains on:
# plain FASTA with a '#' right after every phosphorylated residue.
#
#   python phosf3c_dataset.py --prot_info ../data/dbptm/dbptm_info_chunked.json \
#       --splits ../data/dbptm/splits_S_chunked.json --out_folder ../data/phosf3c/dbptm
#
# PhosF3C reads a single train file and a single test file per run and carves its own validation
# set out of the train file by ratio, so every fold becomes its own directory holding train.fasta,
# test.fasta and a pair of ready to run hparams configs.
#
# Chunked info files are reassembled into whole proteins by default: PhosF3C classifies a 2*w+1
# window around each candidate, so keeping the chunks would only feed it the sites in the chunk
# overlaps twice.

import argparse
import json
import os

from pathlib import Path

import pandas as pd

# Residues that can be phosphorylated, and the only ones PhosF3C enumerates as candidates
PHOSPHO_RESIDUES = {'S', 'T', 'Y'}
CATEGORIES = ['train', 'dev', 'test']

LORA_CONFIG = """#model
model_name: lora
device: {device}
checkpoint: ./model_ckp/lora/lora_{name}.pt

#data
train_path: {train_path}
test_path: {test_path}
predict_path: {test_path}
type: {residues}

#train
patience_limit: 5
patience_key: auc
num_epoch: {num_epoch}
batch: {batch}
window_size: {window_size}
split: {split}

#opt
lr: 1e-4
beta1: 0.9
beta2: 0.999
weight_decay: 1e-4

#log & save
val_interval: 500
threshold: 0.5
log_path: ./log/lora
log_name: lora_{name}
save_path: ./model_ckp/lora
save_name: lora_{name}
predict_result_path: ./result/lora
"""

CONFORMER_CONFIG = """#model
model_name: conformer
device: {device}
conformer_checkpoint: ./model_ckp/conformer/conformer_{name}.pt
lora_checkpoint: ./model_ckp/lora/lora_{name}.pt
alpha: 0.5

#data
train_path: {train_path}
test_path: {test_path}
predict_path: {test_path}
type: {residues}

#train
patience_limit: 5
patience_key: auc
num_epoch: {num_epoch}
batch: {batch}
window_size: {window_size}
split: {split}

#opt
lr: 5e-5
beta1: 0.9
beta2: 0.999
weight_decay: 1e-4

#log & save
val_interval: 500
threshold: 0.5
log_path: ./log/conformer
log_name: conformer_{name}
save_path: ./model_ckp/conformer
save_name: conformer_{name}
predict_result_path: ./result/conformer
"""

def parse_residues(splits_path : str):
    """
    Reads the residue set out of a splits filename, e.g. "splits_ST_chunked_max.json" -> {'S', 'T'}.
    Returns None for names that do not follow the convention, --residues is then required.
    """
    parts = Path(splits_path).stem.split('_')
    if len(parts) < 2 or not parts[1] or set(parts[1]) - PHOSPHO_RESIDUES:
        return None

    return set(parts[1])

def is_chunked(prot_info : pd.DataFrame):
    return {'parent_id', 'offset', 'parent_length'}.issubset(prot_info.columns)

def assemble_protein(chunks : pd.DataFrame):
    """
    Rebuilds a protein from its overlapping chunks. Sites are 1-based inside a chunk, shifting them
    by the chunk offset puts them back into the coordinates of the whole protein. A site kept in
    several overlapping chunks maps onto the same position in all of them.
    """
    length = int(chunks['parent_length'].iloc[0])
    buffer = [None] * length
    sites = set()
    for _, chunk in chunks.iterrows():
        offset = int(chunk['offset'])
        for i, residue in enumerate(chunk['sequence']):
            if buffer[offset + i] not in (None, residue):
                raise AssertionError(f'Chunks of {chunk["parent_id"]} disagree at position {offset + i + 1}')

            buffer[offset + i] = residue

        sites.update(int(site) + offset for site in chunk['sites'])

    if any(residue is None for residue in buffer):
        missing = [i + 1 for i, residue in enumerate(buffer) if residue is None]
        raise AssertionError(f'Chunks of {chunks["parent_id"].iloc[0]} do not cover positions {missing[:10]}')

    return ''.join(buffer), sorted(sites)

def build_records(prot_info : pd.DataFrame, unchunk : bool):
    """
    Turns the info file into the records that get written out, keyed by whatever a FASTA entry
    stands for -- a row of the info file, or a whole protein when chunks are reassembled. Also
    returns the mapping from the dataframe index labels the splits file holds onto those keys.
    """
    if not unchunk:
        records = {label : {'id' : row['id'], 'sequence' : row['sequence'],
                            'sites' : [int(site) for site in row['sites']]}
                   for label, row in prot_info.iterrows()}
        return records, {label : label for label in prot_info.index}

    records = {}
    index_to_key = {}
    for parent_id, chunks in prot_info.groupby('parent_id', sort=False):
        sequence, sites = assemble_protein(chunks)
        records[parent_id] = {'id' : parent_id, 'sequence' : sequence, 'sites' : sites}
        for label in chunks.index:
            index_to_key[label] = parent_id

    return records, index_to_key

def resolve_keys(index_labels, index_to_key, splits_name, category):
    """
    Maps the dataframe index labels of one partition onto record keys, dropping the duplicates that
    reassembling chunks introduces while keeping the order of the splits file.
    """
    keys = []
    seen = set()
    missing = 0
    for label in index_labels:
        key = index_to_key.get(label)
        if key is None:
            missing += 1
        elif key not in seen:
            seen.add(key)
            keys.append(key)

    if missing:
        print(f'Warning: {missing} ids of {splits_name} {category} are not in the info file')

    return keys

def check_partitions(fold : dict, categories):
    """
    All chunks of a protein have to sit in the same partition, otherwise the same sequence shows up
    on both sides of the split.
    """
    for i, category in enumerate(categories):
        for other in categories[i + 1:]:
            shared = set(fold[category]) & set(fold[other])
            if shared:
                print(f'Warning: {len(shared)} proteins are in both the {category} and the {other} '
                      f'partition, the splits file mixes chunks of the same protein')

def mark_sites(record : dict, residues):
    """
    Writes the sites into the sequence the way PhosF3C reads them, a '#' right after every
    phosphorylated residue. Sites are 1-based, and only those on a residue of the target set are
    marked -- PhosF3C drops the rest anyway, and every unmarked candidate counts as a negative.
    """
    sequence = record['sequence']
    marked = list(sequence)
    positives = 0
    off_target = 0
    invalid = 0
    for site in record['sites']:
        if not 1 <= site <= len(sequence):
            raise AssertionError(f'Site {site} of {record["id"]} is outside the sequence')

        residue = sequence[site - 1]
        if residue in residues:
            marked[site - 1] += '#'
            positives += 1
        elif residue in PHOSPHO_RESIDUES:
            off_target += 1
        else:
            invalid += 1

    candidates = sum(residue in residues for residue in sequence)
    stats = {'positives' : positives, 'negatives' : candidates - positives,
             'off_target' : off_target, 'invalid' : invalid}

    return ''.join(marked), stats

def wrap_sequence(sequence : str, width : int):
    """
    Wraps a marked sequence at a fixed number of residues, counting the '#' markers as part of the
    residue they belong to so that a marker never starts a line.
    """
    if not width:
        return [sequence]

    lines = []
    line = []
    residues = 0
    for character in sequence:
        if character != '#' and residues == width:
            lines.append(''.join(line))
            line = []
            residues = 0

        line.append(character)
        residues += character != '#'

    if line:
        lines.append(''.join(line))

    return lines

def write_fasta(path : str, entries, width : int):
    with open(path, 'w') as out_file:
        for identifier, sequence in entries:
            # PhosF3C reads the id as everything between '>' and the newline, so no description
            out_file.write(f'>{identifier}\n')
            out_file.write('\n'.join(wrap_sequence(sequence, width)) + '\n')

def merge_duplicates(keys, records, allow_duplicate_ids : bool):
    """
    PhosF3C keys its sequences by the FASTA id, so two entries sharing one would silently overwrite
    each other. Info files do hold the same protein twice (the same sequence under the same id, with
    the sites split over both rows), so those are folded into a single entry holding all the sites.
    """
    merged = {}
    duplicates = 0
    for key in keys:
        record = records[key]
        existing = merged.get(record['id'])
        if existing is None or allow_duplicate_ids:
            merged.setdefault(record['id'], []).append(dict(record))
            continue

        if existing[0]['sequence'] != record['sequence']:
            raise AssertionError(f'Id {record["id"]} stands for two different sequences, '
                                 f'pass --allow_duplicate_ids to write both')

        existing[0]['sites'] = sorted(set(existing[0]['sites']) | set(record['sites']))
        duplicates += 1

    return [record for entries in merged.values() for record in entries], duplicates

def write_partition(path : str, keys, records, residues, width : int, allow_duplicate_ids : bool):
    entries = []
    totals = {'proteins' : 0, 'positives' : 0, 'negatives' : 0, 'off_target' : 0, 'invalid' : 0}
    partition, duplicates = merge_duplicates(keys, records, allow_duplicate_ids)
    if duplicates:
        print(f'Merged {duplicates} duplicate entries of {path} into the protein they repeat')

    for record in partition:
        sequence, stats = mark_sites(record, residues)
        entries.append((record['id'], sequence))
        totals['proteins'] += 1
        for name, value in stats.items():
            totals[name] += value

    write_fasta(path, entries, width)

    return totals

def write_configs(fold_dir : Path, name : str, paths : dict, residues, split : float, args):
    """
    Writes the two hparams files PhosF3C is driven by. Data paths are absolute because PhosF3C runs
    from its own repository root, the log and checkpoint paths are the ones it ships with.
    """
    # The test partition is missing from release splits, fall back to whatever else was written
    test_path = paths.get('test') or paths.get('dev') or paths['train']
    values = {
        'name' : name,
        'device' : args.device,
        'train_path' : paths['train'].resolve(),
        'test_path' : Path(test_path).resolve(),
        'residues' : '[' + ', '.join(f"'{residue}'" for residue in sorted(residues)) + ']',
        'window_size' : args.window_size,
        'num_epoch' : args.num_epoch,
        'batch' : args.batch,
        'split' : f'{split:.4f}',
    }

    for filename, template in [('lora.yaml', LORA_CONFIG), ('conformer.yaml', CONFORMER_CONFIG)]:
        with open(fold_dir / filename, 'w') as out_file:
            out_file.write(template.format(**values))

def convert_split(splits_path : str, records, index_to_key, residues, out_folder : str, args):
    with open(splits_path, 'r') as splits_file:
        folds = json.load(splits_file)

    name = Path(splits_path).stem
    rows = []
    for fold_index, fold in enumerate(folds):
        if args.folds and fold_index not in args.folds:
            continue

        partitions = {category : resolve_keys(fold[category], index_to_key, name, category)
                      for category in CATEGORIES if category in fold}
        check_partitions(partitions, list(partitions))

        # PhosF3C splits its validation set off the train file by ratio, so the dev partition is
        # folded back into it and the ratio is set to the fraction it makes up
        dev = partitions.pop('dev', [])
        split = args.val_split
        if args.dev == 'train' and dev:
            partitions['train'] = partitions['train'] + dev
            split = len(dev) / len(partitions['train'])
        elif args.dev == 'separate' and dev:
            partitions['dev'] = dev

        fold_dir = Path(out_folder) / f'{name}_fold{fold_index}'
        os.makedirs(fold_dir, exist_ok=True)

        paths = {}
        for category, keys in partitions.items():
            path = fold_dir / f'{category}.fasta'
            totals = write_partition(path, keys, records, residues, args.line_width, args.allow_duplicate_ids)
            paths[category] = path
            rows.append({'splits' : name, 'fold' : fold_index, 'partition' : category,
                         'file' : str(path), **totals})
            print(f'{path}: {totals["proteins"]} proteins, {totals["positives"]} positives, '
                  f'{totals["negatives"]} negatives')
            if totals['invalid']:
                print(f'Warning: {totals["invalid"]} sites of {path} are not on an S/T/Y residue')

        if not args.no_configs:
            write_configs(fold_dir, f'{name}_fold{fold_index}', paths, residues, split, args)

    return rows

def main(args):
    prot_info = pd.read_json(args.prot_info)
    unchunk = is_chunked(prot_info) if args.unchunk == 'auto' else args.unchunk == 'yes'
    if unchunk and not is_chunked(prot_info):
        raise ValueError(f'{args.prot_info} has no chunk columns, there is nothing to reassemble')

    records, index_to_key = build_records(prot_info, unchunk)
    if unchunk:
        print(f'Reassembled {len(prot_info)} chunks into {len(records)} proteins')

    os.makedirs(args.out_folder, exist_ok=True)
    rows = []
    for splits_path in args.splits:
        residues = set(args.residues) if args.residues else parse_residues(splits_path)
        if not residues:
            raise ValueError(f'Cannot tell the residue set of {splits_path} from its name, pass --residues')

        print(f'\n{splits_path} -> residues {"".join(sorted(residues))}')
        rows += convert_split(splits_path, records, index_to_key, residues, args.out_folder, args)

    info_path = Path(args.out_folder) / 'dataset_info.csv'
    pd.DataFrame(rows).to_csv(info_path, index=False)
    print(f'\nWrote {info_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a phossil dataset into PhosF3C training data.')
    parser.add_argument('--prot_info', type=str, default='../data/dbptm/dbptm_info_chunked.json',
                        help='Protein info file, a dataframe with columns ("id", "sequence", "sites"). Sites are 1-based.')
    parser.add_argument('--splits', type=str, nargs='+', required=True,
                        help='Splits files to convert, holding index labels into the info file.')
    parser.add_argument('--out_folder', type=str, default='../data/phosf3c',
                        help='Output folder, one subfolder per splits file and fold.')
    parser.add_argument('--residues', type=str, default=None,
                        help='Residues to treat as candidates, e.g. "ST". Read from the splits filename by default.')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Folds to convert. Defaults to all of them.')
    parser.add_argument('--dev', choices=['train', 'separate', 'drop'], default='train',
                        help='What to do with the dev partition: merge it back into the train file and let '
                             'PhosF3C validate on that fraction, write it as its own file, or leave it out.')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation fraction written into the configs when there is no dev partition to size it.')
    parser.add_argument('--unchunk', choices=['auto', 'yes', 'no'], default='auto',
                        help='Reassemble chunked proteins. "auto" does it whenever the info file has chunk columns.')
    parser.add_argument('--line_width', type=int, default=60,
                        help='Residues per FASTA line, 0 writes each sequence on a single line.')
    parser.add_argument('--allow_duplicate_ids', action='store_true',
                        help='Keep entries whose id already appeared. PhosF3C would only see the last of them.')
    parser.add_argument('--no_configs', action='store_true', help='Do not write the PhosF3C hparams files.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device written into the configs.')
    parser.add_argument('--window_size', type=int, default=15,
                        help='Window size written into the configs. PhosF3C classifies 2*window_size+1 residues.')
    parser.add_argument('--num_epoch', type=int, default=10, help='Epoch count written into the configs.')
    parser.add_argument('--batch', type=int, default=64, help='Batch size written into the configs.')
    args = parser.parse_args()
    main(args)
