from dataset_creation import add_chunk_columns, compute_length_labels, compute_site_labels, \
filter_dataset, get_parent_indices, get_parent_length_classes, write_dataset_info, save_dataset
from sklearn.model_selection import StratifiedKFold
import argparse
import pandas as pd

def create_splits_full(prot_info, set_labels, seed=42):
    """
    Uses a single test set according to the original data, and creates dev set folds.

    Every residue set gets its own subset, holding only the proteins that carry at least one site
    of that set. A protein with no site of the residue set would contribute negatives only.
    """
    total_splits = {}
    for label in set_labels:
        filtered = filter_dataset(prot_info, label)
        if filtered.empty:
            raise ValueError(f'No protein in the dataset has a site of the residue set "{label}"')

        orig_train = filtered[filtered['original_label'] == 'train']
        test = filtered[filtered['original_label'] == 'test']

        # Split over proteins, chunks of one protein are near duplicates and have to stay together
        parent_classes = get_parent_length_classes(orig_train)

        cv = StratifiedKFold(random_state=seed, shuffle=True)
        splits = []
        for train, dev in cv.split(parent_classes.index, parent_classes):
            # Positions returned by the split index the proteins of the train partition
            train_indices = get_parent_indices(orig_train, parent_classes.index[train])
            dev_indices = get_parent_indices(orig_train, parent_classes.index[dev])
            splits.append({'train' : train_indices, 'dev' : dev_indices,
                           'test' : test.index.tolist(), 'total' : list(filtered.index)})

        print(f'{label}: {filtered["parent_id"].nunique()} of {prot_info["parent_id"].nunique()} proteins '
              f'have a site of this residue set, {len(filtered)} chunks '
              f'({len(parent_classes)} train proteins, {test["parent_id"].nunique()} test proteins)')
        total_splits[label] = splits

    return total_splits

def create_full_dataset(prot_info_path, res_sets, out_folder):
    prot_info = pd.read_json(prot_info_path)
    #pruned = prune_long_seqs(prot_info)
    prot_info = add_chunk_columns(prot_info)
    pruned = compute_length_labels(prot_info)
    set_labels = ["".join(sorted(list(res_set))) for res_set in res_sets]

    # Sites of each residue set separately, so that the subsets can be filtered by them
    pruned = compute_site_labels(pruned, res_sets, set_labels)
    splits = create_splits_full(pruned, set_labels=set_labels)

    write_dataset_info(splits, out_folder, suffix=f'_{set_labels[0]}{args.suffix}')
    save_dataset(splits, out_folder, args.suffix)

def main(args):
    res_sets = eval(args.res_sets)
    create_full_dataset(args.prot_info, res_sets, args.out_folder)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot_info', type=str, default='../data/deeppsp/dpsp_info_ST.json')
    parser.add_argument('--res_sets', type=str, default="[{'S', 'T'}]")
    parser.add_argument('--out_folder', type=str, default='../data/deeppsp')
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()
    main(args)
