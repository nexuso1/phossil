from dataset_creation import prune_long_seqs, compute_length_labels, write_dataset_info, save_dataset
from sklearn.model_selection import StratifiedKFold
import argparse
import pandas as pd

def create_splits_full(info_with_length_labels, set_labels, seed=42):
    """
    Uses a single test set according to the original data, and creates dev set folds
    """
    total_splits = {}
    orig_train = info_with_length_labels[info_with_length_labels['original_label'] == 'train']
    test = info_with_length_labels[info_with_length_labels['original_label'] == 'test']
    for label in set_labels:
        cv = StratifiedKFold(random_state=seed, shuffle=True)
        splits = []
        for train, dev in cv.split(orig_train.index, orig_train['length_class']):
            train_indices = info_with_length_labels.index[train].tolist()
            dev_indices = info_with_length_labels.index[dev].tolist()
            splits.append({'train' : train_indices, 'dev' : dev_indices, 'test' : test.index.tolist(), 'total' : list(info_with_length_labels.index)})

        total_splits[label] = splits
    return total_splits

def create_full_dataset(prot_info_path, res_sets, out_folder):
    prot_info = pd.read_json(prot_info_path)
    pruned = prune_long_seqs(prot_info)
    pruned = compute_length_labels(pruned)
    set_labels = ["".join(sorted(list(res_set))) for res_set in res_sets]
    splits = create_splits_full(pruned, set_labels=set_labels)

    write_dataset_info(splits, out_folder, suffix=f'_{set_labels[0]}')
    save_dataset(splits, out_folder)

def main(args):
    res_sets = eval(args.res_sets)
    if args.type == 'full':
        create_full_dataset(args.prot_info, res_sets, args.out_folder)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot_info', type=str, default='../data/deeppsp/dpsp_info_ST.json')
    parser.add_argument('--clusters', type=str, default='../data/clusters_cov1_c05.tsv')
    parser.add_argument('--res_sets', type=str, default="[{'S', 'T'}]")
    parser.add_argument('--type', type=str, default='full', choices=['full', 'clustered'])
    parser.add_argument('--out_folder', type=str, default='../data/deeppsp')

    args = parser.parse_args()
    main(args)