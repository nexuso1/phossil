import pandas as pd
import numpy as np
import kinase_library as kl
import argparse
import sys
import os 
import re
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from functools import partial

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot_info', default='../data/dbptm/dbptm_info.json', help='Prot info file path')
    parser.add_argument('--mode', type=str, choices=['percentile', 'threshold', 'sigmoid'], default='threshold')
    parser.add_argument('--threshold', type=float, default=90, help='Threhsold for percentile thresholding. Can be None. Used only with the "threshold" mode.')
    parser.add_argument('--out_path', type=str, default=None, help='Output path.')
    parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of worker processes.')
    parser.add_argument('--preds_only', action='store_true')
    parser.add_argument('--labels_only', action='store_true')
    return parser

def sigmoid(x : np.ndarray):
    return 1 / (1 + np.exp(-x))

def compute_kinase_labels(scores, percentiles, percentile_threshold : float|None = 90, mode : str = 'threshold'):
    if mode == 'threshold' and percentile_threshold:
        return [(np.asarray(pred) > percentile_threshold).astype(np.uint8) for pred in percentiles]
    if mode == 'percentile':
        return [pred for pred in percentiles]
    if mode == 'sigmoid':
        return [sigmoid(pred) for pred in scores]
    
    raise ValueError("Invalid mode.")

def compute_kinase_predictions_parallel(sequences : list[str], sites_list : list[int], kinase_to_index, pred_func, max_workers : int | None = None,):
    residues = {'S', 'T', 'Y'}
    # Filter out non STY sites
    for i, sites in enumerate(sites_list):
        temp = []
        for site in sites:
            if sequences[i][site] in residues:
                temp.append(site)
            else:
                print(f"Skipped site {site} ({sequences[i][site]}) in {sites} for {sequences[i]}")
        temp.sort()
        sites_list[i] = temp

    if max_workers and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(partial(pred_func, kinase_to_index=kinase_to_index), sequences, sites_list,))
    else:
        return [pred_func(seq, sites, kinase_to_index) for seq, sites in zip(sequences, sites_list)]

def compute_preds_and_labels(sequences : list[str], sites_list : list[list[int]], mode='threshold', percentile_threshold : float|None = 0.9, max_workers=1):
    preds = compute_kinase_predictions_parallel(sequences, sites_list, compute_single_prot_preds, max_workers=max_workers)
    
    labels = compute_kinase_labels(preds, percentile_threshold=percentile_threshold, mode=mode)

    records = []
    for pred_list, label_list in zip(preds, labels):
        record = { key : vals for pred in pred_list for key, vals in pred.to_dict().items() }
        record['kinase_labels'] = label_list
        records.append(record)

    return pd.DataFrame.from_records(records)

def parse_phosphosites(sequence, phosphoacceptor=['s', 't', 'y'], pp=False):
    """

    From kinase-library source, modified. Phosphoacceptors are now case-sensitive.

    Parse protein sequence to identify phosphorylation sites.

    Parameters
    ----------
    sequence : str
        Protein sequence using one-letter amino acid codes.
    phosphoacceptor : List[str], optional
        Phosphoacceptors to parse (any combination of 'S', 'T', 'Y'). Default is ['S', 'T', 'Y'].
    pp : bool, optional
        Phospho-priming. If False, all non-central residues uppercase.
        If True, keep S/T/Y case, others uppercase. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Residue, Position, Sequence.
        Sequence is 15-mer centered on phosphosite, padded with '_'.
    """
    # Create regex pattern for phosphoacceptors
    pattern = '[' + ''.join([aa for aa in phosphoacceptor]) + ']'

    # Find all matches with their positions
    matches = [(m.group(), m.start()) for m in re.finditer(pattern, sequence)]

    sites = []

    for residue, i in matches:
        # Create 15-mer window
        start_idx = max(0, i - 7)
        end_idx = min(len(sequence), i + 8)
        window_seq = sequence[start_idx:end_idx]

        # Pad with '_'
        upstream_pad = max(0, 7-i)
        downstream_pad = max(0, (i+8) - len(sequence))
        raw_window = '_'*upstream_pad + window_seq + '_'*downstream_pad

        # Apply phospho-priming rules
        if pp:
            # Keep S/T/Y case, others uppercase, central always lowercase
            window = ''.join([
                char.upper() if char.upper() not in 'STY' else char
                for char in raw_window[:7]
            ]) + raw_window[7].lower() + ''.join([
                char.upper() if char.upper() not in 'STY' else char
                for char in raw_window[8:]
            ])
        else:
            # All uppercase except central (lowercase)
            window = raw_window[:7].upper() + raw_window[7].lower() + raw_window[8:].upper()

        sites.append({
            'Residue': residue.lower(),
            'Position': i + 1,
            'Sequence': window
        })

    return pd.DataFrame(sites)

def compute_single_prot_preds(sequence, sites, kinase_to_index):
    """
    Computes kinase predictions using the "score_protein" function from kinase-library.

    Args:
        sequence (str): Protein sequence
        sites (list[int]): List of phosphorylated site indices

    Returns:
        list[Prediction]: _description_
    """
    modded_seq = list(sequence)
    for site in sites:
        modded_seq[site] = modded_seq[site].lower()
        
    # Dict with 'ser_thr' and 'tyrosine' keys
    # Contains columns <kinase_name>_score, <kinase_name>_score_rank, <kinase_name>_percentile, <kinase_name>_percentile rank
    # for every kinase (among other cols not relevant right now)
    substrates = parse_phosphosites("".join(modded_seq), pp=True)
    print(len(sequence))
    pps = kl.phosphoproteomics.PhosphoProteomics(data=substrates, seq_col='Sequence', pp=True)

    preds = {
        'ser_thr': pps.predict(kin_type='ser_thr',
                                      score_round_digits=4,
                                      percentile_round_digits=4),
        'tyrosine': pps.predict(kin_type='tyrosine',
                                score_round_digits=4,
                                percentile_round_digits=4)
    }

    total_preds = pd.concat([preds['ser_thr'], preds['tyrosine']])

    neg_inf = -10e8

    # Compute the ordering according to kinase_to_index
    kinase_names = [val.removesuffix('_score') for val in total_preds.columns if val.endswith('_score')]
    ordering = [kinase_to_index[name] for name in kinase_names]

    # Extract scores and percentiles
    score_cols = [val for val in total_preds.columns if val.endswith('_score')]
    percentile_cols = [val for val in total_preds.columns if val.endswith('_percentile')]

    # Fill the missing values (tyrosine kinases will not have any predictions for S/T sites and vice versa)
    total_preds.loc[:, score_cols] = total_preds.loc[:, score_cols].fillna(neg_inf)
    total_preds.loc[:, percentile_cols] = total_preds.loc[:, percentile_cols].fillna(0)

    # Ascending order sort of site indices
    total_preds.sort_values('Position', inplace=True)

    # Make sure the scores and percentiles are in the same order as kinase_to_idx
    scores, percentiles = np.zeros(shape=(total_preds.shape[0], len(list(kinase_to_index.keys())))) + neg_inf,  np.zeros(shape=(total_preds.shape[0], len(kinase_to_index)))
    scores[:, ordering] = total_preds[score_cols].to_numpy()
    percentiles[:, ordering] = total_preds[percentile_cols].to_numpy()

    return {
        'scores' : scores,
        'percentiles' : percentiles
    }

def get_kinase_to_idx():
    st = kl.get_kinase_list('ser_thr')
    y = kl.get_kinase_list('tyrosine')
    res = list(set(st + y))
    res.sort()
    res = { kinase : i for i, kinase in enumerate(res) }
    return res

def main(args):
    prot_info = pd.read_json(args.prot_info)
    prot_info.loc[:, 'sites'] = prot_info.sites.apply(lambda x: sorted(x, key=int))
    sites = prot_info.sites.apply(lambda x: sorted([int(site) - 1 for site in x]))
    sequences = prot_info.sequence
    preds_path = f"{args.prot_info.removesuffix('.json')}_kinase_preds.json"
    kinase_to_index = get_kinase_to_idx()

    if not args.labels_only:
        preds = compute_kinase_predictions_parallel(sequences, sites_list=sites, pred_func=compute_single_prot_preds, kinase_to_index=kinase_to_index, max_workers=args.max_workers)
        total_scores = []
        total_percentiles = []

        for pred in preds:
            total_scores.append(pred['scores'])
            total_percentiles.append(pred['percentiles'])

        prot_info = prot_info.assign(kinase_scores = total_scores, kinase_percentiles=total_percentiles)
        prot_info.to_json(preds_path, indent=2)

        with open(os.path.join(os.path.dirname(args.prot_info), 'kinase_to_idx.json'), 'w') as f:
            json.dump(kinase_to_index, f, indent=2)
        
        if args.preds_only:
            return
    
    preds = pd.read_json(preds_path)
    if args.out_path == None:
        prot_info_name = os.path.basename(args.prot_info).removesuffix(".json")
        use_threshold = args.mode == 'threshold'
        args.out_path = os.path.join(os.path.dirname(args.prot_info), f"{prot_info_name}_kinase_{args.mode}{'_' + str(args.threshold) if use_threshold else ''}.json")
    labels = compute_kinase_labels(preds['kinase_scores'], preds['kinase_percentiles'], args.threshold, args.mode)
    preds = preds.assign(kinase_labels=labels)
    preds.to_json(args.out_path, indent=2)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)