import pandas as pd
import numpy as np
import kinase_library as kl
import argparse
import sys
import os 
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

@dataclass(frozen=True, eq=False, order=False)
class Prediction:
    scores : np.ndarray
    percentiles : np.ndarray
    kinase_names : list[str]

    def to_dict(self):
        return {
            'kinase_scores' : self.scores,
            'kinase_percentiles' : self.percentiles,
            'kinase_names' : self.kinase_names
        }

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot_info', default='../data/dbptm/dbptm_info.json', help='Prot info file path')
    parser.add_argument('--mode', type=str, choices=['percentile', 'threshold', 'sigmoid'], default='threshold')
    parser.add_argument('--threshold', type=float, default=0.9, help='Threhsold for percentile thresholding. Can be None. Used only with the "threshold" mode.')
    parser.add_argument('--out_path', type=str, default=None, help='Output path.')
    parser.add_argument('--max_workers', type=str, default=15, help='Maximum number of worker processes.')
    return parser

def sigmoid(x : np.ndarray):
    return 1 / (1 + np.exp(-x))

def compute_kinase_labels(site_preds : list[Prediction], percentile_threshold : float|None = 0.9, mode : str = 'threshold'):
    if mode == 'threshold' and percentile_threshold:
        return [(pred.percentiles > percentile_threshold).astype(np.uint8) for pred in site_preds]
    if mode == 'percentile':
        return [pred.percentiles for pred in site_preds]
    if mode == 'sigmoid':
        return [sigmoid(pred.scores) for pred in site_preds]
    
    raise ValueError("Invalid mode.")

def compute_kinase_predictions_parallel(sequences : list[str], sites_list : list[int], pred_func, max_workers : int | None = None):
    if max_workers and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(pred_func, sequences, sites_list))
    else:
        return [pred_func(seq, sites) for seq, sites in zip(sequences, sites_list)]

def compute_preds_and_labels(sequences : list[str], sites_list : list[list[int]], mode='threshold', percentile_threshold : float|None = 0.9, max_workers=1):
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
    
    #preds = compute_kinase_predictions(sequences, sites_list)
    preds = compute_kinase_predictions_parallel(sequences, sites_list, compute_single_prot_preds, max_workers=max_workers)
    
    labels = compute_kinase_labels(preds, percentile_threshold=percentile_threshold, mode=mode)

    records = []
    for pred_list, label_list in zip(preds, labels):
        record = { key : vals for pred in pred_list for key, vals in pred.to_dict().items() }
        record['kinase_labels'] = label_list
        records.append(record)

    return pd.DataFrame.from_records(records)

def compute_single_prot_preds(sequence, sites):
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
    # for every kinase (among others not relevant right now)
    preds = kl.score_protein("".join(modded_seq), pp=True, score_round_digits=4,percentile_round_digits=4)

    site_preds = []
    total_preds = pd.concat([preds['ser_thr'], preds['tyrosine']])
    kinase_names = [val.removesuffix('_score') for val in total_preds.columns if val.endswith('_score')]
    score_cols = [val for val in total_preds.columns if val.endswith('_score')]
    percentile_cols = [val for val in total_preds.columns if val.endswith('_percentile')]
    total_preds.loc[:, score_cols] = total_preds.loc[:, score_cols].fillna(-np.inf)
    total_preds.loc[:, percentile_cols] = total_preds.loc[:, percentile_cols].fillna(0)
    total_preds.sort_values('Position', inplace=True)
    scores = total_preds[score_cols].to_numpy()
    percentiles = total_preds[percentile_cols].to_numpy()
    
    for idx in range(scores.shape[0]):
        site_preds.append(Prediction(scores[idx], percentiles[idx], kinase_names=kinase_names))
    return site_preds

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    prot_info = pd.read_json(args.prot_info)
    sites = prot_info.sites.apply(lambda x: [int(site) - 1 for site in x])
    sequences = prot_info.sequence
    output_df = compute_preds_and_labels(sequences, sites, mode=args.mode, percentile_threshold=args.threshold, max_workers=args.max_workers)
    merged = pd.concat([prot_info, output_df], axis=1)

    if args.out_path == None:
        prot_info_name = os.path.basename(args.prot_info).removesuffix(".json")
        use_threshold = args.mode == 'threshold'
        args.out_path = os.path.join(os.path.dirname(args.prot_info), f"{prot_info_name}_kinase_{args.mode}{'_' + str(args.threshold) if use_threshold else ''}.json")
    merged.to_json(args.out_path, indent=2)