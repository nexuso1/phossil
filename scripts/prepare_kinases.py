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
    scores : list[float] | np.ndarray
    percentiles : list[float] | np.ndarray
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
    return parser

def predict_prot_kinases(sequence : str, sites : list[int]):
    # Vals according to https://github.com/TheKinaseLibrary/kinase-library/blob/master/src/notebooks/substrate.ipynb
    window_size = 15
    padding = window_size // 2
    modded_seq = list(sequence)
    for site in sites:
        modded_seq[site] = modded_seq[site].lower()
    
    modded_seq = ['_' for _ in range(padding)] + modded_seq + ['_' for _ in range(padding)]
    site_preds : list[Prediction] = []
    for site in sites:
        window = modded_seq[site : site + window_size]
        substrate = kl.Substrate("".join(window), pp=True)
        preds = substrate.predict(percentile_round_digits=4, log2_score=False, pp=True, sort_by='name')

        # Avoid 0 division by adding a small epsilon
        preds.loc[preds.Score == 0, 'Score'] = 1e-8
        preds['Score'] = np.log(preds['Score'].to_numpy())

        site_preds.append(Prediction(scores=preds['Score'].to_numpy(), percentiles=preds['Percentile'].to_numpy(), kinase_names=list(preds.index)))

    return site_preds

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
    
def compute_kinase_predictions(sequences : list[str], sites_list : list[int]):
    return [predict_prot_kinases(seq, sites) for seq, sites in zip(sequences, sites_list)]

def compute_kinase_predictions_parallel(sequences : list[str], sites_list : list[int], max_workers : int | None = None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(predict_prot_kinases, sequences, sites_list))

def compute_preds_and_labels(sequences : list[str], sites_list : list[list[int]], mode='threshold', percentile_threshold : float|None = 0.9):
    residues = {'S', 'T', 'Y'}
    # Filter out non STY sites
    for i, sites in enumerate(sites_list):
        temp = []
        for site in sites:
            if sequences[i][site] in residues:
                temp.append(site)
            else:
                print(f"Skipped site {site} ({sequences[i][site]}) in {sites} for {sequences[i]}")
        sites_list[i] = temp
    
    preds = compute_kinase_predictions(sequences, sites_list)
    labels = compute_kinase_labels(preds, percentile_threshold=percentile_threshold, mode=mode)

    records = []
    for pred_list, label_list in zip(preds, labels):
        record = { key : vals for pred in pred_list for key, vals in pred.to_dict().items() }
        record['kinase_labels'] = label_list
        records.append(record)

    return pd.DataFrame.from_records(records)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    prot_info = pd.read_json(args.prot_info)
    sites = prot_info.sites.apply(lambda x: [int(site) - 1 for site in x])
    sequences = prot_info.sequence
    output_df = compute_preds_and_labels(sequences, sites, mode=args.mode, percentile_threshold=args.threshold)
    output_df = prot_info.copy().add_suffix('_temp')
    merged = pd.concat([prot_info, output_df], axis=1)

    if args.out_path == None:
        prot_info_name = os.path.basename(args.prot_info).removesuffix(".json")
        use_threshold = args.mode == 'threshold'
        args.out_path = os.path.join(os.path.dirname(args.prot_info), f"{prot_info_name}_kinase_{args.mode}{"_" + str(args.threshold) if use_threshold else ""}.json")
    merged.to_json(args.out_path, indent=2)