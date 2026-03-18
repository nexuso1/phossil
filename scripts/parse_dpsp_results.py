import pandas as pd
import os
import glob
import argparse
import ast
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score, precision_score, recall_score, roc_auc_score

def extract_prediction_records(text_data):
    """
    Parses site predictions from a FASTA-like text format.
    """
    parsed_data = []
    current_id = None
    
    for line in text_data.strip().split('\n'):
        line = line.strip()
        
        if not line:
            continue
            
        if line.startswith('>'):
            current_id = line[1:].strip()
        else:
            parts = line.split()
            
            if len(parts) == 3 and current_id:
                seq_index = int(parts[0])
                residue = parts[1]
                probability = float(parts[2])
                
                parsed_data.append({
                    'ID': current_id,
                    'Sequence_Index': seq_index,
                    'Residue': residue,
                    'Probability': probability
                })
                
    return parsed_data

def load_residue_prediction(folder, residue):
    """
    Loads predictions for each fold.
    """
    dfs = []
    # Assumes files are named something like splits_Y_fold1.txt
    for file in glob.glob(os.path.join(folder, f'splits_{residue}_*')):
        with open(file, 'r') as f:
            # groupby makes 'ID' the index of the resulting DataFrame
            df = pd.DataFrame.from_records(extract_prediction_records(f.read()))
            if not df.empty:
                dfs.append(df.groupby('ID').agg(list))

    return dfs

def compute_metrics(pred_df, prot_info, threshold=0.5):
    """
    Flattens predictions and true labels for a single fold to calculate metrics.
    """
    # pred_df already has 'ID' as index from the groupby operation.
    # We join with prot_info, setting its index to 'id'.
    joined = pred_df.join(prot_info.set_index('id'), how='inner')
    
    y_true_all = []
    y_prob_all = []
    
    # Flatten the lists of sequences into global true and predicted arrays
    for _, row in joined.iterrows():
        seq_indices = row['Sequence_Index']
        probs = row['Probability']
        
        # Ensure true_sites is a set for O(1) lookups
        true_sites = set(row['sites']) if isinstance(row['sites'], (list, set)) else set()
        
        for seq_idx, prob in zip(seq_indices, probs):
            y_true_all.append(1 if seq_idx in true_sites else 0)
            y_prob_all.append(prob)
            
    # Fallback if no valid records were joined
    if not y_true_all:
        return None
        
    y_pred_all = [1 if p >= threshold else 0 for p in y_prob_all]
    
    metrics = {
        'F1_Score': f1_score(y_true_all, y_pred_all),
        'MCC': matthews_corrcoef(y_true_all, y_pred_all),
        'AUPRC': average_precision_score(y_true_all, y_prob_all) if sum(y_true_all) > 0 else 0.0,
        'Precision': precision_score(y_true_all, y_pred_all),
        'Recall': recall_score(y_true_all, y_pred_all),
        'AUROC' : roc_auc_score(y_true_all, y_prob_all)
    }
    
    return metrics

def parse_dpsp_predictions(folder, prot_info_path):
    """
    Parses predictions for each residue and fold, computes the metrics and 
    averages the results across folds. The metrics are saved as a file inside the 
    original folder.
    """
    # Load protein information mapping
    prot_info = pd.read_json(prot_info_path)
    
    # Safely evaluate strings that look like lists (e.g., "[11, 41, 66]") back into Python lists
    if prot_info['sites'].dtype == object:
        prot_info['sites'] = prot_info['sites'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Automatically detect all residues present in the folder by looking at filenames
    all_files = glob.glob(os.path.join(folder, 'splits_*_*'))
    residues = set([os.path.basename(f).split('_')[1] for f in all_files])
    
    results = []
    
    for res in residues:
        fold_dfs = load_residue_prediction(folder, res)
        
        fold_metrics = []
        for i, df in enumerate(fold_dfs):
            metrics = compute_metrics(df, prot_info)
            if metrics:
                metrics['Fold'] = i + 1
                fold_metrics.append(metrics)
                
        if fold_metrics:
            # Create a DataFrame of all folds for this residue to easily calculate the mean
            res_df = pd.DataFrame(fold_metrics)
            avg_metrics = res_df.drop(columns=['Fold']).mean().to_dict()
            
            # Put 'Residue' at the front of the dictionary
            final_row = {'Residue': res}
            final_row.update(avg_metrics)
            results.append(final_row)
            
    # Save output
    if results:
        final_df = pd.DataFrame(results)
        output_file = os.path.join(folder, 'averaged_metrics.csv')
        final_df.to_csv(output_file, index=False)
        print(f"Success! Metrics computed for {len(residues)} residues and saved to {output_file}")
    else:
        print("No predictions matched the true labels, no metrics computed.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate DPSP Site Predictions")
    parser.add_argument('--folder', type=str, help="Path to the folder containing 'splits_*' files.", default='../data/dpsp_results')
    parser.add_argument('--prot_info', type=str, help="Path to the CSV file containing protein info and true sites.", default='../data/dbptm/dbptm_info.json')
    
    args = parser.parse_args()
    parse_dpsp_predictions(args.folder, args.prot_info)

if __name__ == '__main__':
    main()