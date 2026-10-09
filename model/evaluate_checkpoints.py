# Re-evaluates the fold checkpoints of a finished run on the test set of a different splits file, and
# saves the predictions next to the run's own, in the same format.
#
# The motivating case is comparing pooled S and T models against a jointly evaluated S/T test set. A
# residue-specific dataset only holds the proteins with a site of that residue
# (dataset_creation.filter_dataset), so the test set of an S run lacks the proteins with T sites only,
# and their S residues are never scored. Evaluating the S run on the test set of the ST splits scores
# them too. The run's own --residues still decide which positions are candidates, so the S model only
# ever scores S residues.
#
# The residue filter of dataset creation works on rows, which are chunks in a chunked dataset, so a
# test list also lacks the chunks of a listed protein that hold no site of the residue set. By default
# the test list is therefore expanded to every chunk of the proteins it names, so each protein is
# scored whole (--listed-chunks-only evaluates the list as it is).
#
# The splits file has to index the run's info file. On deeppsp, dpsp_info_{S,T}_chunked.json are row
# for row the same as dpsp_info_ST_chunked.json, so splits_ST_chunked.json works for both S and T runs,
# and expanded to whole proteins it covers every S/T residue of the DeepPSP test set.
#
# Output, in every evaluated fold directory:
#   <name>_preds_fold_N.json, <name>_preds_fold_N_stitched.json   same format as test_preds_*
#   <name>_metrics_fold_N.json                                    test_* / stitched_test_* metrics
# metadata.json is deliberately left alone, so this can run while the same run is still training
# other folds without the two racing for that file.
#
# As a check that the model was restored correctly, the new logits are compared with the run's saved
# test predictions on every position both test sets contain. They should agree up to float noise.
#
# Usage, from model/ on the machine that holds the checkpoints:
#   python evaluate_checkpoints.py new_logs/finetuning_drop_S_2_650M_deeppsp_chunked \
#       --entry ft_selective --splits ../data/deeppsp/splits_ST_chunked.json
# Then copy fold_*/test_ST_chunked_whole_* into model/logs/<run>/fold_*/ and pool the S and T runs with
#   python combine_residue_models.py <S run> <T run> --test-split test_ST_chunked_whole

import argparse
import glob
import importlib
import json
import os
import warnings
from argparse import Namespace
from functools import partial

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loading import load_prot_data, parse_residues, prep_batch
from prot_dataset import ProteinDataset
from training import LightningWrapper, create_metrics, parser as training_parser, prepare_model, save_predictions

# Logit differences above this against the run's saved predictions mean the model was not restored
# as it was trained (wrong --entry, wrong backbone, ...), not float noise
MAX_LOGIT_DIFF = 1e-3

# LightningWrapper logs every metric with logger=True, which warns once per metric without a logger
warnings.filterwarnings('ignore', message='.*but have no logger configured.*')


def load_run_args(run_dir, module, overrides):
    """Training args of the run, with the parser defaults filling in args that older metadata lacks."""
    with open(os.path.join(run_dir, 'metadata.json')) as f:
        data = json.load(f)['data']

    if hasattr(module, 'add_arguments'):
        module.add_arguments(training_parser)
    args = Namespace(**{**vars(training_parser.parse_args([])), **data['args']})

    # Inference only: neither changes the outputs, and compiling would cost more than it saves
    args.compile = False
    args.grad_checkpointing = False
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)
    return args, data


def folds_to_evaluate(run_dir, data, requested):
    """Folds with a best checkpoint, restricted to the finished ones unless folds are requested."""
    available = sorted(int(os.path.basename(os.path.dirname(p)).split('_')[1])
                       for p in glob.glob(os.path.join(run_dir, 'fold_*', 'best.ckpt')))
    if requested is not None:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise FileNotFoundError(f'no best.ckpt for fold(s) {missing} in {run_dir}')
        return sorted(requested)

    # An unfinished fold's best.ckpt is only the best epoch so far
    finished = data.get('fold_finished')
    if finished:
        available = [f for f in available if f < len(finished) and finished[f]]
    return available


def test_indices(prot_info, split, whole_proteins):
    """Rows of the split's test set, expanded to every chunk of the proteins it lists."""
    indices = [i for i in split['test'] if i in prot_info.index]
    if not whole_proteins or 'parent_id' not in prot_info:
        return indices
    parents = set(prot_info.loc[indices, 'parent_id'])
    return prot_info.index[prot_info['parent_id'].isin(parents)].tolist()


def keyed_logits(pred_df, subset):
    """{(chunk id, position): logit} of chunk level predictions made on `subset`."""
    out = {}
    for row in pred_df.itertuples():
        chunk_id = subset.iloc[row.df_index]['id']
        for position, logit in zip(row.sequence_indices, row.logits):
            out[(chunk_id, int(position))] = float(logit)
    return out


def compare_with_saved(fold_dir, fold, new_preds, new_subset, prot_info, run_splits):
    """(shared positions, saved positions, max |logit difference|) against the run's own test
    predictions, or None if they are not available here."""
    saved_path = os.path.join(fold_dir, f'test_preds_fold_{fold}.json')
    if run_splits is None or not os.path.exists(saved_path):
        return None

    saved = keyed_logits(pd.read_json(saved_path), prot_info.loc[run_splits[fold]['test']])
    new = keyed_logits(new_preds, new_subset)
    shared = saved.keys() & new.keys()
    diff = max((abs(saved[k] - new[k]) for k in shared), default=float('nan'))
    return len(shared), len(saved), diff


def evaluate_fold(args, module, prot_info, splits, run_splits, fold, run_dir, name, num_workers,
                  whole_proteins=True):
    fold_dir = os.path.join(run_dir, f'fold_{fold}')
    ckpt_path = os.path.join(fold_dir, 'best.ckpt')
    if os.path.isdir(ckpt_path):
        raise ValueError(f'{ckpt_path} is a sharded (deepspeed) checkpoint, consolidate it first')

    test = prot_info.loc[test_indices(prot_info, splits[fold], whole_proteins)]
    test_ds = ProteinDataset(test, kinase_labels=args.kinase)

    model, tokenizer = prepare_model(args, module.create_model)
    step_metrics, epoch_metrics = create_metrics(args.ignore_label)
    wrapper = LightningWrapper(args, model, epoch_metrics=epoch_metrics, step_metrics=step_metrics,
                               ds_size=len(test_ds), logdir=fold_dir, train_epochs=args.epochs, lr=args.lr)
    # Our own checkpoints; they hold the args Namespace and RNG state, which weights_only rejects
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    wrapper.load_state_dict(checkpoint['state_dict'])

    loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label,
                                           modify_prob=0, kinases=args.kinase))
    # A single device: the multi-GPU deepspeed strategy of training is pointless for inference
    trainer = L.Trainer(logger=False, enable_checkpointing=False, devices=1, deterministic=True)
    metrics = trainer.test(wrapper, loader, verbose=False)[0]

    columns = ['logits', 'labels', 'sequence_indices', 'df_index']
    if args.kinase:
        columns += ['kinase_logits']
    pred_name = f'{name}_preds_fold_{fold}'
    metrics.update(save_predictions(wrapper.test_preds, columns, test_ds.data, fold_dir, pred_name,
                                    args.ignore_label, prefix='stitched_test_'))
    metrics = {k: float(v) for k, v in metrics.items()}

    check = compare_with_saved(fold_dir, fold, pd.read_json(os.path.join(fold_dir, f'{pred_name}.json')),
                               test, prot_info, run_splits)
    report = {'splits': args.eval_splits, 'whole_proteins': whole_proteins, 'checkpoint': ckpt_path,
              'chunks': len(test), 'metrics': metrics}
    if check is not None:
        report['saved_positions_compared'], report['saved_positions'], report['max_logit_diff_vs_saved'] = check
    with open(os.path.join(fold_dir, f'{name}_metrics_fold_{fold}.json'), 'w') as f:
        json.dump(report, f, sort_keys=True, indent=4)

    new_parents = set(test['parent_id' if 'parent_id' in test else 'id'])
    if run_splits is not None:
        own = prot_info.loc[run_splits[fold]['test']]
        new_parents -= set(own['parent_id' if 'parent_id' in own else 'id'])
    return metrics, check, len(new_parents)


def main():
    parser = argparse.ArgumentParser(description='Evaluates the fold checkpoints of a run on the test set of another splits file.')
    parser.add_argument('run_dir', help='Run directory with metadata.json and fold_N/best.ckpt.')
    parser.add_argument('--entry', required=True,
                        help='Module that trained the run, whose create_model rebuilds it, e.g. ft_selective or lora_model.')
    parser.add_argument('--splits', required=True, help='Splits file whose test set is evaluated. Must index the run\'s info file.')
    parser.add_argument('--listed-chunks-only', action='store_true',
                        help='Evaluate the test list as it is, instead of every chunk of the proteins it lists.')
    parser.add_argument('--name', default=None,
                        help='Prefix of the output files (default: test_<splits file name without "splits_">, '
                             'plus "_whole" unless --listed-chunks-only).')
    parser.add_argument('--folds', default=None, help='Comma-separated folds (default: every finished fold).')
    parser.add_argument('--prot_info_path', default=None,
                        help='Override the run\'s info file path, e.g. when the run was trained elsewhere.')
    parser.add_argument('--batch_size', type=int, default=None, help='Override the run\'s batch size.')
    parser.add_argument('--num_workers', type=int, default=0)
    cli = parser.parse_args()

    run_dir = os.path.normpath(cli.run_dir)
    module = importlib.import_module(cli.entry)
    args, data = load_run_args(run_dir, module, {'prot_info_path': cli.prot_info_path,
                                                 'batch_size': cli.batch_size})
    args.eval_splits = cli.splits
    whole_proteins = not cli.listed_chunks_only
    name = cli.name or ('test_' + os.path.splitext(os.path.basename(cli.splits))[0].removeprefix('splits_')
                        + ('_whole' if whole_proteins else ''))
    if name == 'test':
        parser.error('"test" would overwrite the run\'s own test predictions, pick another --name')

    requested = [int(f) for f in cli.folds.split(',')] if cli.folds else None
    folds = folds_to_evaluate(run_dir, data, requested)
    if not folds:
        print(f'No finished fold with a best.ckpt in {run_dir}.')
        return

    L.seed_everything(args.seed)
    if getattr(args, 'hpc', False):
        torch.set_float32_matmul_precision('high')

    prot_info = load_prot_data(args.prot_info_path, residues=parse_residues(args.residues),
                               ignore_index=args.ignore_label, kinases=args.kinase)
    with open(cli.splits) as f:
        splits = json.load(f)
    # The run's own splits, to compare against its saved predictions. Missing when the run was
    # trained with paths that do not exist here.
    run_splits = None
    if os.path.exists(args.dataset_path):
        with open(args.dataset_path) as f:
            run_splits = json.load(f)
    else:
        print(f'{args.dataset_path} not found, skipping the comparison with the saved predictions')

    failed = False
    for fold in folds:
        metrics, check, n_new = evaluate_fold(args, module, prot_info, splits, run_splits, fold, run_dir,
                                              name, cli.num_workers, whole_proteins)
        line = (f'fold {fold}: stitched_test_mcc={metrics["stitched_test_mcc"]:.4f}, '
                f'{n_new} protein(s) not in the run\'s own test set')
        if check is not None:
            shared, saved, diff = check
            if shared == 0:
                line += '; no position shared with the saved predictions to compare'
            else:
                line += f'; vs saved predictions: {shared}/{saved} positions, max |logit diff| {diff:.2e}'
                if diff > MAX_LOGIT_DIFF:
                    line += '  <-- MISMATCH'
                    failed = True
        print(line)

    if failed:
        raise SystemExit('The re-evaluated predictions do not reproduce the saved ones; check --entry and '
                         'that --splits indexes the same info file as the run.')


if __name__ == '__main__':
    main()
