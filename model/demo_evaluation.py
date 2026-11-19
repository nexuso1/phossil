"""Per-protein demo evaluation of one run family on a fixed set of well-known phosphoproteins.

For every residue model of the family (e.g. the S, T and Y runs) and every clustering identity
version (30/50/70 %), each demo protein is scored by the fold model that never trained on it or on
its cluster, and the result is reported per candidate residue, per protein and per canonical site.
The canonical sites are the sites labelled functional in --canonical, hand labels of each protein's
UniProt phosphosite mutants (see demo_mutagenesis.py).

    cd model
    python demo_evaluation.py --run 'finetuning_drop_{res}_2_650M_dbptm_{variant}'
    python demo_evaluation.py --run 'lora_rank16_{res}_2_650M_dbptm_{variant}' --residues S T Y
    # on the machine with the checkpoints, to fill in what no saved prediction covers:
    python demo_evaluation.py --run '...' --predict --ckpt_root new_logs

`{res}` is replaced by each of --residues and `{variant}` by each of --variants (chunked_max,
chunked_50_max, chunked_70_max). The S and T runs are also pooled into an "S+T" group, each residue
scored by its own model and its own fold threshold.

Choice of fold
--------------
Each run's splits hold one protein per cluster of its identity level (dbptm_clusters.tsv at 30 %,
dbptm_50.tsv, dbptm_70.tsv). Folds are ranked by, in order:
  1. whether the protein itself is in the fold's train or dev partition,
  2. how many members of its cluster at the run's identity level are in train/dev,
  3. how many members of its clusters at the other identity levels are in train/dev, 30 % first,
and the first fold wins (ties go to the lowest fold). So a protein in a test set is scored by that
fold. A protein that is not in the splits (its cluster is represented by another member) is scored
by the fold whose test set holds that representative, so no member of its cluster was trained on.
A cluster with no member in the splits is unconstrained at its own level, and rule 3 picks the fold
with the fewest lower-identity homologs. The `relation` column says which case applies, and the
`train_mates_<id>` columns how many cluster members at each identity level the chosen fold trained on.
At 50 % and 70 % those lower-identity homologs are expected: that is what the higher threshold means.

Sources of predictions
----------------------
Predictions are looked up in the chosen fold directory, under --logs and --ckpt_root, in order:
  demo_preds_fold_N_stitched.json      written by this script with --predict, whole proteins
  *_whole_preds_fold_N_stitched.json   written by evaluate_checkpoints.py, whole proteins
  test_preds_fold_N_stitched.json      the run's own test predictions
The first source covering every candidate residue of the protein is used. The run's own test
predictions miss the site-less chunks of long proteins (see the CLAUDE.md gotcha on the residue
filter), so they can be partial; `coverage` gives the predicted fraction and the missing residues
have an empty `prob`. With --predict, every protein without a complete source is run through
<ckpt_root>/<run>/fold_N/best.ckpt (whole protein, every chunk) and saved to demo_preds_fold_N_
stitched.json in that fold directory, which later invocations pick up. Copy those files into
logs/<run>/fold_N/ to use them on a machine without the checkpoints. As a check that the model was
restored correctly, a protein of the fold's own test set is predicted alongside and compared with
the saved test predictions. The control must have every one of its chunks in the test list: the
residue filter drops site-less chunks from the splits, and a protein missing some would be stitched
from different chunks here than in the saved predictions, which differ by more than restoring noise.

Calls use the fold's validation-tuned `opt_threshold` (threshold_metrics.py), or 0.5 where the run
has none (`threshold_source`).

Output, in --out_dir (default ../data/demo_eval/<run template without {variant}, {res} -> RES>):
  sites.csv      every candidate residue of every demo protein, per variant and group
  proteins.csv   per protein: fold choice, source, coverage, counts, AUROC / AP
  canonical.csv  the canonical sites, one row per site, prob / call / rank per variant, plus their
                 UniProt literature evidence (evidence, via, pubmed) from demo_site_references.py
"""
import argparse
import functools
import glob
import importlib
import json
import os
import warnings
from argparse import Namespace
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from data_loading import labeling_fn, parse_residues

# Splits variant -> identity threshold it was clustered at, and the cluster file of each threshold
VARIANTS = {'chunked_max': 30, 'chunked_50_max': 50, 'chunked_70_max': 70}
CLUSTER_FILES = {30: '../data/dbptm/dbptm_clusters.tsv', 50: '../data/dbptm/dbptm_50.tsv',
                 70: '../data/dbptm/dbptm_70.tsv'}

# Run name prefix -> entry point whose create_model rebuilds it (first match wins)
ENTRY_PREFIXES = [('finetuning_multi', 'ft_multi_residue'), ('finetuning', 'ft_selective'),
                  ('ft_sweep', 'ft_selective'), ('ft_recycle', 'ft_recycling'), ('lora', 'lora_model'),
                  ('llrd', 'llrd_model'), ('recycle_enc', 'recycling_encoder'), ('baseline', 'baseline')]

# Logit differences above this against the saved predictions mean a wrongly restored model
MAX_LOGIT_DIFF = 1e-3


# UniProt accession -> (name, {UniProt position: position in the numbering the literature uses}).
# The 20 human proteins with the most site-specific phosphosite papers in UniProt
# (scripts/uniprot_site_literature.py: distinct ECO:0000269 papers, ties broken by site-papers, then
# by PhosphoSitePlus rank). STAT1, STAT3, SP1 and TFEB are not in the dbPTM info file; their sequence,
# dbPTM sites and cluster assignments come from EXTRA_PROTEINS / EXTRA_CLUSTERS (demo_extra_proteins.py).
# The canonical sites are not listed here: they are the sites labelled functional in CANONICAL_LABELS
# (see demo_mutagenesis.py).
DEMO = {
    'P04637': ('TP53', {}),
    'P31749': ('AKT1', {}),
    # H2AX sites are named without the initiator methionine (S140 is gamma-H2AX S139)
    'P16104': ('H2AX', {140: 139, 143: 142}),
    'P46527': ('CDKN1B', {}),
    'O14757': ('CHEK1', {}),
    'P42224': ('STAT1', {}),
    'Q00613': ('HSF1', {}),
    'P04049': ('RAF1', {}),
    'P40763': ('STAT3', {}),
    'Q99683': ('MAP3K5', {}),
    'Q04206': ('RELA', {}),
    # EGFR sites are named in mature numbering, without the 24-residue signal peptide
    'P00533': ('EGFR', {p: p - 24 for p in (693, 1016, 1069, 1092, 1197)}),
    # P10636 is the 758-residue PNS isoform; the literature uses 2N4R (441-residue) numbering, 317 lower
    # from the PNS-specific insert on
    'P10636': ('MAPT', {p: p - 317 for p in (515, 516, 519, 531, 548, 552, 579, 713, 721, 726, 739)}),
    'P25963': ('NFKBIA', {}),
    'P08047': ('SP1', {}),
    'P50750': ('CDK9', {}),
    'P19484': ('TFEB', {}),
    'P06401': ('PGR', {}),
    'Q12778': ('FOXO1', {}),
    'P35222': ('CTNNB1', {}),
}
CANONICAL_LABELS = '../data/demo_eval/mutagenesis_labels.csv'
EXTRA_PROTEINS = '../data/demo_eval/extra_proteins.json'
EXTRA_CLUSTERS = '../data/demo_eval/extra_clusters.tsv'


def load_extra_proteins(path=EXTRA_PROTEINS):
    """Info-file rows of the demo proteins missing from the dbPTM info file (demo_extra_proteins.py)."""
    return pd.read_json(path) if os.path.exists(path) else pd.DataFrame(columns=['id', 'sequence', 'sites', 'parent_id', 'offset'])


def demo_sequences(info_path='../data/dbptm/dbptm_info.json'):
    """{accession: sequence} of every demo protein, from the unchunked info file or the extra proteins."""
    info = pd.read_json(info_path)
    extra = load_extra_proteins()
    out = {**dict(zip(extra['id'], extra['sequence'])),
           **dict(zip(info['id'], info['sequence']))}
    missing = set(DEMO) - set(out)
    if missing:
        raise SystemExit(f'demo protein(s) {sorted(missing)} neither in {info_path} nor in {EXTRA_PROTEINS} '
                         f'(run demo_extra_proteins.py)')
    return {pid: out[pid] for pid in DEMO}


@functools.lru_cache(maxsize=None)
def load_labels(path=CANONICAL_LABELS):
    """The hand labels of demo_mutagenesis.py: protein, site (UniProt numbering), functional (0/1)."""
    return pd.read_csv(path)


def resolve_sites(pid, sequence, labels=None):
    """[(0-based position, display name)] of the canonical sites of `pid`: the sites labelled functional,
    with the literature position in parentheses where it differs, checked against the sequence."""
    name, aliases = DEMO[pid]
    labels = load_labels() if labels is None else labels
    out = []
    for site in labels.loc[(labels['protein'] == name) & (labels['functional'] == 1), 'site']:
        residue, position = site[0], int(site[1:])
        if position > len(sequence) or sequence[position - 1] != residue:
            raise ValueError(f'{name}: canonical site {site} does not match the sequence')
        alias = aliases.get(position)
        out.append((position - 1, site + (f' ({residue}{alias})' if alias else '')))
    return out


def label_chunks(chunks, residues, ignore_index):
    """`chunks` with 0-based S/T/Y sites and a 'label' column for `residues`, as load_prot_data makes them."""
    chunks = chunks.copy()
    chunks['sites'] = [[s - 1 for s in sites if seq[s - 1] in 'STY'] for seq, sites in zip(chunks['sequence'], chunks['sites'])]
    chunks['label'] = chunks.apply(partial(labeling_fn, residues=residues, ignore_index=ignore_index), axis=1)
    return chunks[['id', 'sequence', 'label', 'parent_id', 'offset']]


class ProteinInfo:
    """The chunked info file, with whole proteins rebuilt from their chunks for the demo proteins.
    Demo proteins missing from it are appended from EXTRA_PROTEINS, under index labels past the file's
    own, which no splits file refers to."""

    def __init__(self, path):
        df = pd.read_json(path)
        if 'parent_id' not in df:
            df['parent_id'], df['offset'] = df['id'], 0
        extra = load_extra_proteins()
        extra = extra[extra['parent_id'].isin(set(DEMO) - set(df['parent_id']))]
        if len(extra):
            extra = extra[['id', 'sequence', 'sites', 'parent_id', 'offset']].copy()
            extra.index = range(df.index.max() + 1, df.index.max() + 1 + len(extra))
            df = pd.concat([df, extra])
        self.df = df
        self.parent_of = df['parent_id']
        demo = df[df['parent_id'].isin(DEMO)]
        missing = set(DEMO) - set(demo['parent_id'])
        if missing:
            raise ValueError(f'demo protein(s) {sorted(missing)} not in {path}')
        self.sequence = {}
        self.sites = {}
        for pid, group in demo.groupby('parent_id'):
            length = max(o + len(s) for o, s in zip(group['offset'], group['sequence']))
            seq = [''] * length
            sites = set()
            for row in group.itertuples():
                seq[row.offset:row.offset + len(row.sequence)] = row.sequence
                # 1-based chunk positions in the info file
                sites |= {row.offset + s - 1 for s in row.sites if row.sequence[s - 1] in 'STY'}
            self.sequence[pid] = ''.join(seq)
            self.sites[pid] = sites

    def complete_in(self, labels):
        """Proteins every chunk of which is among the info file index `labels`."""
        listed = self.parent_of.loc[list(labels)].value_counts()
        total = self.parent_of.value_counts()
        return set(listed.index[listed == total.loc[listed.index]])

    def chunks_of(self, proteins, residues, ignore_index):
        """Every chunk of `proteins`, labelled for `residues`."""
        return label_chunks(self.df[self.df['parent_id'].isin(proteins)], residues, ignore_index)


def load_clusters():
    """{identity: Series member -> representative}, for the identity levels whose file exists. The 222
    proteins listed twice in the info file are listed twice in the cluster files too, with the same
    representative."""
    out = {}
    for identity, path in CLUSTER_FILES.items():
        if os.path.exists(path):
            clusters = pd.read_csv(path, sep='\t', header=None, names=['rep', 'mem']).drop_duplicates('mem')
            out[identity] = clusters.set_index('mem')['rep']
    return out


def cluster_members(clusters):
    """{identity: {demo protein: set of the other members of its cluster}}. A protein from
    EXTRA_CLUSTERS gets the members of every cluster it was assigned to at that level."""
    extra = pd.read_csv(EXTRA_CLUSTERS, sep='\t') if os.path.exists(EXTRA_CLUSTERS) else None
    out = {}
    for identity, rep_of in clusters.items():
        out[identity] = {}
        for pid in DEMO:
            if pid in rep_of.index:
                reps = {rep_of[pid]}
            elif extra is not None and pid in set(extra['protein']):
                reps = set(extra.loc[(extra['protein'] == pid) & (extra['identity'] == identity), 'rep'])
            else:
                reps = set()
            members = set(rep_of.index[rep_of.isin(reps)])
            out[identity][pid] = members - {pid}
    return out


def choose_fold(pid, partitions, members, own_identity):
    """(fold, relation, {identity: cluster members in that fold's train/dev}) for one protein."""
    order = [own_identity] + sorted(i for i in members if i != own_identity)
    best = None
    for fold, (train, test) in enumerate(partitions):
        counts = {i: len(members[i][pid] & train) for i in order}
        key = (pid in train, *[counts[i] for i in order])
        if best is None or key < best[0]:
            best = (key, fold, counts)
    _, fold, counts = best
    train, test = partitions[fold]
    mates_in_splits = {p for p in members[own_identity][pid] if any(p in tr or p in te for tr, te in partitions)}
    if pid in test:
        relation = 'in test'
    elif pid in train:
        relation = 'in train (no clean fold)'
    elif mates_in_splits & test:
        relation = 'cluster in test (rep ' + ','.join(sorted(mates_in_splits & test)) + ')'
        # Only a protein joining several clusters (EXTRA_CLUSTERS) can have them in different folds
        if counts[own_identity]:
            relation += f', {counts[own_identity]} member(s) in train (no clean fold)'
    elif not mates_in_splits:
        relation = 'cluster not in splits'
    else:
        relation = 'cluster in train (no clean fold)'
    return fold, relation, counts


class PredictionFiles:
    """Lazily loaded stitched prediction files, indexed by protein id."""

    def __init__(self):
        self.cache = {}

    def get(self, path, pid):
        if path not in self.cache:
            df = pd.read_json(path)
            self.cache[path] = {row.id: row for row in df.itertuples()}
        return self.cache[path].get(pid)

    def forget(self, path):
        self.cache.pop(path, None)


def fold_dirs(roots, run, fold):
    seen = []
    for root in roots:
        path = os.path.normpath(os.path.join(root, run, f'fold_{fold}'))
        if os.path.isdir(path) and path not in seen:
            seen.append(path)
    return seen


def find_predictions(files, dirs, fold, pid, candidates):
    """(source file, {position: logit}, coverage) of the best prediction of `pid` in `dirs`."""
    patterns = [f'demo_preds_fold_{fold}_stitched.json', f'*_whole_preds_fold_{fold}_stitched.json',
                f'test_preds_fold_{fold}_stitched.json']
    best = (None, {}, 0.0)
    for pattern in patterns:
        for d in dirs:
            for path in sorted(glob.glob(os.path.join(d, pattern))):
                row = files.get(path, pid)
                if row is None:
                    continue
                logits = {int(p): float(l) for p, l in zip(row.sequence_indices, row.logits)}
                coverage = len(candidates & logits.keys()) / len(candidates)
                if coverage > best[2]:
                    best = (path, logits, coverage)
                if coverage == 1:
                    return best
    return best


def entry_for(run, override):
    if override:
        return override
    for prefix, entry in ENTRY_PREFIXES:
        if run.startswith(prefix):
            return entry
    raise SystemExit(f'cannot tell the entry point of {run} from its name, pass --entry')


_ADDED_ARGUMENTS = set()


def load_run_args(metadata, module):
    """Training args of the run, with the parser defaults filling in args older metadata lacks."""
    from training import parser as training_parser
    if hasattr(module, 'add_arguments') and module.__name__ not in _ADDED_ARGUMENTS:
        module.add_arguments(training_parser)
        _ADDED_ARGUMENTS.add(module.__name__)
    args = Namespace(**{**vars(training_parser.parse_args([])), **metadata['args']})
    args.compile = False
    args.grad_checkpointing = False
    return args


def predict_fold(run, run_dir, fold, metadata, entry, info, proteins, control, batch_size):
    """Runs fold_N/best.ckpt on every chunk of `proteins` (+ the control protein).
    Returns the stitched predictions of `proteins`, and those of the control protein for the
    comparison with the saved ones; nothing is written, so a failed comparison leaves no file behind."""
    import lightning as L
    import torch
    from torch.utils.data import DataLoader

    from data_loading import prep_batch, stitch_chunk_predictions
    from prot_dataset import ProteinDataset
    from training import LightningWrapper, create_metrics, prepare_model

    fold_dir = os.path.join(run_dir, f'fold_{fold}')
    ckpt_path = os.path.join(fold_dir, 'best.ckpt')
    if os.path.isdir(ckpt_path):
        raise SystemExit(f'{ckpt_path} is a sharded (deepspeed) checkpoint, consolidate it first')

    module = importlib.import_module(entry)
    args = load_run_args(metadata, module)
    if batch_size:
        args.batch_size = batch_size
    if args.kinase:
        raise SystemExit(f'{run} predicts kinases, which this script does not handle')
    residues = parse_residues(args.residues)

    chunks = info.chunks_of(list(proteins) + ([control] if control else []), residues, args.ignore_label)
    chunks = chunks.reset_index(drop=True)
    dataset = ProteinDataset(chunks)

    L.seed_everything(args.seed, verbose=False)
    model, tokenizer = prepare_model(args, module.create_model)
    step_metrics, epoch_metrics = create_metrics(args.ignore_label)
    wrapper = LightningWrapper(args, model, epoch_metrics=epoch_metrics, step_metrics=step_metrics,
                               ds_size=len(dataset), logdir=fold_dir, train_epochs=args.epochs, lr=args.lr)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    wrapper.load_state_dict(checkpoint['state_dict'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label,
                                           modify_prob=0, kinases=False))
    trainer = L.Trainer(logger=False, enable_checkpointing=False, devices=1, deterministic=True,
                        enable_progress_bar=False)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*but have no logger configured.*')
        trainer.test(wrapper, loader, verbose=False)

    preds = pd.DataFrame.from_records(wrapper.test_preds, columns=['logits', 'labels', 'sequence_indices', 'df_index'])
    stitched = stitch_chunk_predictions(preds, chunks)
    control_row = stitched[stitched['id'] == control]
    return stitched[stitched['id'] != control], (control_row.iloc[0] if len(control_row) else None)


def save_demo_predictions(fold_dir, fold, stitched):
    """Adds `stitched` to the fold's demo_preds_fold_N_stitched.json, replacing older rows of the same proteins."""
    out_path = os.path.join(fold_dir, f'demo_preds_fold_{fold}_stitched.json')
    if os.path.exists(out_path):
        previous = pd.read_json(out_path)
        stitched = pd.concat([previous[~previous['id'].isin(stitched['id'])], stitched], ignore_index=True)
    stitched.reset_index(drop=True).to_json(out_path)
    return out_path


def check_control(control_pred, saved_path, control):
    """Max |logit difference| of the control protein against the run's saved test predictions."""
    saved = pd.read_json(saved_path)
    saved = saved[saved['id'] == control].iloc[0]
    old = dict(zip(saved['sequence_indices'], saved['logits']))
    new = dict(zip(control_pred['sequence_indices'], control_pred['logits']))
    shared = old.keys() & new.keys()
    return max(abs(old[k] - new[k]) for k in shared) if shared else float('nan'), len(shared)


def read_metadata(roots, run):
    for root in roots:
        path = os.path.join(root, run, 'metadata.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)['data'], os.path.join(root, run)
    return None, None


def fold_threshold(metadata, fold):
    folds = metadata.get('test_metrics') or []
    folds = folds if isinstance(folds, list) else [folds[k] for k in sorted(folds)]
    if fold < len(folds) and isinstance(folds[fold], dict) and 'opt_threshold' in folds[fold]:
        return float(folds[fold]['opt_threshold']), 'opt_threshold'
    return 0.5, 'default 0.5'


def score(group):
    """Counts and threshold-free metrics over the predicted candidates of one protein."""
    g = group.dropna(subset=['prob'])
    y, p, call = g['label'].to_numpy(), g['prob'].to_numpy(), g['call'].to_numpy(dtype=bool)
    out = {'n_cand': len(group), 'n_pred': len(g), 'n_pos': int(group['label'].sum()),
           'TP': int((call & (y == 1)).sum()), 'FP': int((call & (y == 0)).sum()), 'FN': int((~call & (y == 1)).sum()),
           'auroc': np.nan, 'ap': np.nan}
    if 0 < y.sum() < len(y):
        out['auroc'] = roc_auc_score(y, p)
        out['ap'] = average_precision_score(y, p)
    return out


def evaluate_run(run, variant, residue, cli, roots, info_cache, clusters, members, files):
    """Rows of sites.csv and selection info for one run, or None if the run does not exist."""
    metadata, meta_dir = read_metadata(roots, run)
    if metadata is None:
        print(f'note: no run {run}')
        return None
    args = metadata.get('args') or {}
    expected = f'splits_{residue}_{variant}.json'
    if os.path.basename(str(args.get('dataset_path', ''))) != expected:
        raise SystemExit(f'{run} was trained on {args.get("dataset_path")}, not {expected}')
    residues = parse_residues(args['residues'])
    identity = VARIANTS[variant]
    if identity not in clusters:
        raise SystemExit(f'cluster file {CLUSTER_FILES[identity]} for {variant} not found')

    info_path = args['prot_info_path']
    if info_path not in info_cache:
        info_cache[info_path] = ProteinInfo(info_path)
    info = info_cache[info_path]
    with open(args['dataset_path']) as f:
        splits = json.load(f)
    parent = info.parent_of
    partitions = []
    for split in splits:
        train = set(parent.loc[split['train'] + split.get('dev', [])])
        partitions.append((train, set(parent.loc[split['test']])))

    finished = metadata.get('fold_finished') or [True] * len(splits)
    choice = {pid: choose_fold(pid, partitions, members, identity) for pid in DEMO}
    unfinished = sorted({fold for fold, _, _ in choice.values() if not finished[fold]})
    if unfinished:
        print(f'note: {run} fold(s) {unfinished} are not finished, their proteins are skipped')

    def lookup(pid):
        fold = choice[pid][0]
        candidates = {i for i, a in enumerate(info.sequence[pid]) if a in residues}
        return candidates, find_predictions(files, fold_dirs(roots, run, fold), fold, pid, candidates)

    # Proteins no saved prediction covers completely, by fold
    todo = {}
    for pid in DEMO:
        fold = choice[pid][0]
        if not finished[fold]:
            continue
        candidates, (_, _, coverage) = lookup(pid)
        if candidates and coverage < 1:
            todo.setdefault(fold, []).append(pid)

    if cli.predict and todo:
        ckpt_run_dir = os.path.join(cli.ckpt_root, run)
        entry = entry_for(run, cli.entry)
        for fold, proteins in sorted(todo.items()):
            if not os.path.exists(os.path.join(ckpt_run_dir, f'fold_{fold}', 'best.ckpt')):
                print(f'note: no {ckpt_run_dir}/fold_{fold}/best.ckpt, {", ".join(DEMO[p][0] for p in proteins)} stay uncovered')
                continue
            saved = os.path.join(meta_dir, f'fold_{fold}', f'test_preds_fold_{fold}_stitched.json')
            control = None
            if os.path.exists(saved):
                saved_ids = pd.read_json(saved)['id']
                # predict_fold runs every chunk, so a control with chunks missing from the test
                # list would be stitched differently than in the saved predictions
                complete = info.complete_in(splits[fold]['test'])
                control = next((p for p in saved_ids if p not in DEMO and p in complete), None)
            print(f'{run} fold {fold}: predicting {", ".join(DEMO[p][0] for p in proteins)} with {entry}')
            stitched, control_pred = predict_fold(run, ckpt_run_dir, fold, metadata, entry, info, proteins,
                                                  control, cli.batch_size)
            if control_pred is not None:
                diff, shared = check_control(control_pred, saved, control)
                print(f'  control {control}: {shared} positions, max |logit diff| vs saved {diff:.2e}')
                if not diff <= MAX_LOGIT_DIFF:
                    raise SystemExit(f'{run} fold {fold} does not reproduce its saved test predictions; '
                                     f'check --entry ({entry}) and the checkpoint')
            else:
                print(f'  no saved test prediction of a protein with all its chunks in fold {fold}\'s test list '
                      f'to check the restored model against')
            files.forget(save_demo_predictions(os.path.join(ckpt_run_dir, f'fold_{fold}'), fold, stitched))
        todo = {}
        for pid in DEMO:
            fold = choice[pid][0]
            candidates, (_, _, coverage) = lookup(pid)
            if finished[fold] and candidates and coverage < 1:
                todo.setdefault(fold, []).append(pid)

    rows, selection = [], []
    for pid, (name, _) in DEMO.items():
        fold, relation, counts = choice[pid]
        if not finished[fold]:
            continue
        candidates, (source, logits, coverage) = lookup(pid)
        if not candidates:
            continue
        threshold, threshold_source = fold_threshold(metadata, fold)
        canonical = dict(resolve_sites(pid, info.sequence[pid], load_labels(cli.canonical)))
        seq = info.sequence[pid]
        for position in sorted(candidates):
            logit = logits.get(position)
            prob = 1 / (1 + np.exp(-logit)) if logit is not None else np.nan
            rows.append({'variant': variant, 'identity': identity, 'model': residue, 'run': run, 'protein': name,
                         'uniprot': pid, 'fold': fold, 'position': position + 1, 'residue': seq[position],
                         'canonical': canonical.get(position, ''), 'label': int(position in info.sites[pid]),
                         'prob': prob, 'threshold': threshold, 'call': bool(prob >= threshold) if logit is not None else np.nan})
        selection.append({'variant': variant, 'identity': identity, 'model': residue, 'run': run, 'protein': name,
                          'uniprot': pid, 'fold': fold, 'relation': relation,
                          **{f'train_mates_{i}': counts[i] for i in sorted(counts)},
                          'source': os.path.basename(source) if source else 'none', 'coverage': round(coverage, 3),
                          'threshold': round(threshold, 4), 'threshold_source': threshold_source})
    return rows, selection, todo


def pooled_groups(sites):
    """Adds an S+T group pooling the S and T models, when both were evaluated for a variant."""
    sites = sites.assign(group=sites['model'])
    pooled = []
    for variant, g in sites.groupby('variant'):
        if {'S', 'T'} <= set(g['model']):
            pooled.append(g[g['model'].isin(['S', 'T'])].assign(group='S+T'))
    return pd.concat([sites, *pooled], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description='Per-protein demo evaluation of one run family across identity versions.')
    parser.add_argument('--run', required=True,
                        help="Run directory name template with {res} and {variant}, e.g. 'finetuning_drop_{res}_2_650M_dbptm_{variant}'.")
    parser.add_argument('--residues', nargs='+', default=['S', 'T', 'Y'], help='Values of {res} (default: S T Y).')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS),
                        help='Values of {variant} (default: all three identity versions).')
    parser.add_argument('--logs', default='logs', help='Root of the archived runs (default: logs).')
    parser.add_argument('--ckpt_root', default='new_logs',
                        help='Root holding <run>/fold_N/best.ckpt, also searched for predictions (default: new_logs).')
    parser.add_argument('--predict', action='store_true',
                        help='Run the chosen fold checkpoints on the proteins no saved prediction fully covers.')
    parser.add_argument('--entry', default=None, help='Entry point module of the runs (default: from the run name).')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for --predict (default: 1).')
    parser.add_argument('--out_dir', default=None, help='Output directory (default: ../data/demo_eval/<template>).')
    parser.add_argument('--canonical', default=CANONICAL_LABELS,
                        help='Hand labels of the demo sites (demo_mutagenesis.py); the functional ones are canonical.')
    parser.add_argument('--references', default='../data/demo_eval/canonical_references.csv',
                        help='UniProt evidence of the canonical sites from demo_site_references.py, merged into canonical.csv.')
    cli = parser.parse_args()

    if '{res}' not in cli.run:
        parser.error('--run needs a {res} placeholder')
    roots = [cli.logs, cli.ckpt_root]
    clusters = load_clusters()
    members = cluster_members(clusters)
    files = PredictionFiles()
    info_cache = {}

    site_rows, selection, uncovered = [], [], {}
    for variant in cli.variants:
        for residue in cli.residues:
            run = cli.run.format(res=residue, variant=variant)
            result = evaluate_run(run, variant, residue, cli, roots, info_cache, clusters, members, files)
            if result is None:
                continue
            rows, sel, todo = result
            site_rows += rows
            selection += sel
            if todo:
                uncovered[run] = todo
    if not site_rows:
        raise SystemExit('no run of the template found')

    sites = pooled_groups(pd.DataFrame(site_rows))
    keys = ['variant', 'identity', 'group', 'protein', 'uniprot']
    sites['rank'] = sites.groupby(keys)['prob'].rank(ascending=False, method='min')
    sites['n_in_protein'] = sites.groupby(keys)['position'].transform('size')

    summary = pd.DataFrame([{**dict(zip(keys, k)), **score(g)} for k, g in sites.groupby(keys, sort=False)])
    sel = pd.DataFrame(selection)
    agg = sel.assign(group=sel['model']).pipe(pooled_groups_selection)
    proteins = summary.merge(agg, on=['variant', 'group', 'uniprot'], how='left')
    proteins = proteins.sort_values(['protein', 'group', 'identity'])

    canonical = sites[sites['canonical'] != ''].copy()
    canonical['call'] = canonical['call'].map({True: 'Y', False: '-'}).fillna('?')
    canonical['rank_str'] = canonical.apply(
        lambda r: f'{int(r["rank"])}/{r["n_in_protein"]}' if r['rank'] == r['rank'] else '', axis=1)
    wide = canonical[canonical['group'].isin(['S', 'T', 'Y', 'ST', 'STY'])].pivot_table(
        index=['protein', 'uniprot', 'canonical', 'position', 'model'], columns='identity',
        values=['prob', 'call', 'rank_str'], aggfunc='first')
    wide.columns = [f'{value}_{identity}' for value, identity in wide.columns]
    wide = wide.reset_index().sort_values(['protein', 'position'])
    order = ['protein', 'uniprot', 'canonical', 'position', 'model'] + [
        f'{v}_{i}' for i in sorted(sites['identity'].unique()) for v in ('prob', 'call', 'rank_str') if f'{v}_{i}' in wide]
    wide = wide[order]
    if os.path.exists(cli.references):
        refs = pd.read_csv(cli.references, dtype={'pubmed': str}, keep_default_na=False)
        wide = wide.merge(refs[['uniprot', 'position', 'evidence', 'via', 'pubmed']], on=['uniprot', 'position'], how='left')
    else:
        print(f'note: no {cli.references}, canonical.csv carries no literature evidence '
              f'(run demo_site_references.py)')

    name = cli.run.replace('_{variant}', '').replace('{variant}', '').replace('{res}', 'RES')
    out_dir = cli.out_dir or os.path.join('..', 'data', 'demo_eval', name)
    os.makedirs(out_dir, exist_ok=True)
    sites.drop(columns=['n_in_protein']).to_csv(os.path.join(out_dir, 'sites.csv'), index=False)
    proteins.to_csv(os.path.join(out_dir, 'proteins.csv'), index=False)
    wide.to_csv(os.path.join(out_dir, 'canonical.csv'), index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 50)
    shown = ['protein', 'group', 'identity', 'fold', 'relation', 'coverage', 'n_cand', 'n_pos', 'TP', 'FP', 'FN', 'auroc', 'ap']
    print(proteins[[c for c in shown if c in proteins]].round(3).to_string(index=False))
    print()
    print(wide.round(2).to_string(index=False))
    if uncovered:
        print('\nNot fully covered by any saved prediction (run with --predict where the checkpoints are):')
        for run, todo in uncovered.items():
            for fold, pids in sorted(todo.items()):
                print(f'  {run} fold {fold}: {", ".join(DEMO[p][0] for p in pids)}')
    print(f'\nWritten to {out_dir}/: sites.csv, proteins.csv, canonical.csv')


def pooled_groups_selection(sel):
    """Selection rows of the S+T group: the S and T rows of a protein joined into one."""
    pooled = []
    for (variant, uniprot), g in sel[sel['model'].isin(['S', 'T'])].groupby(['variant', 'uniprot']):
        if set(g['model']) != {'S', 'T'}:
            continue
        joined = {'variant': variant, 'uniprot': uniprot, 'group': 'S+T'}
        for column in g.columns:
            if column in joined or column in ('identity', 'protein', 'model', 'run'):
                continue
            values = [f'{m}:{v}' for m, v in zip(g['model'], g[column])]
            joined[column] = ' '.join(values) if len(set(g[column])) > 1 else g[column].iloc[0]
        pooled.append(joined)
    out = pd.concat([sel, pd.DataFrame(pooled)], ignore_index=True)
    return out.drop(columns=['identity', 'protein', 'model', 'run'])


if __name__ == '__main__':
    main()
