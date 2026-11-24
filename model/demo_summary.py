"""Summary tables of a demo evaluation (demo_evaluation.py output) over the 30/50/70 % identity versions.

    cd model
    python demo_summary.py --dir ../data/demo_eval/finetuning_drop_RES_2_650M_dbptm

Reads sites.csv and re-applies each prediction's decision threshold from the run's metadata in
--logs (the fold's validation-tuned opt_threshold of threshold_metrics.py, else 0.5). This matters
when sites.csv was written where the metadata has no opt_threshold (e.g. on the cluster, from
new_logs/): its probabilities are kept, only its calls are redone. Writes into --dir:

    summary_overall.csv    per identity and group (S, T, S+T, Y), every demo protein pooled
    summary_proteins.csv   per protein, identity and group
    summary_canonical.csv  canonical sites called at the tuned threshold, per protein, identity and group
    summary.md             the S+T and Y groups of all three as Markdown tables
    summary_identity_<N>.csv, summary_by_identity.md
                           one table per identity version, one row per protein, S+T and Y side by side,
                           with the canonical sites each misses
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from demo_evaluation import fold_threshold, read_metadata

GROUPS = ['S', 'T', 'S+T', 'Y']
SHOWN = ['S+T', 'Y']


def thresholds(sites, roots):
    """{(run, fold): (threshold, source)} from the metadata of every run in `sites`."""
    out = {}
    for run, folds in sites.groupby('run')['fold'].unique().items():
        metadata, _ = read_metadata(roots, run)
        if metadata is None:
            raise SystemExit(f'no metadata of {run} in {roots}')
        for fold in folds:
            out[run, int(fold)] = fold_threshold(metadata, int(fold))
    return out


def metrics(g):
    g = g.dropna(subset=['prob'])
    y, p, call = g['label'].to_numpy(), g['prob'].to_numpy(), g['call'].to_numpy(dtype=bool)
    tp, fp, fn = int((call & (y == 1)).sum()), int((call & (y == 0)).sum()), int((~call & (y == 1)).sum())
    both = 0 < y.sum() < len(y)
    canonical = g[g['canonical'].notna()]
    return pd.Series({
        'n_cand': len(g), 'n_pos': int(y.sum()), 'TP': tp, 'FP': fp, 'FN': fn,
        'precision': tp / (tp + fp) if tp + fp else np.nan, 'recall': tp / (tp + fn) if tp + fn else np.nan,
        'mcc': matthews_corrcoef(y, call) if both else np.nan,
        'auroc': roc_auc_score(y, p) if both else np.nan, 'ap': average_precision_score(y, p) if both else np.nan,
        'n_canonical': len(canonical), 'canonical_called': int(canonical['call'].sum()),
    })


def fold_kind(relation):
    """How the fold scoring a protein relates to it, from demo_evaluation's `relation`."""
    if relation == 'in test':
        return 'own test fold'
    if 'no clean fold' in relation:
        return 'no clean fold'
    return "test fold of its cluster's representative"


# Markdown marks of fold_kind: the representative's fold is the common case and unmarked
MARKS = {'own test fold': '†', "test fold of its cluster's representative": '', 'no clean fold': '‡'}
LEGEND = ("Every protein is scored by a fold that trained on no member of its cluster, usually the one whose test "
          "set holds the cluster's representative; † its own test fold, the protein being that representative (for "
          "S+T: for both the S and the T model); ‡ no such fold exists, as the protein joins clusters that are in "
          "different folds. The predictions are always the protein's own.")


def fmt(x, digits=2):
    return '–' if pd.isna(x) else f'{x:.{digits}f}'


def markdown(overall, proteins, relation):
    lines = ['## All demo proteins pooled', '',
             '| Identity | Group | Candidates | Sites | AUROC | AP | MCC | Precision | Recall | Canonical called |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (identity, group), r in overall.loc[(slice(None), SHOWN), :].iterrows():
        lines.append(f'| {identity} % | {group} | {r.n_cand:.0f} | {r.n_pos:.0f} | {fmt(r.auroc)} | {fmt(r.ap)} | {fmt(r.mcc)} '
                     f'| {fmt(r.precision)} | {fmt(r.recall)} | {r.canonical_called:.0f}/{r.n_canonical:.0f} |')
    identities = sorted(proteins.index.get_level_values('identity').unique())
    for group in SHOWN:
        lines += ['', f'## Per protein, {group}', '',
                  'AP / AUROC, then sites called at the tuned threshold (TP/sites, FP), then canonical sites called. '
                  + LEGEND, '',
                  '| Protein | Sites | ' + ' | '.join(f'{i} % AP / AUROC | {i} % TP/sites (FP) | {i} % canonical'
                                                     for i in identities) + ' |',
                  '|---|---:|' + '---:|---:|---:|' * len(identities)]
        sub = proteins.xs(group, level='group')
        for protein in sub.index.get_level_values('protein').unique():
            cells = []
            for identity in identities:
                r = sub.loc[(identity, protein)]
                mark = MARKS[fold_kind(relation.get((identity, group, protein), ''))]
                cells += [f'{fmt(r.ap)} / {fmt(r.auroc)}{mark}', f'{r.TP:.0f}/{r.n_pos:.0f} ({r.FP:.0f})',
                          f'{r.canonical_called:.0f}/{r.n_canonical:.0f}' if r.n_canonical else '–']
            n_pos = sub.loc[(identities[0], protein), 'n_pos']
            lines.append(f'| {protein} | {n_pos:.0f} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def by_identity(proteins, sites, relation):
    """{identity: one row per protein, S+T and Y side by side} and their Markdown, one table per identity."""
    # Residues without a prediction (incomplete coverage) are neither called nor missed
    missed = (sites[sites['canonical'].notna() & sites['prob'].notna() & ~sites['call']]
              .groupby(['identity', 'group', 'protein'])['canonical'].agg(', '.join))
    uniprot = sites.drop_duplicates('protein').set_index('protein')['uniprot']
    tables, lines = {}, []
    for identity in sorted(proteins.index.get_level_values('identity').unique()):
        rows = []
        for protein in proteins.xs((identity, SHOWN[0]), level=['identity', 'group']).index:
            row = {'protein': protein, 'uniprot': uniprot[protein]}
            for group in SHOWN:
                r = proteins.loc[(identity, group, protein)]
                row.update({f'{group} {k}': r[k] for k in ['n_cand', 'n_pos', 'auroc', 'ap', 'TP', 'FP', 'FN',
                                                         'canonical_called', 'n_canonical']})
                row[f'{group} missed canonical'] = missed.get((identity, group, protein), '')
                row[f'{group} fold'] = fold_kind(relation.get((identity, group, protein), ''))
            rows.append(row)
        # Highest S+T average precision first, as in scripts/table_demo_identity.py
        table = pd.DataFrame(rows).sort_values(f'{SHOWN[0]} ap', ascending=False, na_position='last', kind='stable')
        tables[identity] = table

        lines += [f'## {identity} % identity', '',
                  '| Protein | ' + ' | '.join(f'{g} sites | {g} AP / AUROC | {g} TP / FP / FN | {g} canonical called '
                                              f'| {g} canonical missed' for g in SHOWN) + ' |',
                  '|---|' + '---:|---:|---:|---:|---|' * len(SHOWN)]
        for _, r in table.iterrows():
            cells = []
            for g in SHOWN:
                mark = MARKS[r[f'{g} fold']]
                cells += [f'{r[f"{g} n_pos"]:.0f}/{r[f"{g} n_cand"]:.0f}',
                          f'{fmt(r[f"{g} ap"])} / {fmt(r[f"{g} auroc"])}{mark}',
                          f'{r[f"{g} TP"]:.0f} / {r[f"{g} FP"]:.0f} / {r[f"{g} FN"]:.0f}',
                          f'{r[f"{g} canonical_called"]:.0f}/{r[f"{g} n_canonical"]:.0f}' if r[f'{g} n_canonical'] else '–',
                          r[f'{g} missed canonical'] or '–']
            lines.append(f'| {r["protein"]} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    header = ['# Demo proteins by identity version', '',
              'One table per identity version of the run family, one row per protein. Sites: known sites / '
              'candidate residues. TP / FP / FN and canonical sites at the fold\'s tuned threshold. ' + LEGEND, '',
              'AP / AUROC are undefined (–) when every candidate is a site.', '']
    return tables, '\n'.join(header + lines)


def main():
    parser = argparse.ArgumentParser(description='Summary tables of a demo evaluation.')
    parser.add_argument('--dir', required=True, help='Output folder of demo_evaluation.py (with sites.csv, proteins.csv).')
    parser.add_argument('--logs', nargs='+', default=['logs'], help='Where the runs\' metadata.json are (default: logs).')
    cli = parser.parse_args()

    sites = pd.read_csv(os.path.join(cli.dir, 'sites.csv'))
    proteins_in = pd.read_csv(os.path.join(cli.dir, 'proteins.csv'), dtype=str)
    th = thresholds(sites, cli.logs)
    sites['threshold'] = [th[r, int(f)][0] for r, f in zip(sites['run'], sites['fold'])]
    sites['call'] = sites['prob'] >= sites['threshold']
    used = pd.Series([s for _, s in th.values()]).value_counts()
    print('thresholds:', ', '.join(f'{n} {s}' for s, n in used.items()))

    sites = sites[sites['group'].isin(GROUPS)]
    overall = sites.groupby(['identity', 'group']).apply(metrics, include_groups=False)
    overall = overall.reindex(pd.MultiIndex.from_product([sorted(sites['identity'].unique()), GROUPS],
                                                         names=['identity', 'group'])).dropna(how='all')
    order = list(dict.fromkeys(sites['protein']))
    proteins = sites.groupby(['identity', 'group', 'protein'], sort=False).apply(metrics, include_groups=False)
    proteins = proteins.reset_index()
    proteins['protein'] = pd.Categorical(proteins['protein'], order, ordered=True)
    proteins = proteins.sort_values(['identity', 'group', 'protein']).set_index(['identity', 'group', 'protein'])
    for frame in (overall, proteins):
        for c in ['n_cand', 'n_pos', 'TP', 'FP', 'FN', 'n_canonical', 'canonical_called']:
            frame[c] = frame[c].astype(int)
    # The protein itself is in its fold's test set (†), or it is not in the splits and the fold is the one
    # testing its cluster's representative; a pooled S+T relation reads e.g.
    # 'S:in test T:cluster in test (rep X)' and counts as the protein itself only if both parts do
    relation = {}
    for i, g, p, r in zip(proteins_in['identity'].astype(int), proteins_in['group'], proteins_in['protein'],
                          proteins_in['relation']):
        parts = [x.strip() for x in re.split(r'[STY]:', re.sub(r' \(rep [^)]*\)', '', r)) if x.strip()]
        relation[i, g, p] = 'in test' if all(x == 'in test' for x in parts) else r

    canonical = (sites[sites['canonical'].notna()]
                 .pivot_table(index=['protein', 'canonical', 'group'], columns='identity', values='call', aggfunc='first')
                 .rename(columns=lambda i: f'called_{i}'))
    proteins_out = proteins.reset_index()
    proteins_out['relation'] = [relation.get((i, g, p), '') for i, g, p in
                                zip(proteins_out['identity'], proteins_out['group'], proteins_out['protein'])]

    overall.round(4).to_csv(os.path.join(cli.dir, 'summary_overall.csv'))
    proteins_out.round(4).to_csv(os.path.join(cli.dir, 'summary_proteins.csv'), index=False)
    canonical.to_csv(os.path.join(cli.dir, 'summary_canonical.csv'))
    with open(os.path.join(cli.dir, 'summary.md'), 'w') as f:
        f.write(markdown(overall, proteins, relation))
    tables, text = by_identity(proteins, sites, relation)
    for identity, table in tables.items():
        table.round(4).to_csv(os.path.join(cli.dir, f'summary_identity_{identity}.csv'), index=False)
    with open(os.path.join(cli.dir, 'summary_by_identity.md'), 'w') as f:
        f.write(text)

    pd.set_option('display.width', 250)
    print(overall[['n_cand', 'n_pos', 'auroc', 'ap', 'mcc', 'precision', 'recall', 'canonical_called', 'n_canonical']]
          .round(3).to_string())
    print(f'\nWritten to {cli.dir}/: summary_overall.csv, summary_proteins.csv, summary_canonical.csv, summary.md, '
          f'summary_identity_*.csv, summary_by_identity.md')


if __name__ == '__main__':
    main()
