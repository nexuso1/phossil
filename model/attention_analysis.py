# Attention analysis of the files written by compute_embeddings.py --attention_layers, the script
# form of notebooks/attention_viz.ipynb. Every file is read once: each chunk's attention is turned
# into all the statistics below in one go, summed per residue subset, and only the sums (plus one
# entropy per query row and head) are kept.
#
# For the last --n_layers saved layers and every subset of --subsets it writes to --out:
#   composition.csv / composition_<subset>.png  share of an attention row by kind of key, with the
#       query residue itself taken out and the row renormalized, and its log2 enrichment over the
#       composition of the chunk (what a row attending everywhere equally would get)
#   offsets.csv / offsets[_enrichment]_<subset>_L<layer>.png  the same by (offset from the query,
#       kind of key), the enrichment taken against the composition at each offset
#   locality.csv / locality_<subset>.png  share of a row within d residues of the query, the distances
#       d10 .. d90 holding 10 .. 90 % of the mass on residues (NaN past --max_distance), [CLS] / [EOS]
#       mass; locality_curves.csv holds the whole cumulative curve, d = 0 .. --max_distance
#   head_examples.{png,pdf}  the --examples heads at a readable size, with their attention summed per offset
#       and per key token; head_examples_locality.{png,pdf} the same heads with a column of the locality
#       curves of every head beside them, at print size for --print_width mm (default 200, B4 text width);
#       --from_tables redraws both
#       from an earlier run's tables (locality only up to --window if it predates locality_curves.csv)
#   entropy.csv / entropy.png  per head row entropy, normalized by log2(keys) of the chunk
#   entropy_tests.csv / entropy_<a>_vs_<b>.png  for every --compare pair, the pooled comparison (AUC =
#       P(normalized entropy of a b row > that of an a row), Mann-Whitney p) and the within protein
#       one (difference of the per protein means, Wilcoxon p), each BH adjusted over the layers and
#       heads of the pair
# In every table head MIXTURE (-1) is the heads of the layer averaged and BACKGROUND (-2) a row that
# attends to every key equally.
#
# A subset is ':' joined terms a query residue must all meet: 'all', 'site' / 'non-site' (info
# file), 'TP' / 'TN' / 'FP' / 'FN' (the --run's stitched predictions of --fold, which should be the
# fold the attention was dumped from), or residue letters ('S', 'ST'). E.g. 'S:site', 'T:TP'.
#
#   cd model
#   python attention_analysis.py ../data/embeddings/finetuning_drop_S_2_650M_dbptm_chunked_max \
#       --run logs/finetuning_drop_S_2_650M_dbptm_chunked_max --fold 0 \
#       --splits ../data/dbptm/splits_S_chunked_max.json --partition test \
#       --n_layers 3 --subsets all site non-site TP TN --compare non-site/site TN/TP

import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.special import entr, expit
from scipy.stats import false_discovery_control, mannwhitneyu, wilcoxon

CLS, EOS = '[CLS]', '[EOS]'
RESIDUES = list('ACDEFGHIKLMNPQRSTVWY')
TOKENS = [CLS] + RESIDUES + ['X', EOS]           # X collects B, O, U, Z and anything else rare
TOKEN_ID = {token : i for i, token in enumerate(TOKENS)}
# Head labels, which double as indices into the head axis, where the two sit last
MIXTURE, BACKGROUND = -1, -2
OUTCOMES = ('TP', 'TN', 'FP', 'FN')
WITHIN = (0, 1, 2, 5, 10, 25)                    # Distances locality.csv reports the share within
MAGMA = plt.get_cmap('magma').with_extremes(bad='black')

# -- Inputs --------------------------------------------------------------------------------------

def load_info(info_path, splits_path=None, fold=0, partition='test'):
    """
    Known sites of every protein of the info file, {parent id : 0 based whole protein positions},
    and the parent ids of one partition of one fold of a splits file (None without one). Sites are
    1 based, and chunk local in a chunked info file.
    """
    info = pd.read_json(info_path)
    parents = info['parent_id'] if 'parent_id' in info else info['id']
    offsets = info['offset'] if 'offset' in info else pd.Series(0, index=info.index)
    sites = {}
    for parent, offset, row_sites in zip(parents, offsets, info['sites']):
        sites.setdefault(parent, set()).update(int(site) - 1 + int(offset)
                                               for site in (row_sites if isinstance(row_sites, list) else ()))

    if splits_path is None:
        return sites, None

    with open(splits_path) as f:
        split = json.load(f)[fold]
    if partition not in split:
        raise SystemExit(f'Fold {fold} of {splits_path} has no {partition} partition')
    return sites, set(parents.loc[split[partition]])

def load_outcomes(run, fold, preds, threshold):
    """
    {protein : {position : 'TP' / 'TN' / 'FP' / 'FN'}} from the stitched predictions of one fold of
    a run. threshold 'opt' takes the fold's tuned threshold from its metadata.json.
    """
    if threshold == 'opt':
        with open(os.path.join(run, 'metadata.json')) as f:
            threshold = json.load(f)['data']['test_metrics'][fold].get('opt_threshold')
        if threshold is None:
            raise SystemExit(f'{run}/metadata.json has no opt_threshold for fold {fold}, run threshold_metrics.py')

    table = pd.read_json(os.path.join(run, f'fold_{fold}', f'{preds}_fold_{fold}_stitched.json'))
    outcomes = {}
    for protein, logits, labels, positions in zip(table['id'], table['logits'], table['labels'],
                                                  table['sequence_indices']):
        called = expit(np.asarray(logits, dtype=float)) >= float(threshold)
        labels = np.asarray(labels) == 1
        cells = np.where(labels, np.where(called, 'TP', 'FN'), np.where(called, 'FP', 'TN'))
        outcomes[protein] = dict(zip(np.asarray(positions, dtype=int).tolist(), cells))

    print(f'Outcomes of {len(outcomes)} proteins from fold {fold} at threshold {float(threshold):.3f}')
    return outcomes

def subset_mask(spec, residue, site, outcome):
    """The query rows a subset spec selects, see the header for the terms."""
    mask = np.ones(len(residue), dtype=bool)
    for term in spec.split(':'):
        if term in ('site', 'non-site'):
            mask &= site == (term == 'site')
        elif term in OUTCOMES:
            mask &= outcome == term
        elif term and set(term) <= set(RESIDUES):
            mask &= np.isin(residue, list(term))
        elif term != 'all':
            raise SystemExit(f'Unknown term {term!r} in subset {spec!r}')
    return mask

# -- The single pass -----------------------------------------------------------------------------

def per_head(sums, background, count):
    """(subsets, layers, heads, ...) sums to means, with BACKGROUND and MIXTURE appended to the heads."""
    mixture = sums.mean(axis=2, keepdims=True)
    background = np.broadcast_to(background[:, None, None], mixture.shape)
    stacked = np.concatenate([sums, background, mixture], axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return stacked / count.reshape((-1,) + (1,) * (stacked.ndim - 1))

class Accumulator:
    """
    Sums over the query rows of every chunk, per subset, so that chunks add up and a folder is read
    once. "background" holds what a row attending to every key equally would contribute instead.
    """
    def __init__(self, n_subsets, window, max_distance):
        self.window, self.max_distance = window, max_distance
        self.count = np.zeros(n_subsets)
        self.sums, self.background, self.entropy = None, None, []

    def allocate(self, n_layers, n_heads):
        S, T, O, M = len(self.count), len(TOKENS), 2 * self.window + 1, self.max_distance + 1
        inner = {'composition' : (T,), 'offsets' : (O, T), 'distance' : (M,), 'special' : (2,)}
        self.sums = {name : np.zeros((S, n_layers, n_heads) + shape) for name, shape in inner.items()}
        self.background = {name : np.zeros((S,) + shape) for name, shape in inner.items()}

    def add(self, weights, sequence, offset, queries, masks):
        """
        One chunk. "weights" is (layers, heads, queries, keys), [CLS] first and [EOS] last, rows
        summing to one; "masks" is (subsets, queries).
        """
        n_layers, n_heads, n_rows, n_keys = weights.shape
        if self.sums is None:
            self.allocate(n_layers, n_heads)

        sums, background, W = self.sums, self.background, self.window
        masks = masks.astype(np.float64)
        self.count += masks.sum(axis=1)
        rows = np.arange(n_rows)
        onehot = np.eye(len(TOKENS), dtype=np.float32)[[TOKEN_ID[CLS]] + [TOKEN_ID.get(res, TOKEN_ID['X'])
                                                        for res in sequence] + [TOKEN_ID[EOS]]]
        self_column = queries - offset + 1
        query_token = onehot[self_column]

        # Entropy of every row in bits, the heads and then their mixture: (queries, layers, heads + 1)
        bits = np.concatenate([entr(weights).sum(-1), entr(weights.mean(axis=1)).sum(-1)[:, None]], axis=1)
        self.entropy.append((bits / np.log(2)).transpose(2, 0, 1))

        # Composition, the query's own column taken out of its row and the rest renormalized
        self_mass = weights[:, :, rows, self_column]
        by_token = weights @ onehot - self_mass[..., None] * query_token
        by_token /= np.maximum(1 - self_mass, 1e-12)[..., None]
        sums['composition'] += np.einsum('sq,lhqt->slht', masks, by_token)
        background['composition'] += masks @ ((onehot.sum(axis=0) - query_token) / (n_keys - 1))

        # The residue keys within "window" of the query, by offset; the special columns and the
        # offsets past either end of the chunk are masked out
        columns = self_column[:, None] + np.arange(-W, W + 1)
        valid = (columns >= 1) & (columns <= n_keys - 2)
        columns = np.clip(columns, 0, n_keys - 1)
        near = weights[:, :, rows[:, None], columns] * valid
        keys = onehot[columns] * valid[..., None]
        sums['offsets'] += np.einsum('sq,lhqo,qot->slhot', masks, near, keys, optimize=True)
        background['offsets'] += np.einsum('sq,qot->sot', masks, keys) / n_keys

        # Cumulative share within d residues, d = 0 .. max_distance, as the difference of a running
        # sum over the residue columns at the two ends of [query - d, query + d], clipped to the chunk
        distance = np.arange(self.max_distance + 1)
        start = np.clip(self_column[:, None] - 1 - distance, 0, n_keys - 2)
        stop = np.clip(self_column[:, None] + distance, 0, n_keys - 2)
        running = np.concatenate([np.zeros(weights.shape[:-1] + (1,), dtype=np.float32),
                                  np.cumsum(weights[..., 1:-1], axis=-1)], axis=-1)
        within = running[:, :, rows[:, None], stop] - running[:, :, rows[:, None], start]
        sums['distance'] += np.einsum('sq,lhqd->slhd', masks, within, optimize=True)
        background['distance'] += masks @ ((stop - start) / n_keys)

        sums['special'] += np.einsum('sq,lhqc->slhc', masks, weights[..., [0, -1]])
        background['special'] += masks.sum(axis=1, keepdims=True) / n_keys

def read_folder(paths, n_layers, subsets, sites, outcomes, window, max_distance):
    """
    Reads every dump file once. Returns the accumulator, the saved layers analysed, and a table of
    the query rows (protein, position, residue, site, outcome) in the order of its entropies.
    """
    acc = Accumulator(len(subsets), window, max_distance)
    layers, meta = None, []
    for i, path in enumerate(paths):
        with np.load(path) as embed:
            if 'attention_layers' not in embed:
                continue

            saved = embed['attention_layers'].tolist()
            if layers is None:
                if n_layers > len(saved):
                    raise SystemExit(f'--n_layers {n_layers}, but the files hold {len(saved)} layers {saved}')
                layers = sorted(saved)[-n_layers:]
            axis = [saved.index(layer) for layer in layers]

            protein, sequence = str(embed['id']), str(embed['sequence'])
            known, predicted = sites.get(protein), outcomes.get(protein, {})
            for chunk, offset in enumerate(embed['chunk_offsets'].tolist()):
                queries = embed[f'attention_positions_{chunk}']
                if not len(queries):
                    continue

                residue_part = embed[f'attention_{chunk}'][axis].astype(np.float32)
                special = embed[f'attention_special_{chunk}'][axis].astype(np.float32)
                weights = np.concatenate([special[..., :1], residue_part, special[..., 1:]], axis=-1)
                # float16 storage leaves the rows a few 1e-4 off one
                weights /= weights.sum(axis=-1, keepdims=True)

                rows = pd.DataFrame({
                    'protein' : protein,
                    'position' : queries,
                    'residue' : [sequence[position] for position in queries],
                    'site' : [-1 if known is None else int(position in known) for position in queries],
                    'outcome' : [predicted.get(position, '') for position in queries.tolist()],
                })
                masks = np.stack([subset_mask(spec, rows['residue'].to_numpy(), rows['site'].to_numpy(),
                                              rows['outcome'].to_numpy()) for spec in subsets])
                acc.add(weights, sequence[offset:offset + residue_part.shape[-1]], offset, queries, masks)
                meta.append(rows.assign(n_keys=weights.shape[-1]))

        print(f'{i + 1}/{len(paths)} files', end='\r', flush=True)

    if layers is None:
        raise SystemExit('None of the files holds attention, run compute_embeddings.py --attention_layers')
    print()
    return acc, layers, pd.concat(meta, ignore_index=True)

# -- Tables --------------------------------------------------------------------------------------

def long_table(values, **axes):
    """Arrays shaped by the axes, given in order as name = labels, as one long table."""
    index = pd.MultiIndex.from_product(list(axes.values()), names=list(axes))
    return pd.DataFrame({name : array.ravel() for name, array in values.items()}, index=index).reset_index()

def adjust(p_values):
    """Benjamini-Hochberg over the tests that ran, the NaN ones left out of the family."""
    adjusted = np.full(len(p_values), np.nan)
    tested = np.isfinite(p_values)
    if tested.any():
        adjusted[tested] = false_discovery_control(p_values[tested])
    return adjusted

def compare(values, proteins, mask_a, mask_b, min_rows):
    """
    Two groups of query rows compared column by column, values being (rows, tests): pooled, and
    within protein over the proteins with at least "min_rows" rows in each group.
    """
    a, b = values[mask_a], values[mask_b]
    result = {'n_a' : len(a), 'n_b' : len(b), 'mean_a' : a.mean(axis=0), 'mean_b' : b.mean(axis=0)}
    result['diff'] = result['mean_b'] - result['mean_a']
    result['auc'] = result['p_value'] = np.full(values.shape[1], np.nan)
    if len(a) > 1 and len(b) > 1:
        test = mannwhitneyu(b, a, axis=0)
        result['auc'], result['p_value'] = test.statistic / (len(a) * len(b)), test.pvalue
    result['q_value'] = adjust(result['p_value'])

    means = []
    for group, mask in ((a, mask_a), (b, mask_b)):
        grouped = pd.DataFrame(group).groupby(proteins[mask])
        means.append(grouped.mean()[grouped.size() >= min_rows])
    mean_a, mean_b = means[0].align(means[1], join='inner', axis=0)
    differences = (mean_b - mean_a).to_numpy()

    result['n_proteins'] = len(differences)
    result['paired_diff'] = differences.mean(axis=0) if len(differences) else np.full(values.shape[1], np.nan)
    result['same_sign'] = np.maximum((differences > 0).sum(axis=0), (differences < 0).sum(axis=0))
    result['paired_p'] = (wilcoxon(differences, axis=0).pvalue if len(differences) > 5
                          else np.full(values.shape[1], np.nan))
    result['paired_q'] = adjust(result['paired_p'])
    return result

def build_tables(acc, layers, rows, subsets, comparisons, min_rows):
    """Every output table, plus the arrays the figures are drawn from."""
    n_heads = acc.sums['special'].shape[2]
    heads = list(range(n_heads)) + [BACKGROUND, MIXTURE]
    means = {name : per_head(acc.sums[name], acc.background[name], acc.count) for name in acc.sums}
    axes = {'subset' : subsets, 'layer' : layers, 'head' : heads}
    out = {'means' : means}

    with np.errstate(invalid='ignore', divide='ignore'):
        composition = means['composition']
        out['composition_enrichment'] = np.log2(composition / composition[:, :, [BACKGROUND]])
        share = means['offsets'] / means['offsets'].sum(axis=-1, keepdims=True)
        out['offsets_enrichment'] = np.log2(share / share[:, :, [BACKGROUND]])

    offsets = list(range(-acc.window, acc.window + 1))
    out['composition'] = long_table({'mass' : composition, 'enrichment' : out['composition_enrichment']},
                                    **axes, token=TOKENS)
    out['offsets'] = long_table({'mass' : means['offsets'], 'enrichment' : out['offsets_enrichment']},
                                **axes, offset=offsets, token=TOKENS)

    curves = out['curves'] = means['distance']
    residue_mass = 1 - means['special'].sum(axis=-1)
    locality = {f'within_{d}' : curves[..., d] for d in WITHIN if d <= acc.max_distance}
    for share in (0.1, 0.25, 0.5, 0.75, 0.9):
        reached = curves >= share * residue_mass[..., None]
        locality[f'd{int(share * 100)}'] = np.where(reached.any(axis=-1), reached.argmax(axis=-1), np.nan)
    locality.update(cls_mass=means['special'][..., 0], eos_mass=means['special'][..., 1],
                    far_mass=residue_mass - curves[..., -1])
    out['locality'] = long_table(locality, **axes)
    out['locality_curves'] = long_table({'within' : curves}, **axes, distance=list(range(acc.max_distance + 1)))

    # Entropy, one row per query: (rows, layers, heads + mixture), normalized by log2(keys)
    bits = np.concatenate(acc.entropy)
    norm = bits / np.log2(rows['n_keys'].to_numpy())[:, None, None]
    masks = {spec : subset_mask(spec, rows['residue'].to_numpy(), rows['site'].to_numpy(),
                                rows['outcome'].to_numpy()) for spec in subsets}
    entropy_axes = {'layer' : layers, 'head' : list(range(n_heads)) + [MIXTURE]}
    with np.errstate(invalid='ignore'):
        out['entropy_grid'] = np.stack([norm[mask].mean(axis=0) for mask in masks.values()])
        out['entropy'] = pd.concat([long_table({
            'n_rows' : np.full(norm.shape[1:], mask.sum()),
            'n_proteins' : np.full(norm.shape[1:], rows.loc[mask, 'protein'].nunique()),
            'norm_entropy' : norm[mask].mean(axis=0), 'norm_entropy_std' : norm[mask].std(axis=0),
            'entropy_bits' : bits[mask].mean(axis=0), 'eff_keys' : 2 ** bits[mask].mean(axis=0),
        }, subset=[spec], **entropy_axes) for spec, mask in masks.items()], ignore_index=True)

    tests, out['norm'], out['masks'] = [], norm, masks
    proteins = rows['protein'].to_numpy()
    for a, b in comparisons:
        result = compare(norm.reshape(len(norm), -1), proteins, masks[a], masks[b], min_rows)
        shape = norm.shape[1:]
        tests.append(long_table({key : np.broadcast_to(value, (np.prod(shape),)).reshape(shape)
                                 for key, value in result.items()}, comparison=[f'{a}/{b}'], **entropy_axes))
    out['entropy_tests'] = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    return out

# -- Figures -------------------------------------------------------------------------------------

def head_name(head):
    return {MIXTURE : 'mix', BACKGROUND : 'uniform'}.get(head, str(head))

def heat(ax, grid, xlabels, ylabels, title, xstep=1, **scale):
    image = ax.imshow(np.ma.masked_invalid(grid), aspect='auto', interpolation='nearest', **scale)
    ax.set_xticks(range(0, len(xlabels), xstep), [str(x) for x in xlabels[::xstep]], fontsize=6)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=5 if len(ylabels) > 8 else 8)
    ax.set_title(title, fontsize=9)
    return image

def save(fig, path):
    fig.savefig(path, dpi=150)          # Every figure is laid out by constrained layout already
    plt.close(fig)

def plot_offsets(grids, heads, offsets, title, path, enrichment, extent=3.0, n_cols=7):
    """
    One (key token, offset) map per head and the mixture, on one shared scale: log scaled share of a
    row, or log2 enrichment over the composition at each offset. A motif reads as a single cell.
    """
    n_rows = -(-len(heads) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows), layout='constrained')
    vmax = np.nanmax(grids)
    scale = ({'cmap' : 'coolwarm', 'vmin' : -extent, 'vmax' : extent} if enrichment
             else {'cmap' : MAGMA, 'norm' : LogNorm(vmin=vmax / 1000, vmax=vmax)})
    for ax, head, grid in zip(axes.flat, heads, grids):
        image = heat(ax, grid.T, offsets, TOKENS, head_name(head), xstep=4, **scale)
        ax.axvline(offsets.index(0), color='cyan', lw=0.6, ls='--', alpha=0.6)
    for ax in axes.flat[len(heads):]:
        ax.axis('off')
    fig.colorbar(image, ax=axes, shrink=0.6, label='log2 enrichment' if enrichment else 'share of a row')
    fig.suptitle(title)
    save(fig, path)

def plot_composition(enrichment, heads, layers, title, path, extent=2.0):
    """log2 of the share of a row by kind of key over the composition share, one panel per layer."""
    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 0.25 * len(heads) + 2),
                             layout='constrained', squeeze=False)
    for ax, layer, grid in zip(axes.flat, layers, enrichment):
        image = heat(ax, grid, TOKENS, [head_name(head) for head in heads], f'layer {layer}',
                     cmap='coolwarm', vmin=-extent, vmax=extent)
    fig.colorbar(image, ax=axes, shrink=0.8, label='log2 enrichment')
    fig.suptitle(title)
    save(fig, path)

def plot_locality(curves, n_heads, layers, title, path, marks=(5, 25)):
    """Share of a row within d residues of the query, per head, and the share within "marks" as bars."""
    fig, axes = plt.subplots(len(layers), 2, figsize=(14, 3.6 * len(layers)), width_ratios=[1.3, 1],
                             layout='constrained', squeeze=False)
    distances, heads = np.arange(curves.shape[-1]), np.arange(n_heads)
    marks = [mark for mark in marks if mark < len(distances)]
    for (left, right), layer, curve in zip(axes, layers, curves):
        left.plot(distances, curve[:n_heads].T, lw=1, alpha=0.8, color='0.6')
        left.plot(distances, curve[MIXTURE], lw=2.2, color='C0', label='mixture')
        left.plot(distances, curve[BACKGROUND], lw=1.6, ls=':', color='black', label='uniform row')
        left.set_xscale('symlog', linthresh=1)       # Linear up to 1, logarithmic beyond
        left.xaxis.set_major_formatter(ScalarFormatter())
        for mark in marks:
            left.axvline(mark, color='0.8', lw=0.8, ls='--', zorder=0)
        left.set_ylabel(f'layer {layer}\nshare of a row within')
        left.legend(fontsize=8, loc='lower right')
        width = 0.8 / len(marks)
        for i, mark in enumerate(marks):
            right.bar(heads + (i + 0.5) * width - 0.4, curve[:n_heads, mark], width=width, label=f'within {mark}')
            right.axhline(curve[BACKGROUND, mark], color=f'C{i}', ls=':', lw=1)
        right.set_xticks(heads)
        right.legend(fontsize=8)
    axes[-1, 0].set_xlabel('residues from the query')
    axes[-1, 1].set_xlabel('head (dotted: uniform row)')
    fig.suptitle(title)
    save(fig, path)

# Font sizes of the figures meant for a page: PRINT for a two-column page width, read at 100 %
PRINT = {'font.size' : 7, 'axes.titlesize' : 7.5, 'axes.labelsize' : 7, 'xtick.labelsize' : 6,
         'ytick.labelsize' : 6, 'legend.fontsize' : 6, 'figure.titlesize' : 8}
SCREEN = {'font.size' : 10, 'axes.titlesize' : 12, 'legend.fontsize' : 8}
EXAMPLE_COLORS = ['C1', 'C2', 'C4', 'C5', 'C6']

def save_both(fig, path):
    """The figure as .png and as .pdf next to it."""
    fig.savefig(os.path.splitext(path)[0] + '.pdf')
    save(fig, path)

def example_title(layer, head, label):
    return f'layer {layer}, head {head}' + (f': {label}' if label else '')

def example_grids(offsets, subset, layer, head):
    """(key token, offset) share of one head and of the uniform row, from the long offsets table."""
    rows = offsets[(offsets['subset'] == subset) & (offsets['layer'] == layer)]
    grids = [rows[rows['head'] == h].pivot(index='token', columns='offset', values='mass').reindex(TOKENS)
             for h in (head, BACKGROUND)]
    if grids[0].columns.empty:
        raise SystemExit(f'No layer {layer} head {head} in the offsets of subset {subset}')
    return grids

def draw_example(sub, grid, composition, title, title_color='black', token_size=10, offset_step=1,
                 share_label='average share', side_width=1.25):
    """
    One head in a subfigure: its (key token, offset) map, its attention summed per offset above, and
    per key token beside (the query's own column left out) with ticks at the average share, what the
    composition at those offsets would give a head spreading the same total evenly over the residues
    there. A head that reads a position is one tall bar above; one that reads a kind of residue has
    bars far off their ticks beside.
    """
    offset_labels = list(grid.columns)
    (top, corner), (main, side) = sub.subplots(2, 2, height_ratios=[1, 4], width_ratios=[4, side_width],
                                               sharex='col', sharey='row')
    corner.axis('off')

    vmax = np.nanmax(grid.to_numpy())
    image = main.imshow(np.ma.masked_invalid(grid.to_numpy()), aspect='auto', interpolation='nearest',
                        cmap=MAGMA, norm=LogNorm(vmin=vmax / 1000, vmax=vmax))
    main.axvline(offset_labels.index(0), color='cyan', lw=0.8, ls='--', alpha=0.7)
    main.set_xticks(range(0, len(offset_labels), offset_step), offset_labels[::offset_step])
    main.set_yticks(range(len(TOKENS)), TOKENS, fontsize=token_size)
    main.set_xlabel('offset from the query')
    sub.colorbar(image, ax=main, location='bottom', shrink=0.7, label='share of a row')

    top.bar(range(len(offset_labels)), grid.sum(axis=0), color='0.35')
    top.set_ylabel('per offset')
    top.set_title(title, color=title_color)

    others = grid.drop(columns=0).sum(axis=1)
    composition = composition.drop(columns=0)
    expected = others.sum() * composition.sum(axis=1) / composition.to_numpy().sum()
    side.barh(range(len(TOKENS)), others, color='0.35')
    side.plot(expected, range(len(TOKENS)), '|', color='C3', ms=token_size * 0.9, mew=1.5, label=share_label)
    side.xaxis.set_major_locator(MaxNLocator(2))
    side.set_xlabel('per key token\n(query left out)')
    # In the empty corner, off the data, and left out of the layout so that it cannot widen the column
    corner.legend(*side.get_legend_handles_labels(), loc='center', handlelength=0.8, handletextpad=0.4,
                  borderpad=0.3).set_in_layout(False)

def plot_head_examples(offsets, subset, examples, path):
    """The --examples heads side by side at screen size, see draw_example."""
    with plt.rc_context(SCREEN):
        fig = plt.figure(figsize=(7 * len(examples), 8.5), layout='constrained')
        for sub, (layer, head, label) in zip(np.atleast_1d(fig.subfigures(1, len(examples))), examples):
            draw_example(sub, *example_grids(offsets, subset, layer, head), example_title(layer, head, label))
        fig.suptitle(f'{subset}: example heads, attention by offset and key token within '
                     f'±{offsets["offset"].max()}')
        save_both(fig, path)

def table_curves(folder, offsets, subset):
    """
    Locality curves of one subset from the tables of an earlier run, as (layers, layers' curves):
    the curves are (layers, heads + BACKGROUND + MIXTURE, distances). Runs from before
    locality_curves.csv existed get the exact curve up to the offset window, rebuilt from offsets.csv.
    """
    path = os.path.join(folder, 'locality_curves.csv')
    if os.path.exists(path):
        table, column, value = pd.read_csv(path), 'distance', 'within'
    else:
        print(f'No {path}, the locality panels stop at the offset window; rerun for the whole curve')
        table, column, value = offsets, 'offset', 'mass'
    table = table[table['subset'] == subset]
    layers = sorted(table['layer'].unique())
    heads = sorted(head for head in table['head'].unique() if head >= 0) + [BACKGROUND, MIXTURE]
    grid = (table.pivot_table(index=['layer', 'head'], columns=column, values=value, aggfunc='sum')
            .reindex(pd.MultiIndex.from_product([layers, heads])).to_numpy())
    if column == 'offset':
        window = grid.shape[1] // 2
        folded = grid[:, window:].copy()
        folded[:, 1:] += grid[:, window - 1::-1]
        grid = folded.cumsum(axis=1)
    return layers, grid.reshape(len(layers), len(heads), -1)

def plot_examples_locality(offsets, curves, layers, subset, examples, path, width=200):
    """
    A figure spanning both columns of a page "width" mm wide (the text width, so that it is placed
    unscaled): the --examples heads side by side (see draw_example) and, in a column on their right,
    how much of a row lies within d residues of the query for every head, one panel per layer, the
    example heads in their title's color.
    """
    n_heads = curves.shape[1] - 2
    distances = np.arange(curves.shape[-1])
    with plt.rc_context(PRINT):
        fig = plt.figure(figsize=(width / 25.4, 4.0), layout='constrained')
        subs = fig.subfigures(1, len(examples) + 1, width_ratios=[2.65] * len(examples) + [1.7], wspace=0.02)
        for sub, (layer, head, label), color, letter in zip(subs[:-1], examples, EXAMPLE_COLORS, 'abcdefgh'):
            draw_example(sub, *example_grids(offsets, subset, layer, head),
                         f'({letter}) {example_title(layer, head, label)}', color, token_size=6.5, offset_step=4,
                         share_label='avg. share', side_width=1.6)

        column = subs[-1]
        axes = np.atleast_1d(column.subplots(len(layers), 1, sharex=True, sharey=True))
        legend = {}
        for ax, layer, curve in zip(axes, layers, curves):
            ax.plot(distances, curve[:n_heads].T, color='0.78', lw=0.6)[0].set_label('single heads')
            ax.plot(distances, curve[MIXTURE], color='C0', lw=1.5, label='head mean')
            ax.plot(distances, curve[BACKGROUND], color='black', ls=':', lw=1.1, label='uniform row')
            for (example_layer, head, _), color, letter in zip(examples, EXAMPLE_COLORS, 'abcdefgh'):
                if example_layer == layer:
                    ax.plot(distances, curve[head], color=color, lw=1.5, label=f'({letter}) head {head}')
            if distances[-1] > 30:
                ax.set_xscale('symlog', linthresh=1)
                ax.xaxis.set_major_formatter(ScalarFormatter())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(0, distances[-1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylabel(f'layer {layer}')
            for handle, label in zip(*ax.get_legend_handles_labels()):
                legend.setdefault(label, handle)
        axes[-1].set_xlabel('d, residues from the query')
        # The shared lines first, then the example heads in panel order
        order = [label for label in legend if not label.startswith('(')] + sorted(l for l in legend if l.startswith('('))
        column.legend([legend[label] for label in order], order, loc='outside lower center', ncol=2,
                      handlelength=1.3, columnspacing=0.8)
        column.suptitle(f'({"abcdefgh"[len(examples)]}) share of a row within d', fontsize=PRINT['axes.titlesize'])
        save_both(fig, path)

def plot_entropy(grids, subsets, layers, heads, path):
    """Mean normalized row entropy as a layer by head grid, one panel per subset."""
    fig, axes = plt.subplots(len(subsets), 1, figsize=(0.35 * len(heads) + 3, 1.1 * len(layers) * len(subsets) + 1),
                             layout='constrained', squeeze=False)
    for ax, subset, grid in zip(axes.flat, subsets, grids):
        image = heat(ax, grid, [head_name(head) for head in heads], [str(layer) for layer in layers],
                     subset, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(image, ax=axes, shrink=0.8, label='entropy / log2(keys)')
    save(fig, path)

def plot_comparison(norm, mask_a, mask_b, tests, names, layers, heads, path):
    """
    The mixture entropy of the two groups per layer (titles: AUC, q), and the AUC of every head as a
    layer by head grid, dots marking a within protein q < 0.05.
    """
    fig = plt.figure(figsize=(max(5 * len(layers), 0.4 * len(heads) + 3), 7.5), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.1])
    tests = tests.set_index(['layer', 'head'])
    for i, (ax, layer) in enumerate(zip(top.subplots(1, len(layers), squeeze=False).flat, layers)):
        values = [norm[mask, i, MIXTURE] for mask in (mask_a, mask_b)]
        edges = np.histogram_bin_edges(np.concatenate(values), bins=40)
        for j, (name, group) in enumerate(zip(names, values)):
            ax.hist(group, bins=edges, density=True, alpha=0.55, color=f'C{j}', label=f'{name} (n={len(group)})')
            ax.axvline(group.mean(), color=f'C{j}', ls='--')
        line = tests.loc[(layer, MIXTURE)]
        ax.set_title(f'layer {layer} mixture: AUC {line["auc"]:.3f}, q {line["q_value"]:.1e}', fontsize=9)
        ax.set_xlabel('entropy / log2(keys)')
        ax.legend(fontsize=7)

    ax = bottom.subplots()
    auc = tests['auc'].unstack().loc[layers, heads].to_numpy()
    extent = max(np.nanmax(np.abs(auc - 0.5)), 1e-3)
    image = heat(ax, auc, [head_name(head) for head in heads], [str(layer) for layer in layers],
                 f'P(entropy of a {names[1]} row > of a {names[0]} row), dots: paired q < 0.05',
                 cmap='coolwarm', vmin=0.5 - extent, vmax=0.5 + extent)
    flagged = np.argwhere(tests['paired_q'].unstack().loc[layers, heads].to_numpy() < 0.05)
    ax.plot(flagged[:, 1], flagged[:, 0], 'k.', ms=5)
    bottom.colorbar(image, ax=ax, label='AUC')
    save(fig, path)

def plot_all(out, layers, subsets, comparisons, window, folder):
    n_heads = out['means']['special'].shape[2] - 2
    heads = list(range(n_heads)) + [MIXTURE]
    offsets = list(range(-window, window + 1))
    for s, subset in enumerate(subsets):
        name = subset.replace(':', '_')
        for l, layer in enumerate(layers):
            for enrichment, grids in ((False, out['means']['offsets']), (True, out['offsets_enrichment'])):
                kind = 'enrichment' if enrichment else 'share'
                plot_offsets(grids[s, l, heads], heads, offsets,
                             f'{subset}, layer {layer}: attention by offset and key token ({kind})',
                             os.path.join(folder, f'offsets{"_enrichment" if enrichment else ""}_{name}_L{layer}.png'),
                             enrichment)
        plot_composition(out['composition_enrichment'][s][:, heads], heads, layers,
                         f'{subset}: log2 share of a row by key token over the composition, query residue taken out',
                         os.path.join(folder, f'composition_{name}.png'))
        plot_locality(out['curves'][s], n_heads, layers, f'{subset}: how local each head is',
                      os.path.join(folder, f'locality_{name}.png'))

    plot_entropy(out['entropy_grid'], subsets, layers, heads, os.path.join(folder, 'entropy.png'))
    for a, b in comparisons:
        if not (out['masks'][a].any() and out['masks'][b].any()):
            print(f'{a}/{b}: a group has no rows, no figure')
            continue
        tests = out['entropy_tests'][out['entropy_tests']['comparison'] == f'{a}/{b}']
        plot_comparison(out['norm'], out['masks'][a], out['masks'][b], tests, (a, b), layers, heads,
                        os.path.join(folder, f'entropy_{a.replace(":", "_")}_vs_{b.replace(":", "_")}.png'))

# -- Main ----------------------------------------------------------------------------------------

def main(args):
    if args.subsets is None:
        args.subsets = ['all', 'site', 'non-site'] + (list(OUTCOMES) if args.run else [])
    if args.compare is None:
        args.compare = [f'{a}/{b}' for a, b in (('non-site', 'site'), ('TN', 'TP'))
                        if a in args.subsets and b in args.subsets]
    examples = []
    for example in args.examples or []:
        parts = example.split(':', 2)
        if len(parts) < 2 or not all(part.lstrip('-').isdigit() for part in parts[:2]):
            raise SystemExit(f'--examples takes layer:head[:label], e.g. 32:4:residue-specific, not {example!r}')
        examples.append((int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else ''))
    example_subset = 'all' if 'all' in args.subsets else args.subsets[0]
    examples_path = os.path.join(args.out, 'head_examples.png')
    combined_path = os.path.join(args.out, 'head_examples_locality.png')
    if args.from_tables:
        if not examples:
            raise SystemExit('--from_tables only redraws the head examples figures, give --examples')
        offsets = pd.read_csv(os.path.join(args.out, 'offsets.csv'))
        plot_head_examples(offsets, example_subset, examples, examples_path)
        plot_examples_locality(offsets, *table_curves(args.out, offsets, example_subset)[::-1], example_subset,
                               examples, combined_path, args.print_width)
        print(f'Written to {examples_path} and {combined_path}')
        return

    comparisons = [tuple(pair.split('/')) for pair in args.compare]
    for pair in comparisons:
        if len(pair) != 2 or not set(pair) <= set(args.subsets):
            raise SystemExit(f'--compare {"/".join(pair)}: both sides must be among --subsets')
    for spec in args.subsets:
        subset_mask(spec, np.array([]), np.array([]), np.array([]))
    if any(term in OUTCOMES for spec in args.subsets for term in spec.split(':')) and not args.run:
        raise SystemExit('Outcome subsets need --run')

    sites, proteins = load_info(args.prot_info, args.splits, args.fold, args.partition)
    outcomes = load_outcomes(args.run, args.fold, args.preds, args.threshold) if args.run else {}
    paths = sorted(glob(os.path.join(args.folder, '*.npz')))
    if proteins is not None:
        paths = [path for path in paths if os.path.basename(path)[:-len('.npz')] in proteins]
        print(f'{len(paths)} files of the {len(proteins)} proteins of the {args.partition} partition of fold {args.fold}')
    if not paths:
        raise SystemExit(f'No dump files to read in {args.folder}')

    acc, layers, rows = read_folder(paths, args.n_layers, args.subsets, sites, outcomes, args.window,
                                    args.max_distance)
    out = build_tables(acc, layers, rows, args.subsets, comparisons, args.min_rows)

    os.makedirs(args.out, exist_ok=True)
    for name in ('composition', 'offsets', 'locality', 'locality_curves', 'entropy', 'entropy_tests'):
        out[name].to_csv(os.path.join(args.out, f'{name}.csv'), index=False, float_format='%.6g')
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args) | {'layers' : layers}, f, indent=2)
    if not args.no_plots:
        plot_all(out, layers, args.subsets, comparisons, args.window, args.out)
        if examples:
            plot_head_examples(out['offsets'], example_subset, examples, examples_path)
            plot_examples_locality(out['offsets'], out['curves'][args.subsets.index(example_subset)], layers,
                                   example_subset, examples, combined_path, args.print_width)

    print(f'Layers {layers}, {rows["protein"].nunique()} proteins, {len(rows)} query rows')
    for spec, mask in out['masks'].items():
        print(f'  {spec:>12}: {mask.sum():6d} rows of {rows.loc[mask, "protein"].nunique()} proteins')
    if len(out['entropy_tests']):
        columns = ['comparison', 'layer', 'n_a', 'n_b', 'mean_a', 'mean_b', 'auc', 'q_value', 'n_proteins',
                   'paired_diff', 'same_sign', 'paired_q']
        print('Mixture entropy:')
        print(out['entropy_tests'].loc[out['entropy_tests']['head'] == MIXTURE, columns].round(4).to_string(index=False))
    print(f'Written to {args.out}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attention statistics of a compute_embeddings.py --attention_layers dump.')
    parser.add_argument('folder', help='Folder of .npz files written by compute_embeddings.py --attention_layers.')
    parser.add_argument('--out', default=None, help='Output folder (default ../data/attention_analysis/<folder name>).')
    parser.add_argument('--n_layers', type=int, default=3, help='Analyse the last N of the saved layers (default 3).')
    parser.add_argument('--subsets', nargs='+', default=None,
                        help='Query residue subsets, ":" joined terms: all, site, non-site, TP, TN, FP, FN, or residue '
                             'letters (e.g. S, ST, S:site). Default all site non-site, plus the outcomes with --run.')
    parser.add_argument('--compare', nargs='*', default=None,
                        help='Subset pairs a/b whose entropies are tested against each other '
                             '(default non-site/site and TN/TP, those whose subsets are given).')
    parser.add_argument('--prot_info', default='../data/dbptm/dbptm_info_chunked.json',
                        help='Info file the sites (and the --splits labels) come from.')
    parser.add_argument('--splits', default=None, help='Restrict to the proteins of one partition of this splits file.')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold of --splits and of the --run predictions; the fold the attention was dumped from.')
    parser.add_argument('--partition', default='test', choices=['train', 'dev', 'test', 'total'])
    parser.add_argument('--run', default=None, help='Run folder whose stitched predictions give the TP/TN/FP/FN outcomes.')
    parser.add_argument('--preds', default='test_preds',
                        help='Prediction file prefix in fold_N, e.g. a *_whole_preds of evaluate_checkpoints.py.')
    parser.add_argument('--threshold', default='0.5', help='Probability cutoff of the outcomes, or "opt" for the fold\'s opt_threshold.')
    parser.add_argument('--window', type=int, default=12, help='Offsets either side of the query in the offset maps.')
    parser.add_argument('--max_distance', type=int, default=1022,
                        help='Longest distance of the locality curves (default 1022, a whole chunk, so every d10 .. d90 exists).')
    parser.add_argument('--examples', nargs='+', default=None,
                        help='layer:head[:label] heads drawn large in head_examples.png and, with the locality curves, '
                             'in head_examples_locality.png (subset all), e.g. 32:4:residue-specific 31:8:position-specific.')
    parser.add_argument('--from_tables', action='store_true',
                        help='Only redraw the --examples figures from the tables already in --out, reading no dumps.')
    parser.add_argument('--print_width', type=float, default=200,
                        help='Width of head_examples_locality in mm, the \\textwidth it is placed at unscaled '
                             '(default 200, B4 with 25 mm margins).')
    parser.add_argument('--min_rows', type=int, default=2,
                        help='Rows a protein needs in each group to enter the within protein test.')
    parser.add_argument('--no_plots', action='store_true', help='Only write the tables, the figures take most of the time.')
    args = parser.parse_args()
    args.out = args.out or os.path.join('../data/attention_analysis', os.path.basename(os.path.normpath(args.folder)))
    main(args)
