# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Prediction of protein phosphorylation sites using protein language models (ESM2 / ESM-C). The task is
per-token binary classification over an amino acid sequence: every S/T/Y residue is a candidate, and
the model predicts whether it is phosphorylated. Evaluation is 5-fold cross validation over
homology-reduced (MMseqs2 clustered) protein splits.

## Environment and commands

A virtualenv lives in `venv/` (gitignored). Use `venv/bin/python` directly, or activate it.

```bash
venv/bin/pip install -r requirements.txt
```

**There is no test suite** — no pytest, no test files. Changes are verified by running the training
or dataset scripts on real or small synthetic inputs.

**Working directory matters.** Every entry point uses relative default paths (`../data/...`) and
writes logs to `new_logs/` relative to the CWD (`model/new_logs` is gitignored). Run training from
`model/` and dataset scripts from `scripts/`.

### Training

Each model is its own entry point in `model/`, sharing the argument parser from `training.py`:

```bash
cd model
python ft_selective.py --epochs=30 --batch_size=4 --accum=8 --pos_weight=3 --lr=5e-5 \
    --prot_info_path ../data/dbptm/dbptm_info.json --dataset_path=../data/dbptm/splits_S.json \
    --residues="['S']" --indices="[-1,-2,-3]" -o "$OUT_DIRNAME"
```

- `--type` selects the backbone: `8M`, `35M`, `150M`, `650M` (default), `3B`, `15B` for ESM2, and
  `600M-C` for ESM-C. The mapping to HF repos is in `get_esm` (`model/utils.py`).
- `--fold N` trains a single fold instead of all five (useful for hyperparameter sweeps).
- `--frozen_epochs N` runs a frozen warmup phase at `--frozen_lr` before the main phase, checkpointed
  separately (`best_frozen.ckpt`, tracked by `frozen_finished` in the metadata).
- `--modify_prob` turns on input residue perturbation during training only (masking / random
  substitution / BLOSUM62 substitution, split by `--mask_prob` / `--rand_prob` / `--sub_prob`).
  Evaluation dataloaders always pass `modify_prob=0`.
- `--grad_checkpointing` and `--compile` trade compute for memory / speed; both are applied to
  `model.base` only and keep state dict keys unchanged, so checkpoints stay interchangeable.
- `--release` trains on all data with no test set — see the gotcha about it below.
- Resuming is automatic: relaunching with the same `-o` continues at the first fold that
  `metadata.json` does not mark finished, restores the training args from that metadata, and picks
  up `fold_N/chkpt.ckpt` if the fold has one. `--fold` still overrides the fold to train.
  Checkpoints also carry the Python/numpy/torch RNG state, so a resumed run shuffles and perturbs
  exactly as an uninterrupted one would have.
- `hyperparameters.md` records the argument sets used for published runs.
- Multiple visible GPUs automatically switch the Lightning strategy to `deepspeed_stage_2`.

Runnable trainers: `ft_selective.py` (partial finetuning, the main one), `lora_model.py` (peft LoRA),
`llrd_model.py` (full finetuning with layer-wise LR decay), `ft_multi_residue.py` (per-residue
heads), `ft_selective_lm.py`, `encoder.py`, `rnn.py`, `linear.py`, `kinase.py`, `baseline.py`
(frozen backbone + linear head), `uniptm.py`, `ft_recycling.py`, `recycling_encoder.py`, `dummy.py`.

### Dataset preparation

```bash
cd scripts
python chunk_proteins.py ../data/dbptm/dbptm_info.json          # -> dbptm_info_chunked.json
python dataset_creation.py --prot_info ../data/dbptm/dbptm_info_chunked.json \
    --clusters ../data/dbptm/dbptm_clusters.tsv --out_folder ../data/dbptm
```

`dataset_creation.py` writes, per residue set, `splits_<label>.json` plus a `_max` variant, along
with `dataset_info*.csv` and `site_increase.csv`.

### Analysis and export tooling

- `model/threshold_metrics.py` — post-hoc decision threshold tuning. The training loop reports every
  threshold-dependent metric at the 0.5 cutoff; this sweeps each fold's *validation* predictions for
  the MCC-maximising threshold, applies it to that fold's test set, and writes the results back into
  `metadata.json` under `*_opt` keys plus `opt_threshold`. Run as `cd model && python
  threshold_metrics.py` (defaults to scanning `logs`, not `new_logs`; `--dry-run` to preview).
- `model/combine_residue_models.py` — evaluates residue-specific runs (e.g. the S, T and Y runs of
  one configuration) as a single predictor by pooling their saved predictions per fold, and writes
  the result as a metrics-only run (`metadata.json` with `combined_from` in its args) under
  `logs/combined/<name with S+T / S+T+Y as the residue token>`, so the results GUI lists it beside
  real runs. `--auto` groups runs by name; `--auto --list` previews. Each component is scored on its
  own splits, so the pooled test set is not the test set of a jointly trained ST/STY run.
  `--test-split <name>` pools the `<name>_preds_fold_N*.json` files written by
  `evaluate_checkpoints.py` instead of the runs' own test predictions.
- `model/evaluate_checkpoints.py` — re-runs a finished run's `fold_N/best.ckpt` (rebuilt through its
  entry point, `--entry ft_selective` / `lora_model` / …) on the test set of another splits file,
  expanded to every chunk of the listed proteins, and writes `<name>_preds_fold_N(_stitched).json`
  plus `<name>_metrics_fold_N.json` into the fold dirs. `--metadata <other run>/metadata.json`
  replaces `--splits`, taking the splits file and the info file it indexes from that run's args
  (relative to `model/`). A fold without `val_preds_fold_N.json` also gets its own dev partition
  evaluated, writing the missing `val_preds_fold_N(_stitched).json` plus `val_metrics_fold_N.json`.
  It never writes `metadata.json`. It checks
  itself against the run's saved test predictions. Needs the checkpoints, which are not in `logs/`.
- `model/compute_embeddings.py` → `model/compute_umap.py` — dump last-hidden-layer residue
  embeddings (one `.npz` per protein) and project them, for comparing trained against pretrained
  representations. The full-width embeddings never fit in memory: by default UMAP is fitted on a
  `--fit_size` subsample and applied in batches; with `--pca_dim N` an `IncrementalPCA` is first
  fitted in batches over every residue, all residues are then loaded reduced to N dimensions, and
  UMAP works on those (`--fit_size 0` fits it on all of them). A 2D projection is also plotted to a
  `.png` next to the output `.npz` (by residue, and by known site when `--prot_info` is given).
  `--attention_layers "[0,16,-1]"` additionally writes the per-head attention of those
  layers, one `attention_<chunk>` array per chunk (rows are the `--residues` positions that chunk
  owns, columns every position of the chunk) plus the `[CLS]`/`[EOS]` columns in
  `attention_special_<chunk>`; the maps are quadratic in the sequence length, so keep `--batch_size`
  small. Both scripts take `--splits` / `--fold` / `--partition` to restrict them to one partition
  of one fold (`compute_umap.py` also needs `--prot_info` for it). `compute_embeddings.py` embeds
  every chunk of each listed protein, including the site-less chunks the splits leave out, and
  `compute_umap.py` matches the `<parent_id>.npz` file names. A missing partition (e.g. no `dev`)
  is an error; neither script carves dev out of train.
- `scripts/identity_significance_test.py` — Nadeau-Bengio corrected t-tests (Benjamini-Hochberg
  over the three pairs) between one run family's 30/50/70 % identity versions (`--run` template, as in
  `table_identity_versions.py`), recomputed from the stitched test predictions, written to
  `data/statistical_tests`. The versions use different splits, so `model/significance_test.py`'s
  paired tests do not apply and pairing folds by index is arbitrary. `scripts/plot_identity_significance.py`
  plots those results to `img/identity_significance.{png,pdf}`, marking significance by the BH p only.
  The paired alternative is a cross-evaluation: the 30/50/70 % clusterings nest (every 70 % cluster
  inside one 50 % and one 30 % cluster), and `scripts/identity_cross_splits.py` writes the higher
  versions' test sets over the lower versions' folds (`splits_<res>_<hi>_test_in_<lo>_folds.json`)
  and the 30 % test proteins none of the 50/70 % folds that would score them has a 30 % homolog of
  (`splits_<res>_chunked_max_leakfree*.json`, ~70 % of them, biased towards small families). After
  `evaluate_checkpoints.py` on those, `scripts/identity_cross_significance.py` runs the cell-based
  paired tests of `cross_representative_significance.py` with BH over each family of three pairs.
- `model/demo_evaluation.py` — per-protein demo evaluation of a run family (`--run
  'finetuning_drop_{res}_2_650M_dbptm_{variant}'`, over the S/T/Y runs and the 30/50/70 % versions)
  on a fixed list of well-known phosphoproteins (`DEMO`: EGFR, AKT1, SRC, ERK2, TP53, APP, tau, RB1,
  β-catenin, p27, EPOR, STAT5B, BAD) and their canonical sites, which are checked against the sequence.
  Each protein is scored by the fold that trained on neither it nor its cluster at the run's identity
  level: its own test fold, or the fold whose test set holds its cluster's representative. Most of these
  proteins are no representative at 30/50 %. Predictions come from saved files (`demo_preds_*`, then
  evaluate_checkpoints' `*_whole_preds_*`, then the partial `test_preds_*`); `--predict` runs the chosen
  `best.ckpt` for the rest, checking the restored model against a saved test protein first. Writes
  `data/demo_eval/<template>/{sites,proteins,canonical}.csv`; S and T runs are also pooled as `S+T`.
  The canonical site list was written from memory, not taken from a source: `model/demo_site_references.py`
  looks each site up in UniProt's Modified residue features (following "by similarity" evidence into
  the ortholog by alignment), caches the entries in `data/demo_eval/uniprot/`, and writes
  `data/demo_eval/canonical_references.csv` (merged into `canonical.csv`) and `references.csv`.
  Sites without any UniProt Modified residue annotation were dropped from the list (AKT1 Y315/Y326,
  TP53 S6/T81, β-catenin Y86/Y654); keep it that way when adding sites.
- `scripts/phosf3c_dataset.py` — exports a dataset into the FASTA-with-`#` format PhosF3C trains on.
- `scripts/plot_site_counts.py` — regenerates the positive-site-count figure in `img/`.
- `scripts/plot_architectures.py` — writes `img/finetuning_architectures.svg` (surgical finetuning,
  LLRD, LoRA, recycling encoder) as hand-laid-out SVG, no dependencies. Coordinates are absolute and
  the legend's label widths are hardcoded, so check an Inkscape render after editing.

### GUIs

`streamlit run gui/gui.py` (prediction demo), `streamlit run result_gui/results_gui.py`.

## Architecture

### Entry point / backbone split

`model/training.py` is the shared backbone: it owns the global `argparse` parser, the Lightning
wrapper, the CV loop, metrics, and metadata. Each model file contributes only a `create_model(args)`
factory plus any extra CLI arguments (added via its own `add_arguments(parser)` before
`parse_args()`), then calls `run_training(args, create_model)`. Adding a model means writing that
factory — not touching the training loop.

A model may also define `get_param_groups(lr, weight_decay, fix_decay=...)`; if present,
`configure_optimizers` uses it instead of building parameter groups itself. This is how
`llrd_model.py` gives each backbone layer its own learning rate.

### Model hierarchy

`TokenClassifier` (`model/token_classifier_base.py`) is the base for the finetuning models: a frozen
or partially unfrozen PLM in `self.base` plus a classification head in `self.classifier`.

- `forward()` returns `(logits, base_model_outputs)` — a tuple, not a HF output object.
- `compute_loss()` flattens logits, drops positions whose label is `ignore_index` (-1), and applies
  the configured loss.
- `freeze_phase()` / `unfreeze_phase()` are called by the training loop around the optional frozen
  warmup phase. Models that wrap or restructure `self.base` (e.g. LoRA, LLRD) **must override
  these**, as the inherited versions reach directly into `self.base.encoder.layer`.
- Layer lookup is backbone-dependent: ESM2 nests transformer layers under `encoder.layer`, ESM-C
  under `transformer.blocks`. Code that indexes layers must handle both.
- Config is a dataclass subclassing `TokenClassifierConfig`, saved alongside the state dict.

`BaseModel` / `LMModel` (`base_model.py`, `lm_model_base.py`) are a parallel hierarchy for the models
that also do masked language modelling.

### Data flow

Two files define a dataset, and they must be generated from each other:

1. **Info file** (`data/*/​*_info.json`) — a dataframe with `id`, `sequence`, `sites`. **Sites are
   1-based**; `load_prot_data` converts to 0-based and filters to S/T/Y.
2. **Splits file** (`data/*/splits_*.json`) — a list of one dict per fold with `train`/`dev`/`test`/
   `total`, each a list of **dataframe index labels into the info file**. Mismatched files silently
   select the wrong proteins.

`prepare_datasets` → `FullProteinDataset.get_fold(i)` → `ProteinDataset` → `prep_batch` collate.
Labels are per-token: `0` for candidate residues, `1` for known sites, `ignore_index` (-1) everywhere
else, including padding and the `[CLS]`/`[EOS]` tokens. Because non-candidates are ignored, the loss
and metrics only ever see S/T/Y positions.

If a fold has no `dev` key, `get_fold` carves one out of `train` with an 80/20
`train_test_split(random_state=42)` instead. Whether a run used a real dev partition or a derived
one is only visible in the training log output, not in the metadata.

`prep_batch` does **no truncation**; long sequences are handled at dataset creation (see below).

### Chunking of long proteins

ESM2 takes 1022 residues. `scripts/chunk_proteins.py` splits longer proteins into overlapping chunks
and adds `parent_id`, `chunk_index`, `offset`, `parent_length` columns. Chunks of a split protein are
suffixed `_0`, `_1`, …; proteins that already fit keep their id.

Invariants that the rest of the pipeline depends on:

- A site is kept in **every** chunk containing it. Dropping it from one would make that chunk label a
  real site as a negative.
- `parent_id` is an explicit column, never parsed from the id suffix (UniProt entry names contain
  underscores).
- `dataset_creation.py` clusters and splits by `parent_id`, so all chunks of a protein land in the
  same fold *and* the same train/dev partition. Splitting them would leak near-identical context.
- Predictions are stitched back by `stitch_chunk_predictions` (`model/data_loading.py`), which shifts
  positions by `offset` and, where chunks overlap, keeps the prediction from the chunk that saw the
  position closest to its center. `stitched_*` metrics count each position once; the plain
  `test_*` / `val_*` metrics double-count overlaps. Unchunked datasets go through the same path, with
  every protein as its own single chunk.

### Cluster representatives

`dataset_creation.py` keeps one protein per MMseqs2 cluster. Two strategies run on every invocation:
the representative assigned by the clustering (`splits_<label>.json`), and the cluster member with
the most sites of that residue set (`splits_<label>_max.json`). The `_max` variant also recovers
clusters whose assigned representative has no site of the residue set at all, which the default
strategy drops entirely. `site_increase.csv` quantifies the difference.

The clustering is connected-component (`--cluster-mode 1`) with no coverage requirement (`-c 0`), so
single shared domains chain proteins together. At 30 % this makes one cluster (`A6NDX5`) of 22,831 of
the 60,176 dbPTM proteins, holding most kinases and signalling proteins (EGFR, AKT1, ERK2, SRC, APP),
and it enters the splits as one representative, titin. At 50 % the largest cluster has 1,363 proteins
(mostly C2H2 zinc fingers); at 70 % there is no outlier (largest 87).

The two files share their clusters but **not their folds**: each is split by its own
`StratifiedKFold`, so only ~1/5 of fold N's test clusters coincide and most of the rest are in the
other file's fold-N train set. Never evaluate fold N of one variant on fold N of the other.
`scripts/cross_representative_splits.py` instead redistributes one variant's test proteins over the
other's folds by cluster, writing `splits_<res>_<test>_test_in_<folds>_folds.json` (e.g.
`splits_S_chunked_max_test_in_chunked_folds.json`) for `evaluate_checkpoints.py --splits`.
`scripts/cross_representative_significance.py` then runs `model/significance_test.py` on such a
pair, regrouping both runs' predictions into the 25 (own fold, other fold) cells so each metric
still comes from one model per run; its corrected t-test column does not apply to cells.

### Outputs of a run

Under `model/new_logs/<name>/`:

- `metadata.json` — args, `fold_finished` / `frozen_finished` (the real resume state; `current_fold`
  is derived from them), per-fold `val_metrics` and `test_metrics`, and `*_metric_avg` averages.
  Validation metrics are computed after loading the best checkpoint and are the right thing to rank
  hyperparameter sweeps by; test metrics should stay untouched.
- `fold_N/best.ckpt`, `chkpt.ckpt`, TensorBoard logs. Both the `ModelCheckpoint` monitor and
  `EarlyStopping` track **`val_mcc`** (maximise) — `best.ckpt` is the best-MCC epoch, not the last.
- `fold_N/{test,val}_preds_fold_N.json` — per-chunk predictions. **`df_index` is a positional index
  into that fold's evaluated subset, not a label into the info file.**
- `fold_N/{test,val}_preds_fold_N_stitched.json` — per-protein predictions with whole-protein
  positions.

`model/logs/` is the (untracked, not gitignored) archive of finished runs — ~720 of them — and is
what the results GUI and `threshold_metrics.py` read by default. `model/new_logs/` is where a run
actually writes. Run directories there are named by convention
`<prefix>_<residues>_<type>_<dataset><variant>` (e.g. `baseline_ST_650M_dbptm_chunked_max`), with a
`pw<N>` fragment for positive-weight sweeps; `scripts/batch_training.sh` documents how the name is
assembled.

## Gotchas
- **The residue-set filter of dataset creation drops chunks, not just proteins.** `filter_dataset`
  (`scripts/dataset_creation.py`) keeps rows with a site of the residue set, and on chunked data a
  row is a chunk. So every split (train/dev/test, residue-specific and joint alike) lacks proteins
  without such a site *and* the site-less chunks of long proteins that have sites elsewhere, about
  25–45 test chunks per dbPTM fold. The missing positions are all negatives. "Per whole protein"
  stitched metrics therefore cover the listed chunks only; `evaluate_checkpoints.py` scores whole
  proteins by expanding the test list to all chunks of its proteins.
- `model/utils.py` imports the `esm` package purely to register the `esmc` architecture with
  transformers' Auto classes — the unused import is load-bearing.
- **Metadata across `model/logs` is heterogeneous.** Runs predate various features, so only some
  carry `stitched_*` metrics, `*_opt` / `opt_threshold` keys, or `fold_finished`; unfinished folds
  are stored as empty dicts. Of ~720 archived runs only about half have validation metrics at all.
  Anything that aggregates or plots over `logs/` must tolerate missing keys and skip empty folds
  rather than assume a fixed schema.
- **`--indices` and `--unfreeze_indices` are different flags.** The finetuning entry points
  (`ft_selective.py`, `ft_multi_residue.py`, `ft_recycling.py`, `ft_selective_lm.py`) read
  `--indices`, which each adds itself. `--unfreeze_indices` on the global parser is legacy and is not
  read by those models; setting it has no effect on which layers unfreeze.
- **`--release` is not a maintained path.** `prepare_full_dataset` uses fold 0's `train` + `test`
  (not every fold), and `run_release_training` calls `train_release_model` with no dev dataloader
  while its `ModelCheckpoint` and `EarlyStopping` monitor `val_f1` — a metric nothing will log.
  Verify this end to end before relying on it.
- `scripts/*.sh` are PBS job scripts for the MetaCentrum cluster with hardcoded
  `/storage/brno12-cerit/...` paths. They are reference for argument sets and the run-naming
  convention, not scripts to run locally.
- `gui/` will be updated in the future, ignore the current implementation.
