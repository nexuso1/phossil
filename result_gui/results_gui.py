import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px

# Set page configuration
st.set_page_config(layout="wide", page_title="ML Experiment Comparator")

# Internal column prefixes. Metric names are namespaced by the metadata block they come from
# (val_metrics / test_metrics) so that the two groups can never collide.
VAL_PREFIX = "val::"
TEST_PREFIX = "test::"  
ARG_PREFIX = "arg::"

# Args that get their own dropdown above the table, in this order. Missing ones are skipped.
PRIMARY_FILTER_ARGS = ["prot_info_path", "residues", "dataset_path", "type", "kinase"]

NO_VALUE = "(none)"

# --- Helper Functions ---


def _arg_to_option(value):
    """Turns an arbitrary arg value into something hashable and printable for a dropdown."""
    if value is None:
        return NO_VALUE
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def _filled_folds(folds):
    """Folds that actually hold metrics. Unfinished folds are saved as empty dicts."""
    if not isinstance(folds, list):
        return []
    return [f for f in folds if isinstance(f, dict) and len(f) > 0]


def _average_folds(folds):
    """Mean of every numeric metric over the *finished* folds only."""
    filled = _filled_folds(folds)
    sums, counts = {}, {}
    for fold in filled:
        for k, v in fold.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1

    return {k: sums[k] / counts[k] for k in sums}


@st.cache_data
def load_experiments(root_dir):
    """
    Recursively finds all metadata.json files in the root_dir and flattens them into a DataFrame.

    Older metadata files may lack `val_metrics` entirely, and runs that are still in progress
    (or were interrupted) have empty per-fold dicts, so everything here is best-effort.
    """
    search_path = os.path.join(root_dir, "**", "metadata.json")
    files = glob.glob(search_path, recursive=True)

    experiments = []

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)

            data_block = content.get("data", {})
            if not isinstance(data_block, dict) or "args" not in data_block:
                # Not one of our experiment metadata files (e.g. a package metadata.json)
                continue

            folder_path = os.path.dirname(file_path)
            args = data_block.get("args", {}) or {}

            test_folds = data_block.get("test_metrics", []) or []
            val_folds = data_block.get("val_metrics", []) or []

            test_avg = _average_folds(test_folds)
            val_avg = _average_folds(val_folds)

            # Fall back to the averages stored by the training script if the per-fold metrics
            # are missing (some older runs only kept the averages).
            if not test_avg:
                test_avg = {k: v for k, v in (data_block.get("test_metric_avg", {}) or {}).items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if not val_avg:
                val_avg = {k: v for k, v in (data_block.get("val_metric_avg", {}) or {}).items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}

            folds_done = max(len(_filled_folds(test_folds)), len(_filled_folds(val_folds)))
            folds_total = max(len(test_folds), len(val_folds))

            row = {
                "Experiment Name": os.path.basename(folder_path),
                "Folds Done": folds_done,
                "Folds Total": folds_total,
                "Full Path": folder_path,
                "raw_metadata": content,
                "raw_test_folds": test_folds,
                "raw_val_folds": val_folds,
            }

            for k, v in val_avg.items():
                row[f"{VAL_PREFIX}{k}"] = v
            for k, v in test_avg.items():
                row[f"{TEST_PREFIX}{k}"] = v
            for k, v in args.items():
                row[f"{ARG_PREFIX}{k}"] = _arg_to_option(v)

            experiments.append(row)

        except Exception as e:
            st.warning(f"Skipping corrupt or malformed file: {file_path}. Error: {e}")

    return pd.DataFrame(experiments)


def metric_label(col):
    """Strips the internal namespace prefix for display."""
    return col.split("::", 1)[1] if "::" in col else col


def folds_to_frame(folds):
    """Per-fold metrics as a table, one row per finished fold plus a mean row."""
    rows, index = [], []
    for i, fold in enumerate(folds):
        if isinstance(fold, dict) and fold:
            rows.append(fold)
            index.append(f"Fold {i}")

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows, index=index)
    numeric = frame.select_dtypes("number")
    if not numeric.empty:
        frame.loc["mean"] = numeric.mean()

    return frame


def style_metrics(frame, metric_cols):
    """Colour-codes the metric columns, skipping ones that are entirely empty."""
    gradient_cols = [c for c in metric_cols
                     if pd.api.types.is_numeric_dtype(frame[c]) and frame[c].notna().any()]
    styler = frame.style
    for col in gradient_cols:
        # Column-by-column so that a single all-NaN column cannot break the whole table.
        styler = styler.background_gradient(subset=[col], cmap="viridis")
    return styler


# --- Main Layout ---

st.title("ML Model Comparison Dashboard")

# 1. Sidebar: Configuration
with st.sidebar:
    st.header("Data Source")
    default_dir = "../model/new_logs"
    root_dir = st.text_input("Experiments Root Directory", value=default_dir)

    if st.button("Reload Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.write("point this to the parent folder containing your experiment subfolders.")

# 2. Data Loading
if os.path.isdir(root_dir):
    df = load_experiments(root_dir)
else:
    st.error(f"Directory not found: {root_dir}")
    st.stop()

if df.empty:
    st.info("No 'metadata.json' files found in the specified directory.")
    st.stop()

all_cols = df.columns.tolist()
val_metric_cols = sorted(c for c in all_cols if c.startswith(VAL_PREFIX))
test_metric_cols = sorted(c for c in all_cols if c.startswith(TEST_PREFIX))
arg_cols = sorted(c for c in all_cols if c.startswith(ARG_PREFIX))

# --- Tab View ---
tab1, tab2 = st.tabs(["Comparison Table", "Fold Variance"])

# TAB 1: Main Comparison Table
with tab1:
    st.subheader("Leaderboard")

    # --- Filters on the metadata ---
    st.markdown("**Filters**")

    filter_args = [f"{ARG_PREFIX}{a}" for a in PRIMARY_FILTER_ARGS if f"{ARG_PREFIX}{a}" in arg_cols]
    filters = {}

    if filter_args:
        filter_columns = st.columns(len(filter_args))
        for column, col_name in zip(filter_columns, filter_args):
            options = sorted(df[col_name].dropna().unique().tolist())
            with column:
                filters[col_name] = st.multiselect(
                    metric_label(col_name).replace("_", " ").title(),
                    options,
                    default=[],
                    # Paths are long, show the file name but keep filtering on the full value.
                    format_func=lambda v: os.path.basename(v) if "/" in str(v) else v,
                    placeholder="All",
                    help=f"Filter on args.{metric_label(col_name)}",
                )

    with st.expander("More filters"):
        name_query = st.text_input("Experiment name contains", value="")

        col_a, col_b = st.columns(2)
        with col_a:
            max_folds = int(df["Folds Total"].max())
            min_folds = st.number_input(
                "Minimum finished folds", min_value=0, max_value=max_folds,
                value=min(1, max_folds),
                help="Runs that never finished a fold have no metrics at all.",
            )
        with col_b:
            extra_candidates = [c for c in arg_cols
                                if c not in filter_args and 1 < df[c].nunique(dropna=True) <= 30]
            extra_selected = st.multiselect(
                "Filter on additional args",
                extra_candidates,
                format_func=metric_label,
                placeholder="Pick args to filter on",
            )

        for col_name in extra_selected:
            options = sorted(df[col_name].dropna().unique().tolist())
            filters[col_name] = st.multiselect(
                f"args.{metric_label(col_name)}", options, default=[], placeholder="All"
            )

    # --- Apply filters ---
    filtered = df
    for col_name, selection in filters.items():
        if selection:
            filtered = filtered[filtered[col_name].isin(selection)]
    if name_query:
        filtered = filtered[filtered["Experiment Name"].str.contains(name_query, case=False, na=False)]
    filtered = filtered[filtered["Folds Done"] >= min_folds]

    # --- Metric selection ---
    has_val = bool(val_metric_cols) and df[val_metric_cols].notna().any().any()

    st.markdown("**Metrics**")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        metric_options = ["Validation", "Test", "Both"]
        default_scope = "Validation" if has_val else "Test"
        scope = st.radio(
            "Metric set", metric_options, index=metric_options.index(default_scope),
            horizontal=True,
        )

    if scope == "Validation":
        available_metrics = val_metric_cols
    elif scope == "Test":
        available_metrics = test_metric_cols
    else:
        available_metrics = val_metric_cols + test_metric_cols

    if scope in ("Validation", "Both") and not has_val:
        st.info("None of the loaded runs contain validation metrics.")

    with col_b:
        selected_metrics = st.multiselect(
            "Metrics to display", available_metrics,
            default=available_metrics,
            format_func=metric_label,
        )

    # --- Table ---
    display_cols = ["Experiment Name", "Folds Done", "Folds Total"] + selected_metrics
    display_cols = [c for c in display_cols if c in filtered.columns]

    table = filtered[display_cols].reset_index(drop=True)

    column_config = {
        "Experiment Name": st.column_config.TextColumn("Experiment", width="large"),
        "Folds Done": st.column_config.NumberColumn(
            "Folds Done", format="%d", help="Number of folds with saved metrics."
        ),
        "Folds Total": st.column_config.NumberColumn("Folds Total", format="%d"),
    }
    for col in selected_metrics:
        column_config[col] = st.column_config.NumberColumn(metric_label(col), format="%.4f")

    st.caption(f"Showing {len(table)} of {len(df)} runs. "
               "Click a row to expand its full metadata, click column headers to sort.")

    event = st.dataframe(
        style_metrics(table, selected_metrics),
        height=500,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key="leaderboard",
    )

    # --- Row detail view ---
    selected_rows = event.selection.rows if event and event.selection else []

    if selected_rows:
        record = filtered.reset_index(drop=True).iloc[selected_rows[0]]

        st.markdown("---")
        st.subheader(record["Experiment Name"])
        st.caption(record["Full Path"])

        metadata = record["raw_metadata"] or {}
        data_block = metadata.get("data", {}) if isinstance(metadata, dict) else {}

        info_cols = st.columns(3)
        info_cols[0].metric("Finished Folds", f"{record['Folds Done']} / {record['Folds Total']}")
        info_cols[1].metric("Current Fold", data_block.get("current_fold", "n/a"))
        frozen = data_block.get("frozen_finished")
        info_cols[2].metric(
            "Frozen Phase Done",
            f"{sum(bool(x) for x in frozen)} / {len(frozen)}" if isinstance(frozen, list) else "n/a",
        )

        detail_tabs = st.tabs(["Arguments", "Validation Folds", "Test Folds", "Raw JSON"])

        with detail_tabs[0]:
            args = data_block.get("args", {}) or {}
            if args:
                args_frame = pd.DataFrame(
                    [{"Argument": k, "Value": _arg_to_option(v)} for k, v in sorted(args.items())]
                )
                st.dataframe(args_frame, hide_index=True, height=400)
            else:
                st.info("No arguments stored in this metadata file.")

        with detail_tabs[1]:
            val_frame = folds_to_frame(record["raw_val_folds"])
            if val_frame.empty:
                st.info("No validation metrics stored for this run.")
            else:
                st.dataframe(val_frame.style.format("{:.4f}"))

        with detail_tabs[2]:
            test_frame = folds_to_frame(record["raw_test_folds"])
            if test_frame.empty:
                st.info("No test metrics stored for this run.")
            else:
                st.dataframe(test_frame.style.format("{:.4f}"))

        with detail_tabs[3]:
            st.json(metadata)
    else:
        st.info("Select a row in the table above to see the full metadata of that run.")

# TAB 2: Fold Variance (Deep Dive)
with tab2:
    st.subheader("Cross-Validation Fold Variance")
    st.write("Select experiments to compare the stability of the model across different folds.")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        fold_source = st.radio("Metric set", ["Validation", "Test"], index=0, horizontal=True,
                               key="fold_variance_source")
    source_col = "raw_val_folds" if fold_source == "Validation" else "raw_test_folds"

    with col_b:
        selected_exp_names = st.multiselect("Select Experiments", df["Experiment Name"].unique())

    if selected_exp_names:
        subset = df[df["Experiment Name"].isin(selected_exp_names)]

        expanded_rows = []
        for _, row in subset.iterrows():
            for i, fold_data in enumerate(row[source_col] or []):
                if not isinstance(fold_data, dict) or not fold_data:
                    continue
                expanded_rows.append(
                    {"Experiment Name": row["Experiment Name"], "Fold": f"Fold {i}", **fold_data}
                )

        if expanded_rows:
            folds_df = pd.DataFrame(expanded_rows)

            fold_metric_keys = [c for c in folds_df.columns if c not in ["Experiment Name", "Fold"]]
            target_metric = st.selectbox("Select Metric for Distribution", fold_metric_keys)

            fig_box = px.box(
                folds_df,
                x="Experiment Name",
                y=target_metric,
                points="all",
                color="Experiment Name",
                title=f"Distribution of {target_metric} across Folds",
            )
            st.plotly_chart(fig_box)
        else:
            st.warning(f"No finished {fold_source.lower()} folds found for the selected experiments.")
    else:
        st.info("Please select at least one experiment above.")
