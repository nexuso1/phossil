import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout="wide", page_title="ML Experiment Comparator")

# --- Helper Functions ---

@st.cache_data
def load_experiments(root_dir):
    """
    Recursively finds all metadata.json files in the root_dir
    and flattens them into a Pandas DataFrame.
    """
    # Find all metadata.json files recursively
    search_path = os.path.join(root_dir, "**", "metadata.json")
    files = glob.glob(search_path, recursive=True)
    
    experiments = []

    if not files:
        return pd.DataFrame()

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
                
            # Access the main data block
            data_block = content.get("data", {})
            
            # 1. Identify the experiment (Folder Name)
            folder_path = os.path.dirname(file_path)
            experiment_name = os.path.basename(folder_path)
            
            # 2. Flatten Args
            args = data_block.get("args", {})
            
            # 3. Flatten Average Metrics
            avg_metrics = data_block.get("test_metric_avg", {})
            
            # 4. Store raw fold data for later visualization (store as a list of dicts)
            fold_metrics = data_block.get("test_metrics", [])
            
            # Combine into a single dictionary
            row = {
                "Experiment Name": experiment_name,
                "Full Path": folder_path,
                "Fold Count": len(fold_metrics),
                "raw_folds": fold_metrics # Hidden column for data processing
            }
            
            # Add prefix to metrics to distinguish them easily in UI
            for k, v in avg_metrics.items():
                row[f"METRIC_{k}"] = v
                
            # Add args
            for k, v in args.items():
                row[f"ARG_{k}"] = v
                
            experiments.append(row)
            
        except Exception as e:
            st.warning(f"Skipping corrupt or malformed file: {file_path}. Error: {e}")

    return pd.DataFrame(experiments)

# --- Main Layout ---

st.title("ML Model Comparison Dashboard")

# 1. Sidebar: Configuration
with st.sidebar:
    st.header("Data Source")
    default_dir = "." 
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

# Separate columns into categories for easier filtering
all_cols = df.columns.tolist()
metric_cols = [c for c in all_cols if c.startswith("METRIC_")]
arg_cols = [c for c in all_cols if c.startswith("ARG_")]
meta_cols = ["Experiment Name", "Fold Count"]

# --- Tab View ---
tab1, tab2, tab3 = st.tabs(["Comparison Table", "Hyperparameters", "Fold Variance"])

# TAB 1: Main Comparison Table
with tab1:
    st.subheader("Leaderboard")
    
    # Column Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_metrics = st.multiselect("Select Metrics to Display", metric_cols, default=metric_cols[:3])
    with col2:
        selected_args = st.multiselect("Select Args to Display", arg_cols, default=["ARG_lr", "ARG_batch_size", "ARG_dropout", "ARG_model_path"])

    # Prepare display dataframe
    display_cols = ["Experiment Name"] + selected_metrics + selected_args
    # Filter out columns that might not exist if selection changed
    display_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(
        df[display_cols].style.background_gradient(subset=selected_metrics, cmap="viridis"),
        use_container_width=True,
        height=500
    )
    
    st.caption("Tip: Click column headers to sort. The colors indicate higher (yellow) vs lower (purple) values.")

# TAB 2: Hyperparameter Analysis (Scatter Plots)
with tab2:
    st.subheader("Hyperparameter vs. Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X-Axis (Hyperparameter)", arg_cols, index=arg_cols.index("ARG_lr") if "ARG_lr" in arg_cols else 0)
    with col2:
        y_axis = st.selectbox("Y-Axis (Metric)", metric_cols, index=0)
    with col3:
        color_col = st.selectbox("Color By (Optional)", ["None"] + arg_cols + ["Experiment Name"])

    # Handle "None" selection for color
    color_arg = None if color_col == "None" else color_col

    if x_axis and y_axis:
        fig = px.scatter(
            df, 
            x=x_axis, 
            y=y_axis, 
            color=color_arg,
            hover_data=["Experiment Name"],
            title=f"{y_axis} vs {x_axis}",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Fold Variance (Deep Dive)
with tab3:
    st.subheader("Cross-Validation Fold Variance")
    st.write("Select experiments to compare the stability of the model across different folds.")

    # User selects experiments
    selected_exp_names = st.multiselect("Select Experiments", df["Experiment Name"].unique())

    if selected_exp_names:
        # Filter data
        subset = df[df["Experiment Name"].isin(selected_exp_names)].copy()
        
        # We need to expand the 'raw_folds' list into rows for plotting
        expanded_rows = []
        for _, row in subset.iterrows():
            exp_name = row["Experiment Name"]
            folds = row["raw_folds"] # This is the list of dicts from JSON
            
            for i, fold_data in enumerate(folds):
                fold_info = {"Experiment Name": exp_name, "Fold": f"Fold {i}"}
                # Add all metrics from the fold
                for k, v in fold_data.items():
                    fold_info[k] = v
                expanded_rows.append(fold_info)
        
        if expanded_rows:
            folds_df = pd.DataFrame(expanded_rows)
            
            # Select metric to visualize distribution
            # Get metrics available in the folds data
            fold_metric_keys = [c for c in folds_df.columns if c not in ["Experiment Name", "Fold"]]
            target_metric = st.selectbox("Select Metric for Distribution", fold_metric_keys)
            
            # Box Plot
            fig_box = px.box(
                folds_df, 
                x="Experiment Name", 
                y=target_metric, 
                points="all", # Show individual points
                color="Experiment Name",
                title=f"Distribution of {target_metric} across Folds"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
        else:
            st.warning("No fold data found for the selected experiments.")
    else:
        st.info("Please select at least one experiment above.")