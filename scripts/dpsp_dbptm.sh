#!/bin/bash
cd /storage/brno12-cerit/home/nexuso1/DeepPSP
python -m pip install keras==2.1.2 numpy>=1.8.0

# 1. Check if the residue argument was passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <residue>"
    echo "Example: $0 ST"
    exit 1
fi

RESIDUE=$1

# Define the base directory where the fold folders are located
BASE_DIR="../phossil/data/dbptm/fastas"

# 3. Check if the base directory actually exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR not found."
    exit 1
fi

echo "Starting training pipeline for residue: $RESIDUE"
echo "------------------------------------------------"

# 4. Loop through all matching fold directories dynamically
for FOLD_DIR in "$BASE_DIR"/splits_${RESIDUE}_fold*; do
    echo $FOLD_DIR
    # Skip if the glob didn't match anything (evaluates to a literal string)
    [ -d "$FOLD_DIR" ] || continue

    # Extract the fold number for logging purposes
    FOLD=$(basename "$FOLD_DIR")

    # 5. Define the completion flag file
    # If this file exists, we know this fold is already done.
    COMPLETION_FLAG="$FOLD_DIR/dpsp_training.completed"

    if [ -f "$COMPLETION_FLAG" ]; then
        echo "Skipping $FOLD - Already completed."
        continue
    fi

    # 6. Find the train data file(s)
    for TRAIN_FILE in "$FOLD_DIR"/train###.fasta; do
        
        # Check if the file actually exists (in case the glob fails)
        [ -f "$TRAIN_FILE" ] || { echo "WARNING: No train*.fasta files found in $FOLD_DIR."; continue; }

        echo "Training on $FOLD with data: $(basename "$TRAIN_FILE")..."
        
        # 7. Execute your Python training command
        python train.py -input "$TRAIN_FILE" -train-type general -residue "$RESIDUE" -name "$RESIDUE_fold$FOLD"
	
        # 8. Check if the python script executed successfully
        if [ $? -eq 0 ]; then
            echo "Successfully trained $FOLD."
            # Create the flag file so it gets skipped in future runs
            touch "$COMPLETION_FLAG"
        else
            echo "Error: Training failed on $FOLD."
            echo "Stopping script"
            # Exit the whole script so you don't cascade failures
            exit 1 
        fi
    done

done

echo "------------------------------------------------"
echo "All available folds for $RESIDUE have been processed!"
