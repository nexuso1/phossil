#!/bin/bash
cd /storage/brno12-cerit/home/nexuso1/DeepPSP
python -m pip install keras==2.1.2 numpy>=1.8.0

# 1. Check if the residue argument was passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <residue> <fold_number>"
    echo "Example: $0 ST 1"
    exit 1
fi

RESIDUE=$1
FOLD=$2

# Define the base directory where the fold folders are located
BASE_DIR="../phossil/data/dbptm/fastas"

# 3. Check if the base directory actually exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR not found."
    exit 1
fi

# 4. Loop through all matching fold directories dynamically
FOLD_DIR="$BASE_DIR"/splits_${RESIDUE}_fold$FOLD
echo $FOLD_DIR
# Skip if the glob didn't match anything (evaluates to a literal string)
[ -d "$FOLD_DIR" ] || continue

# 6. Find the train data file(s)
TRAIN_FILE="$FOLD_DIR"/train###.fasta
TEST_FILE="$FOLD_DIR"/test.fasta

# Check if the file actually exists (in case the glob fails)
[ -f "$TRAIN_FILE" ] || { echo "WARNING: No train*.fasta files found in $FOLD_DIR."; continue; }

Cecho "Testing on $FOLD with data: $(basename "$TEST_FILE" )..."
MODEL_NAME=splits_"$RESIDUE"_fold"$FOLD"
OUTPUT="$MODEL_NAME"_result

echo "$MODEL_NAME"

# 7. Execute your Python training command
python predict.py -input "$TEST_FILE" -train "$TRAIN_FILE" -predict-type general -residue "$RESIDUE" -output "$OUTPUT" -name "$MODEL_NAME"

