#!/bin/bash

# 1. Script Configuration
# set -e: Exit immediately if a command exits with a non-zero status.
# This ensures we don't train on broken data if a previous step fails.
set -e

# Define colors for better logging visibility
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color (Reset)

# Helper functions for logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 2. Start Pipeline

echo "======================================================="
echo "   🚀 STARTING BERT END-TO-END TRAINING PIPELINE"
echo "======================================================="

# --- STEP 1: DATA PROCESSING ---
log_info "Step 1/5: Downloading and Processing Data..."
# Run data.py to download Wikipedia, clean, and split the dataset
python3 data.py
if [ $? -eq 0 ]; then
    log_success "Data processing completed."
else
    log_error "Data processing failed."
    exit 1
fi

# --- STEP 2: BUILD TOKENIZER ---
log_info "Step 2/5: Building Tokenizer (Vocab)..."
# Run tokenizer.py to build the vocabulary from the training set
# Ensure 'save_path' matches the one defined in src/config.py
python3 tokenizer.py \
    --train_pkl "data/processed/train_wiki.pkl" \
    --vocab_size 30000 \
    --save_path "data/processed/vocab_wiki.json"

log_success "Tokenizer built and vocab saved."

# --- STEP 3: MODEL TRAINING ---
log_info "Step 3/5: Training BERT Model..."
echo "       (This may take a long time depending on your GPU resources...)"

# Run train.py. 
# It automatically loads hyperparameters from src/config.py
python3 train.py

log_success "Training process finished."

# --- STEP 4: EVALUATION (METRICS) ---
log_info "Step 4/5: Evaluating Model (Calculating PPL & Accuracy)..."
# Evaluate on the Validation set and save the report to reports/eval_metrics.json
python3 eval.py --mode metrics

log_success "Evaluation completed. Report saved to reports/eval_metrics.json"

# --- STEP 5: DEMO (INFERENCE) ---
log_info "Step 5/5: Running a quick demo..."
TEST_SENTENCE="The capital of Vietnam is <mask> city ."
echo "       Input: '$TEST_SENTENCE'"

# Run a quick fill-mask test to verify the model visually
python3 eval.py --mode demo --text "$TEST_SENTENCE"

echo "======================================================="
log_success "🎉 PIPELINE FINISHED SUCCESSFULLY!"
echo "======================================================="