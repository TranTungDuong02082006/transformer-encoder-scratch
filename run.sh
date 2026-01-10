%%writefile run.sh
#!/bin/bash

# 1. Script Configuration
# set -e: Dừng ngay lập tức nếu có lệnh bị lỗi
set -e

# Define colors for logging
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
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

# --- STEP 0: INSTALL DEPENDENCIES ---
log_info "Step 0/6: Installing dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    log_success "Libraries installed successfully."
else
    log_error "Installation failed."
    exit 1
fi

# --- STEP 1: DATA PROCESSING ---
log_info "Step 1/6: Downloading and Processing Data..."
python3 data.py
log_success "Data processing completed."

# --- STEP 2: BUILD TOKENIZER ---
log_info "Step 2/6: Building Tokenizer (Vocab)..."
# Sử dụng tham số khớp với config của bạn
python3 tokenizer.py \
    --train_pkl "train_wiki.pkl" \
    --vocab_size 30000 \
    --save_path "vocab_wiki.json"

log_success "Tokenizer built and vocab saved."

# --- STEP 3: MODEL TRAINING ---
log_info "Step 3/6: Training BERT Model..."
echo "       (This may take a long time depending on your GPU resources...)"
python3 train.py
log_success "Training process finished."

# --- STEP 4: EVALUATION (METRICS) ---
log_info "Step 4/6: Evaluating Model (Calculating PPL & Accuracy)..."
python3 eval.py --mode metrics
log_success "Evaluation completed. Report saved to reports/eval_metrics.json"

# --- STEP 5: PLOTTING RESULTS ---
log_info "Step 5/6: Plotting Training Graphs..."
# Bước này sẽ đọc file CSV và vẽ biểu đồ đẹp
python3 plot_results.py
log_success "Graphs generated! Check 'reports/figures' folder."

# --- STEP 6: DEMO (INFERENCE) ---
log_info "Step 6/6: Running a quick demo..."
TEST_SENTENCE="The capital of Vietnam is <mask > city ."
echo "       Input: '$TEST_SENTENCE'"

python3 eval.py --mode demo --text "$TEST_SENTENCE"

echo "======================================================="
log_success "🎉 PIPELINE FINISHED SUCCESSFULLY!"
echo "Check output files in: checkpoints/ and reports/figures/"
echo "======================================================="