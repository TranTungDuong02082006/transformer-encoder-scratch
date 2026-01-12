set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "======================================================="
echo "   🚀 STARTING BERT END-TO-END TRAINING PIPELINE"
echo "======================================================="

log_info "Step 0/6: Installing dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    log_success "Libraries installed successfully."
else
    log_error "Installation failed."
    exit 1
fi

log_info "Step 1/6: Downloading and Processing Data..."
python3 data.py
log_success "Data processing completed."

log_info "Step 2/6: Building Tokenizer (Vocab)..."
python3 tokenizer.py
log_success "Tokenizer built and vocab saved."

log_info "Step 3/6: Training BERT Model..."
echo "       (This may take a long time depending on your GPU resources...)"
python3 train.py
log_success "Training process finished."

log_info "Step 4/6: Evaluating Model (Calculating PPL & Accuracy)..."
python3 eval.py --mode metrics
log_success "Evaluation completed. Report saved to reports/eval_metrics.json"

log_info "Step 5/6: Plotting Training Graphs..."
python3 plot_results.py
log_success "Graphs generated! Check 'reports/figures' folder."

log_info "Step 6/6: Running a quick demo..."
TEST_SENTENCE="The capital of Vietnam is <mask > city ."
echo "       Input: '$TEST_SENTENCE'"

python3 eval.py --mode demo --text "$TEST_SENTENCE"

echo "======================================================="
log_success "🎉 PIPELINE FINISHED SUCCESSFULLY!"
echo "Check output files in: checkpoints/ and reports/figures/"
echo "======================================================="