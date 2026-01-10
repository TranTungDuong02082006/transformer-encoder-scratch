@echo off
REM =======================================================
REM WINDOWS TRAINING PIPELINE FOR BERT
REM =======================================================

chcp 65001 > NUL

echo =======================================================
echo    STARTING BERT TRAINING PIPELINE (WINDOWS)
echo =======================================================

echo.
echo [INFO] Step 1/5: Processing Data (Download ^& Clean)...
python data.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data processing failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 2/5: Building Tokenizer (Vocab)...
python tokenizer.py --train_pkl "data/processed/train_wiki.pkl" --vocab_size 30000 --save_path "data/processed/vocab_wiki.json"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Tokenizer failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 3/5: Training BERT Model...
echo        (This process depends on your GPU/CPU speed...)
python train.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 4/5: Evaluating Metrics (PPL & Accuracy)...
python eval.py --mode metrics
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Evaluation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 5/5: Running Demo...
echo Input: "The capital of Vietnam is <mask > city ."
python eval.py --mode demo --text "The capital of Vietnam is <mask > city ."

echo.
echo =======================================================
echo    PIPELINE FINISHED SUCCESSFULLY!
echo =======================================================
PAUSE