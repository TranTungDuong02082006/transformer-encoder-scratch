@echo off
REM =======================================================
REM WINDOWS TRAINING PIPELINE FOR BERT
REM =======================================================

chcp 65001 > NUL

echo =======================================================
echo    STARTING BERT TRAINING PIPELINE (WINDOWS)
echo =======================================================

echo.
echo [INFO] Step 0/6: Installing dependencies...
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Library installation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 1/6: Processing Data (Download ^& Clean)...
python data.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data processing failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 2/6: Building Tokenizer (Vocab)...
python tokenizer.py --train_pkl "train_wiki.pkl" --vocab_size 30000 --save_path "vocab_wiki.json"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Tokenizer failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 3/6: Training BERT Model...
echo        (This process depends on your GPU/CPU speed...)
python train.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 4/6: Evaluating Metrics (PPL ^& Accuracy)...
python eval.py --mode metrics
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Evaluation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 5/6: Plotting Training Results...
python plot_results.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Plotting failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [INFO] Step 6/6: Running Demo...
echo Input: "The capital of Vietnam is <mask > city ."
python eval.py --mode demo --text "The capital of Vietnam is <mask > city ."

echo.
echo =======================================================
echo    PIPELINE FINISHED SUCCESSFULLY!
echo    Check 'reports/figures' for graphs.
echo =======================================================
PAUSE