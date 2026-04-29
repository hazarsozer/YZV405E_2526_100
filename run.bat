@echo off
set PYTHONPATH=%~dp0
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%~dp0.hf_cache

echo === Step 1: Installing dependencies ===
uv sync
if %errorlevel% neq 0 ( echo uv sync failed & exit /b 1 )

echo === Step 2: Training classifier ===
python scripts/train_lr.py ^
  --en-tsvs ^
    data/raw/train_data/EN/train/subtask_a_train.tsv ^
    data/raw/train_data/EN/dev/subtask_a_dev.tsv ^
    data/raw/train_data/EN/test/subtask_a_test.tsv ^
    data/raw/train_data/EN/xeval/subtask_a_xe.tsv ^
  --ptbr-tsvs ^
    data/raw/train_data/PT/train/subtask_a_train.tsv ^
    data/raw/train_data/PT/dev/subtask_a_dev.tsv ^
    data/raw/train_data/PT/test/subtask_a_test.tsv ^
    data/raw/train_data/PT/xeval/subtask_a_xp.tsv ^
  --output models/lr_sense_classifier.joblib ^
  --device cuda
if %errorlevel% neq 0 ( echo Training failed & exit /b 1 )

echo === Step 3: Running inference ===
python scripts/run_inference.py ^
  --data-root data/raw/admire2_data ^
  --templates-root data/submissions/templates ^
  --classifier models/lr_sense_classifier.joblib ^
  --output-dir data/submissions/output ^
  --device cuda
if %errorlevel% neq 0 ( echo Inference failed & exit /b 1 )

echo === Done! Submissions written to data/submissions/output/ ===
