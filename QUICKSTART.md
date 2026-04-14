# Quick Start Guide

## Step 1: Open PowerShell and navigate to the project

```powershell
cd D:\minillm\mini-llm
```

## Step 2: Create and activate virtual environment

```powershell
# Create venv
python -m venv venv

# Activate (on Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Step 3: Install dependencies

```powershell
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2
pip install sentencepiece==0.1.99
```

Verify installation:
```powershell
python -c "import torch; import sentencepiece; print('✅ All dependencies installed')"
```

## Step 4: Train the model (one-time setup)

```powershell
python scripts/train.py
```

This will:
- Process the CSV files in `data/`
- Train a tokenizer
- Train the mini-LLM model
- Save model to `artifacts/tiny_llm.pt`

**This takes a few minutes.**

## Step 5: Generate text

```powershell
python scripts/generate.py --prompt "hello" --tokens 50
```

Or use the interactive chat:
```powershell
python scripts/chat.py
```

## Alternative: Run everything at once

```powershell
python setup_and_run.py
```

This will automatically train if no model exists, then generate text.
