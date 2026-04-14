# Mini-LLM Setup & Run Guide (Windows PowerShell)

## Quick Start

```powershell
# Run the automated setup script
python setup_and_run.py
```

This will:
1. Create virtual environment
2. Install dependencies (PyTorch, sentencepiece)
3. Train the model (if not already trained)
4. Generate sample text

## Manual Steps (if needed)

### 1. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install sentencepiece
```

### 3. Train Model (one-time)
```powershell
python scripts/train.py
```

### 4. Generate Text
```powershell
python scripts/generate.py --prompt "hi" --tokens 80 --temperature 0.8 --top-k 40
```

### 5. Interactive Chat
```powershell
python scripts/chat.py
```

---

## Docker Instructions (if you prefer containers)

See [DOCKER.md](DOCKER.md) for Docker setup.

Quick Docker commands:
```powershell
# Build
docker compose build

# Train
docker compose run --rm train

# Generate
docker compose --profile tools run --rm generate
```






