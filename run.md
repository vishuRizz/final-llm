cd ~/Desktop/mini-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train:
# - auto-cleans data/train.csv + data/validation.csv (or data/val.csv) + data/test.csv
# - writes processed files to data/processed/
# - trains SentencePiece tokenizer to artifacts/tokenizer.model
# - trains model and saves best checkpoint to artifacts/tiny_llm.pt
python scripts/train.py

# Generate one response
python scripts/generate.py --prompt "hi" --tokens 80 --temperature 0.8 --top-k 40

# Interactive chat
python scripts/chat.py