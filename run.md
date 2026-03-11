source .venv/bin/activate
python train_char_lm.py

python generate.py --prompt "The " --tokens 100