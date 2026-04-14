#!/usr/bin/env python
"""Setup and run the mini-llm project."""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Success: {description}")

def main():
    """Main setup and run function."""
    project_root = Path(__file__).parent
    
    print("\n" + "="*60)
    print("Mini-LLM Setup & Execution")
    print("="*60)
    
    # Detect or create venv
    venv_dir = project_root / "venv"
    if sys.platform == "win32":
        python_exe = venv_dir / "Scripts" / "python.exe"
        activate_script = venv_dir / "Scripts" / "Activate.ps1"
    else:
        python_exe = venv_dir / "bin" / "python"
        activate_script = venv_dir / "bin" / "activate"
    
    # Create venv if it doesn't exist
    if not venv_dir.exists():
        print(f"\n🔨 Creating virtual environment at {venv_dir}...")
        run_command(
            [sys.executable, "-m", "venv", str(venv_dir)],
            "Create virtual environment"
        )
    
    # Install dependencies in venv
    print(f"\n📦 Installing dependencies in venv...")
    run_command(
        [str(python_exe), "-m", "pip", "install", 
         "--index-url", "https://download.pytorch.org/whl/cu121",
         "torch==2.5.1"],
        "Install PyTorch"
    )
    run_command(
        [str(python_exe), "-m", "pip", "install", "sentencepiece>=0.1.99"],
        "Install sentencepiece"
    )
    
    # Check if artifacts dir exists with model
    artifacts = project_root / "artifacts"
    model_file = artifacts / "tiny_llm.pt"
    tokenizer_file = artifacts / "tokenizer.model"
    
    if not model_file.exists() or not tokenizer_file.exists():
        print("\n⚠️  Model or tokenizer not found. Training first...")
        run_command(
            [str(python_exe), str(project_root / "scripts" / "train.py")],
            "Train model"
        )
    else:
        print(f"\n✅ Found pre-trained model at {model_file}")
    
    # Run generate
    print("\n" + "="*60)
    print("Running text generation...")
    print("="*60)
    run_command(
        [str(python_exe), str(project_root / "scripts" / "generate.py"), 
         "--prompt", "hello", "--tokens", "50"],
        "Generate text"
    )

if __name__ == "__main__":
    main()

