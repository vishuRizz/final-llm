# Docker (Windows, macOS, Linux)

Training runs inside a **Linux** container with **CPU PyTorch**, so host OS and Python installs do not matter. Install [Docker Desktop](https://docs.docker.com/desktop/) on Windows or macOS, or Docker Engine on Linux.

## One-time: build the image

From the project root (`mini-ml`):

```bash
docker build -t mini-ml:latest .
```

Or:

```bash
docker compose build
```

## Train

Mount the project folder so `artifacts/`, `data/processed/`, and logs stay on your machine:

```bash
docker run --rm -v "%cd%":/app mini-ml:latest
```

On PowerShell:

```powershell
docker run --rm -v "${PWD}:/app" mini-ml:latest
```

On macOS/Linux:

```bash
docker run --rm -v "$(pwd):/app" mini-ml:latest
```

Or with Compose (same bind mount):

```bash
docker compose run --rm train
```

Training writes:

- `artifacts/tiny_llm.pt`
- `artifacts/tokenizer.model`
- `data/processed/*.txt`

## After training: generate text

```bash
docker run --rm -v "${PWD}:/app" mini-ml:latest python scripts/generate.py --prompt "hi" --tokens 80
```

(Adjust the volume flag for Windows CMD/PowerShell as above.)

## Optional: GPU

This image uses **CPU** PyTorch for maximum compatibility. GPU training needs an NVIDIA GPU, the NVIDIA Container Toolkit, and a CUDA-enabled PyTorch install in the Dockerfile; that is not included by default.
