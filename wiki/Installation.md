# Installation

## Prerequisites

- **Python 3.10+** (3.13 supported with `audioop-lts` fallback)
- **FFmpeg** — required for audio I/O
- **PyTorch 2.3+** — installed automatically
- **Git** — to clone the repository

## Install FFmpeg

### Ubuntu / Debian
```bash
sudo apt update && sudo apt install ffmpeg
```

### macOS
```bash
brew install ffmpeg
```

### Windows
Download from [ffmpeg.org](https://ffmpeg.org/download.html) or:
```cmd
choco install ffmpeg
```

## Clone and Install

```bash
git clone https://github.com/BF667-IDLE/vsep.git
cd vsep
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "from separator import Separator; print('vsep ready!')"
```

## GPU Support

### NVIDIA CUDA
```bash
pip uninstall onnxruntime -y && pip install onnxruntime-gpu
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
```

### Apple Silicon (MPS)
PyTorch automatically uses MPS on Apple Silicon Macs. No extra configuration needed.

### AMD / Intel (DirectML, Windows only)
```bash
pip uninstall onnxruntime -y && pip install onnxruntime-directml
pip install torch-directml
```

## Google Colab

No local installation needed. Open the [Colab notebook](https://colab.research.google.com/github/BF667-IDLE/vsep/blob/main/notebooks/vsep_demo.ipynb) and run — everything is installed in the notebook cells.

## Development Setup

```bash
git clone https://github.com/BF667-IDLE/vsep.git
cd vsep
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run tests:
```bash
pytest tests/ -v
```

Format code:
```bash
black . --line-length 140
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VSEP_MODEL_DIR` | Directory for downloaded models | `/tmp/audio-separator-models/` |

```bash
export VSEP_MODEL_DIR=/path/to/my/models
python utils/cli.py song.mp3
```
