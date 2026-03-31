<div align="center">
<img src="https://raw.githubusercontent.com/BF667-IDLE/vsep/main/docs/logo.svg" alt="vsep logo" width="300"/>
</div>

**vsep** is a lightning-fast audio stem separator powered by 100+ AI models from the Ultimate Vocal Remover ecosystem. It splits music into vocals, drums, bass, and other instruments using state-of-the-art architectures.

## Quick Links

- 🚀 [Installation](Installation) — Get started in 2 minutes
- 📋 [CLI Reference](CLI-Reference) — Full command-line docs
- 🎵 [Model Catalog](Model-Catalog) — All models with recommendations
- 📖 [Python API](Python-API) — Use vsep as a Python library
- 🔧 [Configuration](Configuration) — All settings explained
- 🏗️ [Architecture](Architecture) — How it works internally
- 📓 [Google Colab](Google-Colab) — Run in your browser
- 🔧 [Troubleshooting](Troubleshooting) — Fix common issues

## Features

- **100+ models** across 4 architectures (VR, MDX-Net, Demucs, Roformer/MDXC)
- **Automatic architecture detection** from model filename
- **GPU acceleration** (NVIDIA CUDA, Apple Silicon MPS, AMD DirectML)
- **11 ensemble algorithms** for combining multiple models
- **Audio chunking** for arbitrarily long files
- **Parallel model downloads** with resume support
- **Google Colab notebook** — no local setup needed
- **CLI and Python API** for integration
- **Remote deployment** (Modal, Google Cloud Run)

## Example

```bash
# Separate vocals from a song
python utils/cli.py song.mp3

# Use a specific model
python utils/cli.py song.mp3 -m Kim_Vocal_2.onnx

# List all available models
python utils/cli.py --list_models
```

```python
from separator import Separator

separator = Separator()
output_files = separator.separate("song.mp3")
```

## Supported Architectures

| Architecture | Models | Extension | Best For |
|:-------------|-------:|:---------|:---------|
| **VR** (Band Split) | 29 | `.pth` | Lightweight separation, TTA support |
| **MDX-Net** | 39 | `.onnx` | Fast inference, community models |
| **MDXC** (Roformer) | 83+ | `.ckpt` | Best vocal quality, high SDR |
| **Demucs** | 24 | `.th` / `.yaml` | 4-stem separation |

## License

MIT License. Individual AI models may have their own licenses — check before commercial use.
