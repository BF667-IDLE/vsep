# vsep - Lightning-Fast Audio Stem Separator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667-IDLE/vsep/blob/main/notebooks/vsep_demo.ipynb)

**vsep** is a high-performance audio stem separator that splits music into vocals, drums, bass, and other instruments using state-of-the-art AI models from UVR (Ultimate Vocal Remover).

## 🚀 Features

- **⚡ Fast Downloads** - Parallel model downloads (4-8x faster than standard)
- **🔄 Resume Support** - Automatically resumes interrupted downloads
- **🎯 Multiple Architectures** - Support for MDX, VR, Demucs, and MDXC models
- **🎚️ Ensemble Mode** - Combine multiple models for better quality
- **💻 GPU/CPU/DirectML** - Works with NVIDIA GPU, Apple MPS, AMD DirectML, or CPU
- **🔧 Configurable** - Easy-to-use configuration system for custom settings
- **🌐 Remote API** - Deploy to Cloud Run or Modal for cloud processing

## 🎯 Quick Start

### Installation

**Clone and install dependencies:**

```bash
git clone https://github.com/BF667-IDLE/vsep.git
cd vsep
pip install -r requirements.txt
```

**For development (includes testing tools):**

```bash
pip install -r requirements-dev.txt
```

See [INSTALL.md](INSTALL.md) for detailed platform-specific instructions (Windows GPU, macOS, Linux).

### Basic Usage

**Separate vocals from instrumentation:**

```bash
python utils/cli.py your_song.mp3
```

**Use a specific model:**

```bash
python utils/cli.py your_song.mp3 -m UVR-MDX-NET-Inst_1.onnx
```

**List available models:**

```bash
python utils/cli.py --list_models
```

**Download a model:**

```bash
python utils/cli.py --download_model_only UVR-MDX-NET-Inst_1.onnx
```

### Python API

```python
from separator import Separator

# Initialize
separator = Separator()

# Separate audio
output_files = separator.separate("your_song.mp3")

print(f"Separated files: {output_files}")
```

**Advanced usage with custom settings:**
```python
from separator import Separator
import config.variables as cfg

# Customize download settings
cfg.MAX_DOWNLOAD_WORKERS = 8  # More parallel downloads
cfg.DOWNLOAD_CHUNK_SIZE = 524288  # 512KB chunks

separator = Separator(
    model_file_dir="./models",
    sample_rate=44100,
    use_soundfile=True,
)

output_files = separator.separate("your_song.mp3")
```

## 🎵 Available Models

vsep supports 100+ models from UVR. Here are some popular ones:

| Model | Architecture | Stems | Quality |
|-------|-------------|-------|---------|
| `ht-demucs-ft.yaml` | Demucs v4 | vocals, drums, bass, other | ⭐⭐⭐⭐⭐ |
| `UVR-MDX-NET-Inst_1.onnx` | MDX-Net | vocals, instrumental | ⭐⭐⭐⭐ |
| `BS-Roformer-Viperx-1297.ckpt` | Roformer | vocals, instrumental | ⭐⭐⭐⭐⭐ |
| `Mel-Roformer-Viperx-1053.ckpt` | Roformer | vocals, instrumental | ⭐⭐⭐⭐⭐ |

**List all available models:**
```bash
audio-separator --list_models
```

**Download a specific model:**
```bash
audio-separator --download_model_only UVR-MDX-NET-Inst_1.onnx
```

## 🧪 Try It Online

Run vsep in Google Colab with free GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667-IDLE/vsep/blob/main/notebooks/vsep_demo.ipynb)

The Colab notebook includes:
- Interactive audio upload
- Model selection dropdown
- Audio playback for results
- Download separated stems

## ⚙️ Configuration

All configuration is centralized in `config/variables.py`:

```python
import config.variables as cfg

# Use mirror repository
cfg.UVR_PUBLIC_REPO_URL = "https://your-mirror.com/models"

# Adjust for your connection
cfg.MAX_DOWNLOAD_WORKERS = 8  # Parallel downloads (default: 4)
cfg.DOWNLOAD_CHUNK_SIZE = 524288  # Chunk size (default: 256KB)
cfg.DOWNLOAD_TIMEOUT = 600  # Timeout in seconds (default: 300)
```

See [`config/README.md`](config/README.md) for full documentation.

## 📊 Performance

**Download Speed Comparison:**

| Method | Time (100MB model) |
|--------|-------------------|
| Standard | ~60 seconds |
| vsep (parallel) | ~15 seconds |
| vsep + mirror | ~8 seconds |

**Separation Speed:**

| Model | CPU (RTX 3060) | GPU (RTX 3060) |
|-------|---------------|----------------|
| Demucs v4 | ~30 seconds | ~8 seconds |
| MDX-Net | ~45 seconds | ~12 seconds |
| Roformer | ~60 seconds | ~15 seconds |

## 🛠️ Advanced Features

### Ensemble Separation

Combine multiple models for superior quality:

```bash
# Use built-in preset
audio-separator song.mp3 --ensemble_preset vocals_ensemble

# Custom ensemble
audio-separator song.mp3 --model_filename model1.onnx --extra_models model2.onnx model3.onnx --ensemble_algorithm median_wave
```

Available ensemble algorithms:
- `avg_wave` - Average waveforms (default)
- `median_wave` - Median waveform (removes artifacts)
- `max_wave` - Maximum amplitude
- `avg_fft` - Average in frequency domain
- `median_fft` - Median in frequency domain

### Chunking for Long Files

Process long audio files in chunks to reduce memory usage:

```python
separator = Separator(chunk_duration=60)  # Process in 60-second chunks
output_files = separator.separate("long_mix.mp3")
```

### Remote Deployment

Deploy vsep as a cloud API:

**Modal (GPU):**
```bash
python remote/deploy_modal.py deploy
```

**Google Cloud Run:**
```bash
python remote/deploy_cloudrun.py deploy
```

See [`remote/README.md`](remote/README.md) for deployment details.

## 📁 Project Structure

```
vsep/
├── config/              # Configuration and settings
│   ├── variables.py     # Centralized config
│   ├── __init__.py
│   └── README.md
├── separator/           # Core separation logic
│   ├── separator.py     # Main Separator class
│   ├── architectures/   # Model architectures (MDX, VR, Demucs)
│   └── uvr_lib_v5/      # UVR library code
├── remote/              # Cloud deployment
│   ├── deploy_modal.py
│   ├── deploy_cloudrun.py
│   └── api_client.py
├── utils/               # Utilities
│   └── cli.py          # Command-line interface
├── notebooks/           # Jupyter/Colab demos
└── tools/              # Development tools
```

## 🔧 Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black . --line-length 140
```

### Building from Source

```bash
poetry build
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UVR Team** - For the amazing models and training data
- **Anjok07** - Primary model trainer and UVR developer
- **TRvlvr** - For the model repository
- **NomadKaraoke** - For the python-audio-separator project

## 💬 Support

- **Issues:** [GitHub Issues](https://github.com/BF667-IDLE/vsep/issues)
- **Discussions:** [GitHub Discussions](https://github.com/BF667-IDLE/vsep/discussions)
- **Colab Demo:** [Try it free](https://colab.research.google.com/github/BF667-IDLE/vsep/blob/main/notebooks/vsep_demo.ipynb)

## 📬 Citation

If you use vsep in your research, please cite:

```bibtex
@software{vsep2024,
  author = {Beveridge, Andrew and contributors},
  title = {vsep: Fast Audio Stem Separator},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/BF667-IDLE/vsep}
}
```

---

<div align="center">

**Made with ❤️ by the audio separation community**

[Report Bug](https://github.com/BF667-IDLE/vsep/issues) · [Request Feature](https://github.com/BF667-IDLE/vsep/issues) · [Try Colab Demo](https://colab.research.google.com/github/BF667-IDLE/vsep/blob/main/notebooks/vsep_demo.ipynb)

</div>
