# vsep Installation Guide

## Quick Install

### Standard Installation (CPU)

```bash
# Clone the repository
git clone https://github.com/BF667-IDLE/vsep.git
cd vsep

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from separator import Separator; print('✅ vsep installed successfully!')"
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements-dev.txt
```

## Platform-Specific Instructions

### Windows (NVIDIA GPU)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vsep dependencies
pip install -r requirements.txt

# For GPU acceleration with ONNX
pip install onnxruntime-gpu
```

### Windows (AMD/Intel GPU - DirectML)

```bash
# Install DirectML support
pip install torch-directml
pip install onnxruntime-directml

# Install vsep dependencies
pip install -r requirements.txt
```

### macOS (Apple Silicon)

```bash
# Install dependencies (Metal GPU acceleration is automatic)
pip install -r requirements.txt

# PyTorch with MPS support is included by default
```

### Linux (NVIDIA GPU)

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vsep dependencies
pip install -r requirements.txt

# For GPU acceleration with ONNX
pip install onnxruntime-gpu
```

## Usage

### Command Line

```bash
# Separate audio file
python -m utils.cli your_song.mp3

# Or run directly
python utils/cli.py your_song.mp3

# With specific model
python utils/cli.py your_song.mp3 -m ht-demucs_ft.yaml

# List available models
python utils/cli.py --list_models
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

## Optional Dependencies

### GPU Acceleration

For faster inference with NVIDIA GPUs:

```bash
# Uncomment in requirements.txt or install separately:
pip install onnxruntime-gpu
```

### Python 3.13+ Support

For Python 3.13 and above, `audioop-lts` is automatically installed to replace the deprecated `audioop` module.

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the project directory:

```bash
cd /path/to/vsep
python -c "from separator import Separator"
```

### Missing Dependencies

If you get missing module errors:

```bash
pip install -r requirements.txt --upgrade
```

### GPU Not Detected

1. Make sure you have the correct GPU drivers installed
2. Install the appropriate PyTorch version for your GPU
3. For NVIDIA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
4. For AMD/Intel (Windows): `pip install torch-directml`

## Uninstall

```bash
# Remove installed packages
pip uninstall -y vsep separator torch torchvision torchaudio
pip uninstall -y onnx onnxruntime librosa pydub

# Or manually remove the vsep directory
```

## Update

```bash
# Pull latest changes
git pull origin master

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

For more information, see the main [README.md](README.md)
