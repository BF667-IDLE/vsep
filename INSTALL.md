# vsep Installation Guide

This guide covers installing vsep on all major platforms. If you run into any issues not covered here, please [open an issue](https://github.com/BF667-IDLE/vsep/issues).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Install (CPU)](#quick-install-cpu)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [Windows (NVIDIA GPU / CUDA)](#windows-nvidia-gpu--cuda)
  - [Windows (AMD or Intel GPU / DirectML)](#windows-amd-or-intel-gpu--directml)
  - [macOS (Apple Silicon)](#macos-apple-silicon)
  - [macOS (Intel)](#macos-intel)
  - [Linux (NVIDIA GPU / CUDA)](#linux-nvidia-gpu--cuda)
  - [Linux (CPU only)](#linux-cpu-only)
- [Post-Installation Verification](#post-installation-verification)
- [Optional Dependencies](#optional-dependencies)
- [Troubleshooting](#troubleshooting)
- [Updating](#updating)
- [Uninstalling](#uninstalling)

---

## Prerequisites

Before installing vsep, make sure the following are available on your system:

| Requirement | Minimum Version | How to Check | Install |
|:------------|:----------------|:-------------|:--------|
| **Python** | 3.10 | `python --version` | [python.org](https://www.python.org/downloads/) |
| **pip** | Latest | `pip --version` | `python -m pip install --upgrade pip` |
| **FFmpeg** | Any recent | `ffmpeg -version` | See below |
| **Git** | Any recent | `git --version` | [git-scm.com](https://git-scm.com/downloads) |

### Installing FFmpeg

FFmpeg is **required** — it is used by pydub to read and write audio files in formats other than WAV.

- **Ubuntu / Debian:** `sudo apt update && sudo apt install ffmpeg`
- **macOS (Homebrew):** `brew install ffmpeg`
- **Windows (Chocolatey):** `choco install ffmpeg`
- **Windows (manual):** Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin/` folder to your system `PATH`.
- **Windows (Scoop):** `scoop install ffmpeg`

---

## Quick Install (CPU)

This is the simplest way to get started. GPU acceleration is not required — vsep works perfectly on CPU, just a bit slower.

```bash
# 1. Clone the repository
git clone https://github.com/BF667-IDLE/vsep.git
cd vsep

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python -c "from separator import Separator; print('vsep installed successfully!')"
```

For a development setup (includes pytest, black, coverage):

```bash
pip install -r requirements-dev.txt
```

---

## Platform-Specific Instructions

### Windows (NVIDIA GPU / CUDA)

NVIDIA GPUs provide the fastest separation times. You need the CUDA toolkit and matching PyTorch version.

```bash
# Step 1: Install system prerequisites
# - NVIDIA GPU driver (latest from nvidia.com/drivers)
# - CUDA Toolkit 11.8 or 12.1 (from developer.nvidia.com/cuda-downloads)
# - FFmpeg (see Prerequisites section)

# Step 2: Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install vsep dependencies
pip install -r requirements.txt

# Step 4: Replace CPU onnxruntime with GPU version
pip uninstall onnxruntime -y
pip install onnxruntime-gpu

# Step 5: Verify GPU is detected
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import onnxruntime as ort; print(f'Providers: {ort.get_available_providers()}')"
```

**Expected output:**
```
CUDA available: True
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

> **Tip:** If you see `CUDA available: False`, double-check that your NVIDIA driver version is compatible with the installed CUDA toolkit version. A common issue is having a newer driver that requires CUDA 12.x while you installed the cu118 (CUDA 11.8) variant of PyTorch.

### Windows (AMD or Intel GPU / DirectML)

DirectML enables GPU acceleration on AMD and Intel GPUs on Windows. Performance is typically 60–80% of CUDA on comparable hardware.

```bash
# Step 1: Install FFmpeg (see Prerequisites section)

# Step 2: Install PyTorch with DirectML support
pip install torch-directml

# Step 3: Install vsep dependencies
pip install -r requirements.txt

# Step 4: Replace onnxruntime with DirectML version
pip uninstall onnxruntime -y
pip install onnxruntime-directml

# Step 5: Verify
python -c "import torch_directml; print(f'DirectML available: {torch_directml.is_available()}')"
```

### macOS (Apple Silicon)

M1/M2/M3/M4 Macs use Apple's Metal Performance Shaders (MPS) for GPU acceleration through PyTorch. This is enabled automatically when MPS-compatible PyTorch is installed.

```bash
# Step 1: Install FFmpeg via Homebrew
brew install ffmpeg

# Step 2: Install vsep dependencies (PyTorch MPS is included by default)
pip install -r requirements.txt

# Step 3: Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

> **Note:** MPS support in PyTorch is still maturing. Some edge cases may fall back to CPU automatically. If you encounter errors, try setting `use_autocast=False` when creating the `Separator`.

### macOS (Intel)

Intel-based Macs do not have MPS support. vsep will use CPU inference, which is still functional for all models but slower than GPU.

```bash
brew install ffmpeg
pip install -r requirements.txt
```

### Linux (NVIDIA GPU / CUDA)

```bash
# Step 1: Install system packages (Ubuntu/Debian example)
sudo apt update
sudo apt install ffmpeg build-essential python3-dev

# Step 2: Install NVIDIA driver and CUDA toolkit
# Follow the guide at: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
# Or use the NVIDIA package manager:
# sudo apt install nvidia-driver-535 nvidia-cuda-toolkit

# Step 3: Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install vsep dependencies
pip install -r requirements.txt

# Step 5: GPU-accelerated ONNX runtime
pip uninstall onnxruntime -y
pip install onnxruntime-gpu

# Step 6: Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Linux (CPU only)

```bash
sudo apt update
sudo apt install ffmpeg
pip install -r requirements.txt
```

---

## Post-Installation Verification

After installing vsep, run these checks to confirm everything is working:

```bash
# 1. Import test
python -c "from separator import Separator; print('Import OK')"

# 2. Environment info (shows OS, Python, PyTorch, ONNX, FFmpeg versions)
python utils/cli.py -e

# 3. List models (verifies network connectivity to model repos)
python utils/cli.py --list_models --list_limit 5

# 4. Quick separation test (downloads a model and separates a short clip)
python utils/cli.py your_test_song.mp3 -m UVR-MDX-NET-Inst_1.onnx
```

---

## Optional Dependencies

### GPU Acceleration for ONNX Runtime

By default, vsep installs `onnxruntime` (CPU). For GPU acceleration:

| Platform | Package | Install Command |
|:---------|:--------|:----------------|
| NVIDIA (CUDA) | `onnxruntime-gpu` | `pip install onnxruntime-gpu` |
| AMD/Intel (Windows) | `onnxruntime-directml` | `pip install onnxruntime-directml` |
| Apple Silicon | `onnxruntime-silicon` | `pip install onnxruntime-silicon` |

Only **one** onnxruntime variant should be installed at a time. Install the GPU variant and uninstall the CPU one:

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu  # Replace with your platform variant
```

### Python 3.13+ Compatibility

The `audioop` module was removed from Python's standard library in version 3.13. vsep automatically installs `audioop-lts` as a replacement when running on Python 3.13+. No manual action is needed.

### Remote Deployment Dependencies

To deploy vsep as a cloud API:

```bash
# Modal deployment
pip install modal

# Google Cloud Run deployment
pip install google-cloud-run
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'separator'`

**Cause:** You are not in the `vsep/` project directory, or Python cannot find the module.

**Fix:**
```bash
cd /path/to/vsep
python -c "from separator import Separator"
```

If you want to import vsep from anywhere, add the project root to your `PYTHONPATH`:
```bash
export PYTHONPATH="/path/to/vsep:$PYTHONPATH"
```

### `FFmpeg is not installed`

**Cause:** FFmpeg is not on your system `PATH`, or is not installed.

**Fix:** Install FFmpeg following the [Prerequisites](#prerequisites) section, then verify:
```bash
ffmpeg -version  # Should print version info, not an error
```

### `CUDA available: False` (NVIDIA GPU)

**Possible causes and fixes:**

1. **Wrong PyTorch variant:** You may have installed the CPU-only PyTorch. Reinstall with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
   ```

2. **Outdated GPU driver:** Update your NVIDIA driver to the latest version from [nvidia.com/drivers](https://www.nvidia.com/drivers).

3. **CUDA/Driver version mismatch:** Ensure your NVIDIA driver supports the CUDA version you installed. Check compatibility at [NVIDIA CUDA Compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions).

4. **No NVIDIA GPU:** If you are on a laptop with dual GPUs (Intel + NVIDIA), ensure PyTorch is using the NVIDIA GPU, not the integrated Intel GPU.

### `onnxruntime-gpu` does not show `CUDAExecutionProvider`

**Cause:** The installed `onnxruntime-gpu` version may not match your CUDA version, or there is a conflict with the CPU variant.

**Fix:**
```bash
# Remove all onnxruntime variants
pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml onnxruntime-silicon -y

# Reinstall the correct variant
pip install onnxruntime-gpu

# Verify
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### `OSError: [Errno 28] No space left on device`

**Cause:** Model files can be large (100 MB to 1.5 GB each). Ensure your model directory has sufficient disk space.

**Fix:**
```bash
# Check available space
df -h /tmp/vsep-models  # Default model directory

# Or set a custom model directory with more space
export VSEP_MODEL_DIR=/path/to/large/disk
python utils/cli.py song.mp3
```

### Download hangs or is very slow

**Fix:** Try increasing parallel download workers and using a mirror:

```python
import config.variables as cfg
cfg.MAX_DOWNLOAD_WORKERS = 8
cfg.DOWNLOAD_CHUNK_SIZE = 524288  # 512 KB
cfg.DOWNLOAD_TIMEOUT = 600        # 10 minutes
```

### `audioop` errors on Python 3.13+

**Cause:** The `audioop` module was deprecated and removed in Python 3.13.

**Fix:** Install the backport:
```bash
pip install audioop-lts
```
This is handled automatically by `requirements.txt` for Python 3.13+.

---

## Updating

```bash
cd /path/to/vsep

# Pull the latest changes
git pull origin main

# Update all dependencies
pip install -r requirements.txt --upgrade
```

> **Warning:** If you are using a GPU variant of onnxruntime, you may need to reinstall it after updating, as `pip install -r requirements.txt` will install the CPU variant by default.

---

## Uninstalling

```bash
# Remove all vsep-related packages
pip uninstall -y onnx onnx onnxruntime onnxruntime-gpu librosa pydub soundfile tqdm

# Remove PyTorch (if not needed by other projects)
pip uninstall -y torch torchvision torchaudio

# Delete the project directory
rm -rf /path/to/vsep

# Delete cached models (default location)
rm -rf /tmp/vsep-models
```

---

For more information, see the main [README.md](README.md) or [CONTRIBUTING.md](CONTRIBUTING.md).
