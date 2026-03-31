# vsep API Reference

> Comprehensive reference for the vsep audio stem separation library. This document covers the Python API, CLI interface, configuration system, ensemble system, remote API client, and architecture comparison.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Separator Class](#2-separator-class)
3. [CLI Reference](#3-cli-reference)
4. [Configuration Reference](#4-configuration-reference)
5. [Ensemble System](#5-ensemble-system)
6. [Remote API Client](#6-remote-api-client)
7. [Architecture Comparison](#7-architecture-comparison)

---

## 1. Overview

vsep is an AI-powered audio stem separator that supports multiple neural network architectures for splitting audio into individual components such as vocals, drums, bass, and other instruments. It provides both a Python API for programmatic use and a command-line interface for batch processing.

### Supported Architectures

| Architecture | Description | Model Format | Backend |
|---|---|---|---|
| **MDX-Net** | Open-unmix based architecture using multi-band decomposition | `.onnx` | ONNX Runtime |
| **VR Band Split** | Vision-Roadmap band-split RNN model | `.onnx` | ONNX Runtime |
| **Demucs v4** | Facebook Research hybrid transformer model (v4 only) | `.th` + `.yaml` | PyTorch |
| **MDXC / Roformer** | MDX23C and Roformer attention-based models | `.ckpt` + `.yaml` | PyTorch |

### Quick Start

```python
from separator import Separator

# Create a separator instance with default model
separator = Separator(output_dir="./output", output_format="FLAC")

# Load the default model (downloads automatically on first use)
separator.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")

# Separate an audio file
output_files = separator.separate("song.mp3")
print(f"Output files: {output_files}")
```

```bash
# CLI quick start
python utils/cli.py song.mp3 -m model_bs_roformer_ep_317_sdr_12.9755.ckpt --output_format FLAC
```

For more details on installation and usage, see the [main README](../README.md).

---

## 2. Separator Class

The `Separator` class is the primary entry point for audio separation. It manages model loading, hardware configuration, and the separation pipeline.

**Import:**

```python
from separator import Separator
```

### 2.1 Constructor

```python
Separator(
    log_level=logging.INFO,
    log_formatter=None,
    model_file_dir="/tmp/audio-separator-models/",
    output_dir=None,
    output_format="WAV",
    output_bitrate=None,
    normalization_threshold=0.9,
    amplification_threshold=0.0,
    output_single_stem=None,
    invert_using_spec=False,
    sample_rate=44100,
    use_soundfile=False,
    use_autocast=False,
    use_directml=False,
    chunk_duration=None,
    mdx_params={...},
    vr_params={...},
    demucs_params={...},
    mdxc_params={...},
    ensemble_algorithm=None,
    ensemble_weights=None,
    ensemble_preset=None,
    info_only=False,
)
```

#### Common Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `log_level` | `int` | `logging.INFO` | Logging level (e.g., `logging.DEBUG`, `logging.INFO`, `logging.WARNING`) |
| `log_formatter` | `logging.Formatter` | `None` | Custom log formatter. If `None`, uses `%(asctime)s - %(levelname)s - %(module)s - %(message)s` |
| `model_file_dir` | `str` | `"/tmp/audio-separator-models/"` | Directory where model files are stored. Overridden by `VSEP_MODEL_DIR` or `AUDIO_SEPARATOR_MODEL_DIR` environment variable if set |
| `output_dir` | `str` or `None` | `None` | Directory for output files. If `None`, uses the current working directory |
| `output_format` | `str` | `"WAV"` | Output audio format. Common values: `"WAV"`, `"FLAC"`, `"MP3"`, `"OGG"` |
| `output_bitrate` | `str` or `None` | `None` | Output bitrate for lossy formats (e.g., `"320k"` for MP3). Only used when `output_format` is a lossy format |
| `normalization_threshold` | `float` | `0.9` | Max peak amplitude to normalize audio to. Must be in range `(0, 1]` |
| `amplification_threshold` | `float` | `0.0` | Min peak amplitude to amplify audio to. Must be in range `[0, 1]`. Disabled by default |
| `output_single_stem` | `str` or `None` | `None` | If set, only output this stem (e.g., `"Instrumental"`, `"Vocals"`, `"Drums"`) |
| `invert_using_spec` | `bool` | `False` | If `True`, invert the secondary stem using spectrogram instead of waveform. Slightly slower but may improve quality |
| `sample_rate` | `int` | `44100` | Output sample rate in Hz. Must be a positive integer less than 12,800,000 |
| `use_soundfile` | `bool` | `False` | If `True`, use `soundfile` for audio writing instead of `pydub`. Can help with OOM issues |
| `use_autocast` | `bool` | `False` | If `True`, use PyTorch autocast for faster inference. Do not use for CPU inference |
| `use_directml` | `bool` | `False` | If `True`, attempt to use DirectML acceleration (Windows only, requires `torch_directml` package) |
| `chunk_duration` | `float` or `None` | `None` | Split audio into chunks of this duration in seconds. Chunks are concatenated without overlap/crossfade. Useful for processing very long audio files on systems with limited memory |
| `ensemble_algorithm` | `str` or `None` | `None` | Algorithm for ensembling multiple models. Defaults to `"avg_wave"` if not set. See [Ensemble System](#5-ensemble-system) for all options |
| `ensemble_weights` | `list` or `None` | `None` | Per-model weights for ensembling. Must match the number of models. Equal weights used if `None` |
| `ensemble_preset` | `str` or `None` | `None` | Named ensemble preset (e.g., `"vocal_balanced"`). Presets define models, algorithm, and optional weights |
| `info_only` | `bool` | `False` | If `True`, skip hardware setup and initialization logging. Useful for listing models without loading GPU |

#### Architecture-Specific Parameters

These parameters are passed as dictionaries to the constructor:

**MDX Parameters** (`mdx_params`)

| Key | Type | Default | Description |
|---|---|---|---|
| `hop_length` | `int` | `1024` | Hop length for STFT. Usually called stride in neural networks |
| `segment_size` | `int` | `256` | Segment size for processing. Larger consumes more resources but may give better results |
| `overlap` | `float` | `0.25` | Amount of overlap between prediction windows, range `0.001-0.999`. Higher is better but slower |
| `batch_size` | `int` | `1` | Batch size for processing. Larger consumes more RAM but may be slightly faster |
| `enable_denoise` | `bool` | `False` | Enable denoising during separation |

**VR Parameters** (`vr_params`)

| Key | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | `1` | Number of batches to process at a time |
| `window_size` | `int` | `512` | Window size. Balance quality and speed: `1024` = fast but lower quality, `320` = slower but better |
| `aggression` | `int` | `5` | Intensity of primary stem extraction, range `-100` to `100`. Typically `5` for vocals and instrumentals |
| `enable_tta` | `bool` | `False` | Enable Test-Time Augmentation. Slow but improves quality |
| `enable_post_process` | `bool` | `False` | Identify leftover artifacts within vocal output. May improve separation for some songs |
| `post_process_threshold` | `float` | `0.2` | Threshold for post-processing feature, range `0.1-0.3` |
| `high_end_process` | `bool` | `False` | Mirror the missing frequency range of the output |

**Demucs Parameters** (`demucs_params`)

| Key | Type | Default | Description |
|---|---|---|---|
| `segment_size` | `str` | `"Default"` | Size of segments for processing, range `1-100`. Higher = slower but better quality |
| `shifts` | `int` | `2` | Number of predictions with random shifts. Higher = slower but better quality |
| `overlap` | `float` | `0.25` | Overlap between prediction windows, range `0.001-0.999`. Higher = slower but better quality |
| `segments_enabled` | `bool` | `True` | Enable segment-wise processing |

**MDXC Parameters** (`mdxc_params`)

| Key | Type | Default | Description |
|---|---|---|---|
| `segment_size` | `int` | `256` | Segment size for processing. Larger consumes more resources but may give better results |
| `override_model_segment_size` | `bool` | `False` | Override the model's default segment size instead of using the value stored in the model config |
| `batch_size` | `int` | `1` | Batch size for processing. Larger consumes more RAM but may be slightly faster |
| `overlap` | `int` | `8` | Overlap between prediction windows, range `2-50`. Higher is better but slower |
| `pitch_shift` | `int` | `0` | Shift audio pitch by this many semitones while processing. May improve output for deep/high vocals |

### 2.2 Environment Variables

| Variable | Description |
|---|---|
| `VSEP_MODEL_DIR` | Override `model_file_dir` parameter. Path to model storage directory |
| `AUDIO_SEPARATOR_MODEL_DIR` | Legacy equivalent of `VSEP_MODEL_DIR` (still supported) |

### 2.3 Public Methods

#### `separate()`

Perform audio source separation on one or more audio files.

```python
def separate(self, audio_file_path, custom_output_names=None) -> list[str]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_file_path` | `str` or `list[str]` | required | Path to an audio file, a directory of audio files, or a list of paths |
| `custom_output_names` | `dict[str, str]` or `None` | `None` | Mapping of stem names to custom output filenames (e.g., `{"Vocals": "my_vocals"}`) |

**Returns:** `list[str]` -- List of file paths to the separated audio stem files.

**Raises:** `ValueError` if model not loaded or initialization failed.

**Supported audio formats:** `.wav`, `.flac`, `.mp3`, `.ogg`, `.opus`, `.m4a`, `.aiff`, `.ac3`

When `audio_file_path` is a directory, all audio files within it (recursively) are processed.

When `chunk_duration` is set and the file exceeds that duration, the audio is automatically split into chunks, processed separately, and merged back together.

```python
# Separate a single file
output_files = separator.separate("input/song.mp3")

# Separate multiple files
output_files = separator.separate(["song1.mp3", "song2.flac"])

# Separate a directory of audio files
output_files = separator.separate("input/album/")

# With custom output names
output_files = separator.separate(
    "song.mp3",
    custom_output_names={"Vocals": "lead_vocal", "Instrumental": "backing_track"}
)
```

#### `load_model()`

Download (if needed) and load a separation model into memory.

```python
def load_model(self, model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt") -> None
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_filename` | `str` or `list[str]` | `"model_bs_roformer_ep_317_sdr_12.9755.ckpt"` | Model filename or list of model filenames for ensembling |

**Returns:** `None`

**Raises:** `ValueError` if model file not found in supported models list, or model type not supported. `Exception` if using Demucs with Python < 3.10.

When a list of filenames is provided (more than one model), the separator operates in ensemble mode. Each model is loaded and run sequentially, and the results are combined using the configured ensemble algorithm.

If an ensemble preset was configured and no explicit model list is given, the preset's models are used automatically.

```python
# Load default model
separator.load_model()

# Load a specific model
separator.load_model("MDX23C-8KFFT-InstVoc_HQ.ckpt")

# Load multiple models for ensembling
separator.load_model([
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "MDX23C-8KFFT-InstVoc_HQ.ckpt"
])
```

#### `download_model_and_data()`

Download model files and associated data without loading the model into memory.

```python
def download_model_and_data(self, model_filename) -> None
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_filename` | `str` | required | Filename of the model to download |

**Returns:** `None`

```python
# Pre-download a model for later use
separator.download_model_and_data("htdemucs_ft.yaml")
```

#### `download_model_files()`

Download the model files for a given model filename.

```python
def download_model_files(self, model_filename) -> tuple
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_filename` | `str` | required | Filename of the model to download |

**Returns:** `tuple[str, str, str, str, str or None]` -- A tuple of `(model_filename, model_type, model_friendly_name, model_path, yaml_config_filename)`.

- `model_type` is one of `"MDX"`, `"VR"`, `"Demucs"`, or `"MDXC"`
- `yaml_config_filename` is `None` for MDX and VR models (which use hash-based parameter lookup)

**Raises:** `ValueError` if the model filename is not found in the supported model list.

Files are downloaded in parallel (up to 4 concurrent workers) with automatic fallback to an alternate repository if the primary source fails.

#### `list_supported_model_files()`

List all supported model files with performance scores and download information.

```python
def list_supported_model_files(self) -> dict
```

**Returns:** `dict` -- A nested dictionary grouped by architecture type. Each model entry contains:

| Key | Type | Description |
|---|---|---|
| `filename` | `str` | Primary model filename |
| `scores` | `dict` | Performance scores (SDR, SIR, SAR, ISR) per stem |
| `stems` | `list[str]` | List of output stems this model produces |
| `target_stem` | `str` | The primary target stem |
| `download_files` | `list[str]` | List of filenames or URLs to download |

The returned dict is keyed by architecture: `"VR"`, `"MDX"`, `"Demucs"`, `"MDXC"`.

```python
models = separator.list_supported_model_files()
for arch_type, arch_models in models.items():
    print(f"\n{arch_type} Models:")
    for name, info in arch_models.items():
        print(f"  {name}: {info['filename']}")
        if info['scores']:
            for stem, scores in info['scores'].items():
                print(f"    {stem} SDR: {scores.get('SDR', 'N/A')}")
```

#### `list_ensemble_presets()`

List all available ensemble presets.

```python
def list_ensemble_presets(self) -> dict
```

**Returns:** `dict` -- A dictionary mapping preset IDs to their full preset data. Each preset contains `name`, `description`, `models`, `algorithm`, `weights` (optional), and `contributor`.

```python
presets = separator.list_ensemble_presets()
for preset_id, preset in presets.items():
    print(f"{preset_id}: {preset['description']} "
          f"({len(preset['models'])} models, algorithm: {preset['algorithm']})")
```

#### `get_model_hash()`

Calculate the MD5 hash of a model file.

```python
def get_model_hash(self, model_path) -> str
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | required | Path to the model file |

**Returns:** `str` -- The MD5 hash of the model file (hex digest).

For files larger than 10 MB, only the last 10 MB are hashed (seeking to the end minus 10 MB). This is the same hashing strategy used by UVR to identify model parameters.

**Raises:** `FileNotFoundError` if the model file does not exist.

```python
hash_value = separator.get_model_hash("/tmp/vsep-models/model.ckpt")
print(f"Model hash: {hash_value}")
```

#### `setup_accelerated_inferencing_device()`

Configure hardware acceleration for PyTorch and ONNX Runtime.

```python
def setup_accelerated_inferencing_device(self) -> None
```

**Returns:** `None`

This method is called automatically during initialization (unless `info_only=True`). It probes the system for available acceleration backends in this order:

1. **CUDA** (NVIDIA GPU) -- sets `torch_device` to `"cuda"` and ONNX provider to `CUDAExecutionProvider`
2. **MPS/CoreML** (Apple Silicon, ARM only) -- sets `torch_device` to `"mps"` and ONNX provider to `CoreMLExecutionProvider`
3. **DirectML** (Windows, if `use_directml=True`) -- sets `torch_device` to DirectML device and ONNX provider to `DmlExecutionProvider`
4. **CPU** (fallback) -- sets `torch_device` to `"cpu"` and ONNX provider to `CPUExecutionProvider`

After calling this method, `self.torch_device` and `self.onnx_execution_provider` are populated with the selected device and provider.

#### `get_simplified_model_list()`

Return a simplified, user-friendly model list with sorting and filtering.

```python
def get_simplified_model_list(self, filter_sort_by=None) -> dict
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filter_sort_by` | `str` or `None` | `None` | Sort/filter criteria: `"name"`, `"filename"`, or a stem name like `"vocals"`, `"drums"`, etc. |

**Returns:** `dict` -- Dictionary keyed by model filename, with values containing `Name`, `Type`, `Stems` (with SDR scores), and `SDR` dictionary.

When filtering by a stem name, only models that produce that stem are returned, sorted by SDR score (highest first).

```python
# Get all models sorted by name
models = separator.get_simplified_model_list(filter_sort_by="name")

# Get only vocal models sorted by vocal SDR
vocal_models = separator.get_simplified_model_list(filter_sort_by="vocals")
```

### 2.4 Internal Methods

These methods are used internally but may be useful for advanced users:

| Method | Description |
|---|---|
| `get_system_info()` | Log and return system information (OS, CPU, Python version) |
| `check_ffmpeg_installed()` | Verify FFmpeg is installed and log its version |
| `log_onnxruntime_packages()` | Log installed ONNX Runtime packages (GPU, Silicon, CPU, DirectML) |
| `get_package_distribution(package_name)` | Return package distribution object if installed, `None` otherwise |
| `download_file_if_not_exists(url, output_path)` | Download a file with resume support and progress bar |
| `load_model_data_from_yaml(yaml_config_filename)` | Load model parameters from a YAML config file |
| `load_model_data_using_hash(model_path)` | Load model parameters by computing file hash and looking up in UVR data |

---

## 3. CLI Reference

The CLI is invoked via `python utils/cli.py` and provides full access to all separation capabilities from the command line.

```
usage: vsep [-h] [audio_files ...]
```

### 3.1 Info and Debugging

| Argument | Short | Type | Default | Description |
|---|---|---|---|
| `--version` | `-v` | flag | -- | Show program version number and exit |
| `--debug` | `-d` | flag | `False` | Enable debug logging (equivalent to `--log_level=debug`) |
| `--env_info` | `-e` | flag | `False` | Print environment information and exit |
| `--list_models` | `-l` | flag | `False` | List all supported models and exit |
| `--log_level` | -- | `str` | `"info"` | Log level: `info`, `debug`, `warning` |
| `--list_filter` | -- | `str` | `None` | Filter/sort model list by `name`, `filename`, or any stem name |
| `--list_limit` | -- | `int` | `None` | Limit the number of models shown |
| `--list_format` | -- | `str` | `"pretty"` | Format for listing models: `pretty` or `json` |

```bash
# List all models as JSON
python utils/cli.py --list_models --list_format json

# Show top 10 vocal models sorted by SDR
python utils/cli.py --list_models --list_filter vocals --list_limit 10

# Print environment info (GPU, ONNX Runtime, etc.)
python utils/cli.py --env_info
```

### 3.2 Separation I/O

| Argument | Short | Type | Default | Description |
|---|---|---|---|
| `--model_filename` | `-m` | `str` | `"model_bs_roformer_ep_317_sdr_12.9755.ckpt"` | Primary model to use for separation |
| `--extra_models` | -- | `list[str]` | `None` | Additional models for ensembling. Requires `-m` for the primary model |
| `--output_format` | -- | `str` | `"FLAC"` | Output format: `WAV`, `FLAC`, `MP3`, `OGG`, etc. |
| `--output_bitrate` | -- | `str` | `None` | Output bitrate for lossy formats (e.g., `320k`) |
| `--output_dir` | -- | `str` | `None` | Output directory (default: current directory) |
| `--model_file_dir` | -- | `str` | `"/tmp/vsep-models/"` | Model files directory (overridden by `VSEP_MODEL_DIR` env var) |
| `--download_model_only` | -- | flag | `False` | Download model only, without performing separation |

```bash
# Download a model for later use
python utils/cli.py --download_model_only -m MDX23C-8KFFT-InstVoc_HQ.ckpt

# Specify output directory and format
python utils/cli.py song.mp3 -m model.ckpt --output_dir ./separated --output_format MP3 --output_bitrate 320k
```

### 3.3 Common Separation Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--invert_spect` | flag | `False` | Invert secondary stem using spectrogram |
| `--normalization` | `float` | `0.9` | Max peak amplitude for normalization (range: `0-1`) |
| `--amplification` | `float` | `0.0` | Min peak amplitude for amplification (range: `0-1`) |
| `--single_stem` | `str` | `None` | Output only a single stem: `Instrumental`, `Vocals`, `Drums`, `Bass`, `Guitar`, `Piano`, `Other` |
| `--sample_rate` | `int` | `44100` | Output sample rate in Hz |
| `--use_soundfile` | flag | `False` | Use soundfile for audio output (can help with OOM) |
| `--use_autocast` | flag | `False` | Use PyTorch autocast for faster inference (GPU only) |
| `--chunk_duration` | `float` | `None` | Split into chunks of N seconds (e.g., `600` for 10-min chunks) |
| `--ensemble_algorithm` | `str` | `None` | Ensemble algorithm. Choices: `avg_wave`, `median_wave`, `min_wave`, `max_wave`, `avg_fft`, `median_fft`, `min_fft`, `max_fft`, `uvr_max_spec`, `uvr_min_spec`, `ensemble_wav` |
| `--ensemble_weights` | `list[float]` | `None` | Per-model weights for ensembling (must match model count) |
| `--ensemble_preset` | `str` | `None` | Named ensemble preset (e.g., `vocal_balanced`, `karaoke`) |
| `--list_presets` | flag | `False` | List all available ensemble presets and exit |
| `--custom_output_names` | `JSON str` | `None` | Custom names for output files (e.g., `'{"Vocals": "my_vocals"}'`) |

```bash
# Extract only vocals
python utils/cli.py song.mp3 --single_stem Vocals

# Use ensemble with preset
python utils/cli.py song.mp3 --ensemble_preset vocal_balanced

# Use ensemble with custom models and weights
python utils/cli.py song.mp3 -m model1.ckpt --extra_models model2.onnx model3.ckpt \
    --ensemble_algorithm avg_wave --ensemble_weights 1.0 0.5 0.5
```

### 3.4 MDX Architecture Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mdx_segment_size` | `int` | `256` | Segment size. Larger = more resources, potentially better results |
| `--mdx_overlap` | `float` | `0.25` | Overlap between prediction windows (`0.001-0.999`). Higher = slower but better |
| `--mdx_batch_size` | `int` | `1` | Batch size. Larger = more RAM, slightly faster |
| `--mdx_hop_length` | `int` | `1024` | Hop length (stride). Only change if you know what you are doing |
| `--mdx_enable_denoise` | flag | `False` | Enable denoising during separation |

```bash
python utils/cli.py song.mp3 -m MDX-Net_Model.onnx --mdx_segment_size 512 --mdx_enable_denoise
```

### 3.5 VR Architecture Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--vr_batch_size` | `int` | `1` | Batches to process at a time |
| `--vr_window_size` | `int` | `512` | Window size: `1024` = fast/low quality, `320` = slow/high quality |
| `--vr_aggression` | `int` | `5` | Intensity of primary stem extraction (`-100` to `100`) |
| `--vr_enable_tta` | flag | `False` | Enable Test-Time Augmentation (slow but better quality) |
| `--vr_high_end_process` | flag | `False` | Mirror the missing frequency range |
| `--vr_enable_post_process` | flag | `False` | Identify leftover artifacts in vocal output |
| `--vr_post_process_threshold` | `float` | `0.2` | Post-process threshold (`0.1-0.3`) |

```bash
python utils/cli.py song.mp3 -m VR_Model.onnx --vr_aggression 2 --vr_window_size 320
```

### 3.6 Demucs Architecture Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--demucs_segment_size` | `str` | `"Default"` | Segment size (`1-100`). Higher = slower but better quality |
| `--demucs_shifts` | `int` | `2` | Number of predictions with random shifts. Higher = slower but better |
| `--demucs_overlap` | `float` | `0.25` | Overlap between prediction windows (`0.001-0.999`) |
| `--demucs_segments_enabled` | `bool` | `True` | Enable segment-wise processing |

```bash
python utils/cli.py song.mp3 -m htdemucs_ft.yaml --demucs_shifts 4 --demucs_overlap 0.35
```

### 3.7 MDXC Architecture Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mdxc_segment_size` | `int` | `256` | Segment size. Larger = more resources, potentially better results |
| `--mdxc_override_model_segment_size` | flag | `False` | Override the model's built-in segment size |
| `--mdxc_overlap` | `int` | `8` | Overlap between prediction windows (`2-50`). Higher = better but slower |
| `--mdxc_batch_size` | `int` | `1` | Batch size. Larger = more RAM, slightly faster |
| `--mdxc_pitch_shift` | `int` | `0` | Shift pitch by N semitones. May help with deep/high vocals |

```bash
python utils/cli.py song.mp3 -m MDX23C-8KFFT-InstVoc_HQ.ckpt --mdxc_pitch_shift 2 --mdxc_overlap 16
```

---

## 4. Configuration Reference

Configuration is defined in `config/variables.py` and re-exported via `config/__init__.py`.

### 4.1 Repository URLs

| Variable | Type | Value | Description |
|---|---|---|---|
| `UVR_PUBLIC_REPO_URL` | `str` | `"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"` | Primary UVR model repository (public models) |
| `UVR_VIP_REPO_URL` | `str` | `"https://github.com/Anjok0109/ai_magic/releases/download/v5"` | VIP models repository (Anjok07's paid subscriber models) |
| `AUDIO_SEPARATOR_REPO_URL` | `str` | `"https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"` | vsep-specific models and config fallback repository |
| `UVR_MODEL_DATA_URL_PREFIX` | `str` | `"https://raw.githubusercontent.com/TRvlvr/application_data/main"` | Base URL for UVR model parameter data files |
| `UVR_VR_MODEL_DATA_URL` | `str` | `"{prefix}/vr_model_data/model_data_new.json"` | URL for VR model parameter lookup data |
| `UVR_MDX_MODEL_DATA_URL` | `str` | `"{prefix}/mdx_model_data/model_data_new.json"` | URL for MDX model parameter lookup data |

### 4.2 Model Path Mappings

| Variable | Type | Value | Description |
|---|---|---|---|
| `MDXC_YAML_PATH_PREFIX` | `str` | `"mdx_model_data/mdx_c_configs"` | Path prefix for MDXC YAML config files within the UVR repository |

### 4.3 Download Configuration

| Variable | Type | Value | Description |
|---|---|---|---|
| `MAX_DOWNLOAD_WORKERS` | `int` | `4` | Maximum number of parallel download threads |
| `DOWNLOAD_CHUNK_SIZE` | `int` | `262144` | Download chunk size in bytes (256 KB) |
| `DOWNLOAD_TIMEOUT` | `int` | `300` | HTTP request timeout in seconds (5 minutes) |
| `HTTP_POOL_CONNECTIONS` | `int` | `10` | Number of connection pool connections |
| `HTTP_POOL_MAXSIZE` | `int` | `10` | Maximum size of the connection pool |

### 4.4 Model Data Files

| Variable | Type | Value | Description |
|---|---|---|---|
| `VR_MODEL_DATA_FILENAME` | `str` | `"vr_model_data.json"` | Local filename for VR model parameter data |
| `MDX_MODEL_DATA_FILENAME` | `str` | `"mdx_model_data.json"` | Local filename for MDX model parameter data |

### 4.5 Helper Functions

#### `get_repo_url(is_vip=False)`

Returns the appropriate repository URL based on model type.

```python
from config import get_repo_url

# Public model URL
url = get_repo_url(is_vip=False)
# "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"

# VIP model URL
url = get_repo_url(is_vip=True)
# "https://github.com/Anjok0109/ai_magic/releases/download/v5"
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `is_vip` | `bool` | `False` | If `True`, return the VIP repository URL |

**Returns:** `str` -- The repository URL.

#### `get_mdx_yaml_url(filename)`

Constructs the full URL for an MDXC YAML config file.

```python
from config import get_mdx_yaml_url

url = get_mdx_yaml_url("model_2_stem_full_band_8k.yaml")
# "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/mdx_model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml"
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` | required | YAML config filename |

**Returns:** `str` -- Full URL to the YAML file.

#### `get_fallback_url(filename)`

Returns the fallback URL from the audio-separator repository.

```python
from config import get_fallback_url

url = get_fallback_url("some_model.onnx")
# "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/some_model.onnx"
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` | required | Model or config filename |

**Returns:** `str` -- Fallback URL string.

---

## 5. Ensemble System

The ensemble system allows combining outputs from multiple separation models to produce higher-quality results. Instead of relying on a single model, you can run several models on the same audio and merge their outputs using a configurable algorithm.

### 5.1 How Ensembling Works

1. Multiple models are specified (either explicitly or via a preset)
2. Each model separates the audio independently
3. The intermediate stems are grouped by stem name (e.g., all "Vocals" outputs together)
4. The grouped waveforms are merged using the selected ensemble algorithm
5. The final merged stems are written as output

### 5.2 Ensemble Algorithms

All 11 supported ensemble algorithms are defined in the `Ensembler` class (`separator/ensembler.py`):

| Algorithm | Category | Description | Supports Weights |
|---|---|---|---|
| `avg_wave` | Wave | Weighted average of waveforms in the time domain | Yes |
| `median_wave` | Wave | Median of waveforms (robust to outliers) | No (ignored) |
| `min_wave` | Wave | Element-wise minimum absolute amplitude (conservative) | No (ignored) |
| `max_wave` | Wave | Element-wise maximum absolute amplitude (aggressive) | No (ignored) |
| `avg_fft` | FFT | Weighted average of spectrograms (frequency domain) | Yes |
| `median_fft` | FFT | Median of spectrogram magnitudes (complex median) | No (ignored) |
| `min_fft` | FFT | Minimum magnitude spectrogram (conservative in frequency domain) | No (ignored) |
| `max_fft` | FFT | Maximum magnitude spectrogram (aggressive in frequency domain) | No (ignored) |
| `uvr_max_spec` | UVR | UVR's maximum spectrogram ensembling algorithm | No (via UVR) |
| `uvr_min_spec` | UVR | UVR's minimum spectrogram ensembling algorithm | No (via UVR) |
| `ensemble_wav` | UVR | UVR's legacy waveform ensembling algorithm | No (via UVR) |

**Default algorithm:** `avg_wave`

#### Wave Domain Algorithms

Wave domain algorithms operate directly on the audio waveform samples. They are generally faster than FFT-based methods because they avoid the STFT/ISTFT transform overhead.

- **`avg_wave`**: Computes a weighted sum of all model waveforms and divides by the total weight. This is the most commonly used algorithm and produces a balanced blend of all model outputs.
- **`median_wave`**: Takes the median value at each sample position across all models. This is robust to outlier models that produce very different results from the consensus.
- **`min_wave`**: At each sample position, selects the waveform with the smallest absolute amplitude. This produces a conservative output that retains only what all models agree on.
- **`max_wave`**: At each sample position, selects the waveform with the largest absolute amplitude. This is aggressive and can retain more detail but may include artifacts.

#### FFT Domain Algorithms

FFT domain algorithms transform waveforms to the frequency domain via STFT, perform the ensemble operation on the complex spectrograms, then convert back via ISTFT. These can be more musically accurate because they preserve phase relationships.

- **`avg_fft`**: Weighted average of complex spectrograms. Phase-aware blending.
- **`median_fft`**: Median of real and imaginary parts of the spectrograms separately.
- **`min_fft`**: Selects the spectrogram bin with the minimum magnitude at each frequency/time position.
- **`max_fft`**: Selects the spectrogram bin with the maximum magnitude at each frequency/time position.

#### UVR Algorithms

These algorithms use UVR's built-in spectrogram utilities for ensembling, originally from the Ultimate Vocal Remover project.

- **`uvr_max_spec`**: Maximum spectrogram ensembling using UVR's `MAX_SPEC` algorithm.
- **`uvr_min_spec`**: Minimum spectrogram ensembling using UVR's `MIN_SPEC` algorithm.
- **`ensemble_wav`**: Legacy UVR waveform ensembling.

### 5.3 Weights

Weights allow you to give more influence to certain models in the ensemble. Weights are only supported by `avg_wave` and `avg_fft` algorithms.

```python
# Give model1 double the influence of model2
separator = Separator(
    ensemble_algorithm="avg_wave",
    ensemble_weights=[2.0, 1.0]
)
separator.load_model(["model1.ckpt", "model2.ckpt"])
```

```bash
# CLI equivalent
python utils/cli.py song.mp3 -m model1.ckpt --extra_models model2.ckpt \
    --ensemble_algorithm avg_wave --ensemble_weights 2.0 1.0
```

If weights are not specified, equal weights (all `1.0`) are used. If the number of weights does not match the number of models, a warning is logged and equal weights are used instead. Non-finite weights (NaN, infinity) or weights summing to zero also trigger a fallback to equal weights.

### 5.4 Ensemble Presets

Presets bundle a curated selection of models with an appropriate algorithm and optional weights. Presets are defined in `ensemble_presets.json` (bundled with the package).

```python
# Use a preset
separator = Separator(ensemble_preset="vocal_balanced")
separator.load_model()  # Loads the preset's models automatically
output_files = separator.separate("song.mp3")
```

```bash
# List available presets
python utils/cli.py --list_presets

# Use a preset
python utils/cli.py song.mp3 --ensemble_preset vocal_balanced
```

Each preset contains:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | Yes | Human-readable preset name |
| `description` | `str` | Yes | Brief description of the preset's purpose |
| `models` | `list[str]` | Yes | List of model filenames (minimum 2) |
| `algorithm` | `str` | Yes | Ensemble algorithm to use |
| `weights` | `list[float]` or `None` | No | Optional per-model weights |
| `contributor` | `str` | No | Preset author attribution |

Explicit user arguments always take priority over preset defaults. If you specify `--ensemble_algorithm` alongside `--ensemble_preset`, your explicit algorithm is used.

---

## 6. Remote API Client

The `AudioSeparatorAPIClient` class provides a Python client for interacting with a remotely deployed vsep API server. This enables offloading separation work to a remote machine with GPU resources.

**Import:**

```python
from remote.api_client import AudioSeparatorAPIClient
```

### 6.1 Constructor

```python
AudioSeparatorAPIClient(api_url: str, logger: logging.Logger)
```

| Parameter | Type | Description |
|---|---|---|
| `api_url` | `str` | Base URL of the remote vsep API server (e.g., `"https://api.example.com"`) |
| `logger` | `logging.Logger` | Logger instance for client-side logging |

The client maintains a persistent `requests.Session` for connection pooling.

### 6.2 Methods

#### `separate_audio_and_wait()`

Submit an audio separation job and wait for completion. This is the primary convenience method for most use cases, handling the full workflow: upload, poll, and download.

```python
def separate_audio_and_wait(
    self,
    file_path: str,
    model: Optional[str] = None,
    models: Optional[List[str]] = None,
    preset: Optional[str] = None,
    timeout: int = 600,
    poll_interval: int = 10,
    download: bool = True,
    output_dir: Optional[str] = None,
    output_format: str = "flac",
    output_bitrate: Optional[str] = None,
    normalization_threshold: float = 0.9,
    amplification_threshold: float = 0.0,
    output_single_stem: Optional[str] = None,
    invert_using_spec: bool = False,
    sample_rate: int = 44100,
    use_soundfile: bool = False,
    use_autocast: bool = False,
    custom_output_names: Optional[Dict[str, str]] = None,
    mdx_segment_size: int = 256,
    mdx_overlap: float = 0.25,
    mdx_batch_size: int = 1,
    mdx_hop_length: int = 1024,
    mdx_enable_denoise: bool = False,
    vr_batch_size: int = 1,
    vr_window_size: int = 512,
    vr_aggression: int = 5,
    vr_enable_tta: bool = False,
    vr_high_end_process: bool = False,
    vr_enable_post_process: bool = False,
    vr_post_process_threshold: float = 0.2,
    demucs_segment_size: str = "Default",
    demucs_shifts: int = 2,
    demucs_overlap: float = 0.25,
    demucs_segments_enabled: bool = True,
    mdxc_segment_size: int = 256,
    mdxc_override_model_segment_size: bool = False,
    mdxc_overlap: int = 8,
    mdxc_batch_size: int = 1,
    mdxc_pitch_shift: int = 0,
) -> dict
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str` | required | Path to the local audio file to upload |
| `model` | `str` or `None` | `None` | Single model filename |
| `models` | `list[str]` or `None` | `None` | List of model filenames for ensembling |
| `preset` | `str` or `None` | `None` | Ensemble preset name |
| `timeout` | `int` | `600` | Maximum wait time for job completion in seconds |
| `poll_interval` | `int` | `10` | Seconds between status polls |
| `download` | `bool` | `True` | Whether to automatically download result files |
| `output_dir` | `str` or `None` | `None` | Directory to save downloaded files |

Plus all architecture-specific parameters (same as the Separator class).

**Returns:** `dict` with keys:

| Key | Type | Description |
|---|---|---|
| `task_id` | `str` | The job task ID |
| `status` | `str` | `"completed"`, `"error"`, or `"timeout"` |
| `files` | `list` or `dict` | Output filenames (list) or hash-to-filename mapping (dict) |
| `downloaded_files` | `list[str]` | Local file paths of downloaded files (if `download=True`) |
| `error` | `str` | Error message (if status is `"error"` or `"timeout"`) |

```python
import logging
from remote.api_client import AudioSeparatorAPIClient

logger = logging.getLogger(__name__)
client = AudioSeparatorAPIClient("https://my-vsep-api.example.com", logger)

result = client.separate_audio_and_wait(
    file_path="song.mp3",
    model="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    output_format="flac",
    output_dir="./output",
    timeout=300,
)
print(f"Task {result['task_id']}: {result['status']}")
print(f"Files: {result.get('downloaded_files', [])}")
```

#### `separate_audio()`

Submit an audio separation job without waiting for completion (asynchronous).

```python
def separate_audio(self, file_path: str, ...) -> dict
```

Parameters are the same as `separate_audio_and_wait()` minus `timeout`, `poll_interval`, `download`, and `output_dir`.

**Returns:** `dict` with key `task_id` -- use this to poll status with `get_job_status()`.

```python
result = client.separate_audio("song.mp3", model="model.ckpt")
task_id = result["task_id"]
print(f"Job submitted: {task_id}")
# Later, check status:
status = client.get_job_status(task_id)
```

#### `get_job_status()`

Poll the status of a submitted job.

```python
def get_job_status(self, task_id: str) -> dict
```

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `str` | The task ID returned by `separate_audio()` |

**Returns:** `dict` with keys:

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"completed"`, `"processing"`, `"error"`, etc. |
| `progress` | `int` | Progress percentage (0-100) |
| `current_model_index` | `int` | Current model being processed (for ensembles) |
| `total_models` | `int` | Total models to process (for ensembles) |
| `files` | `list` or `dict` | Output files (when status is `"completed"`) |
| `error` | `str` | Error message (when status is `"error"`) |

#### `list_models()`

List available models on the remote server.

```python
def list_models(self, format_type: str = "pretty", filter_by: Optional[str] = None) -> dict
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `format_type` | `str` | `"pretty"` | `"pretty"` for formatted text, `"json"` for raw JSON |
| `filter_by` | `str` or `None` | `None` | Filter/sort criteria (same as CLI `--list_filter`) |

**Returns:** `dict` -- If `format_type="json"`, returns the parsed model list. If `"pretty"`, returns `{"text": "..."}` with the formatted output.

#### `download_file_by_hash()`

Download a result file from a completed job using its hash identifier.

```python
def download_file_by_hash(self, task_id: str, file_hash: str, filename: str, output_path: Optional[str] = None) -> str
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str` | required | The job task ID |
| `file_hash` | `str` | required | Hash identifier of the file (from `get_job_status` response) |
| `filename` | `str` | required | Original filename (used for local save path) |
| `output_path` | `str` or `None` | `None` | Local save path. If `None`, uses `filename` |

**Returns:** `str` -- The local file path of the downloaded file.

#### `download_file()`

Legacy method to download a file from a completed job by filename (backward compatibility).

```python
def download_file(self, task_id: str, filename: str, output_path: Optional[str] = None) -> str
```

This method URL-encodes the filename and uses it directly in the download URL. Prefer `download_file_by_hash()` for newer servers.

#### `get_server_version()`

Get the version string of the remote vsep API server.

```python
def get_server_version(self) -> str
```

**Returns:** `str` -- The server version (e.g., `"0.25.0"`) or `"unknown"` if unavailable.

```python
version = client.get_server_version()
print(f"Remote server version: {version}")
```

---

## 7. Architecture Comparison

The following table provides a detailed comparison of all four supported architectures.

| Feature | MDX-Net | VR Band Split | Demucs v4 | MDXC / Roformer |
|---|---|---|---|---|
| **Full Name** | Multi-Decoder X-Net | Vision-Roadmap Band Split RNN | Hybrid Transformer Demucs v4 | MDX23C / Roformer |
| **Backend** | ONNX Runtime | ONNX Runtime | PyTorch | PyTorch |
| **Model Format** | `.onnx` | `.onnx` | `.th` + `.yaml` | `.ckpt` + `.yaml` |
| **Parameter Lookup** | MD5 hash from UVR data | MD5 hash from UVR data | YAML config file | YAML config file |
| **Typical Output Stems** | 2 (vocals + instrumental) | 2 (vocals + instrumental) | 4 (vocals, drums, bass, other) | 2 (vocals + instrumental) or more |
| **Min Python Version** | 3.8+ | 3.8+ | 3.10+ | 3.8+ |
| **GPU Acceleration** | CUDA, CoreML, DirectML | CUDA, CoreML, DirectML | CUDA, MPS, DirectML | CUDA, MPS, DirectML |
| **Segment Size Param** | `mdx_segment_size` (int) | `vr_window_size` (int) | `demucs_segment_size` (str) | `mdxc_segment_size` (int) |
| **Default Segment Size** | 256 | 512 | "Default" | 256 |
| **Overlap Param** | `mdx_overlap` (float) | N/A | `demucs_overlap` (float) | `mdxc_overlap` (int) |
| **Default Overlap** | 0.25 | N/A | 0.25 | 8 |
| **Batch Size** | `mdx_batch_size` | `vr_batch_size` | N/A | `mdxc_batch_size` |
| **Default Batch Size** | 1 | 1 | N/A | 1 |
| **Denoise Support** | Yes (`--mdx_enable_denoise`) | No | No | No |
| **Pitch Shift** | No | No | No | Yes (`--mdxc_pitch_shift`) |
| **TTA Support** | No | Yes (`--vr_enable_tta`) | Yes (`--demucs_shifts`) | No |
| **Post-Processing** | No | Yes (`--vr_enable_post_process`) | No | No |
| **Autocast Support** | N/A (ONNX) | N/A (ONNX) | Yes | Yes |
| **High End Processing** | No | Yes (`--vr_high_end_process`) | No | No |
| **Aggression Control** | No | Yes (`--vr_aggression`, -100 to 100) | No | No |
| **Model Override Segment Size** | N/A | N/A | N/A | Yes (`--mdxc_override_model_segment_size`) |
| **Ensemble Support** | Yes | Yes | Yes | Yes |
| **Chunk Duration Support** | Yes | Yes | Yes | Yes |

### Choosing an Architecture

- **Best vocal quality**: Roformer models (MDXC architecture) typically achieve the highest SDR scores for vocal separation
- **Best multi-stem separation**: Demucs v4 is the only architecture that natively produces 4+ stems (vocals, drums, bass, other)
- **Fastest inference on CPU**: MDX-Net and VR models using ONNX Runtime are typically faster on CPU
- **Best on GPU**: Demucs v4 and MDXC/Roformer models benefit most from GPU acceleration via PyTorch
- **Fine-tuned control**: VR architecture offers the most tunable parameters (aggression, TTA, post-processing, high end processing)
- **Best overall balance**: MDXC with Roformer models (e.g., `model_bs_roformer_ep_317_sdr_12.9755.ckpt`) provides the best quality-to-speed ratio for 2-stem separation

### Model File Naming Conventions

| Pattern | Architecture | Example |
|---|---|---|
| `*.onnx` | MDX-Net or VR | `MDX-Net_Model.onnx`, `5_HP-Karaoke-UVR.pth.onnx` |
| `htdemucs*.yaml` | Demucs v4 | `htdemucs_ft.yaml`, `htdemucs.yaml` |
| `MDX23C-*.ckpt` | MDXC | `MDX23C-8KFFT-InstVoc_HQ.ckpt` |
| `model_*_roformer_*.ckpt` | MDXC (Roformer) | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` |
| `Mel-Band-Roformer*.ckpt` | MDXC (Mel-Band Roformer) | `Mel-Band-Roformer-Karaoke-Run1.ckpt` |
