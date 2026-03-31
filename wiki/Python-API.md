# Python API

## Basic Usage

```python
from separator import Separator

# Initialize with defaults
separator = Separator()

# Separate audio — returns list of output file paths
output_files = separator.separate("song.mp3")
print(output_files)
# ['song_(Vocals).flac', 'song_(Instrumental).flac']
```

## Constructor

```python
Separator(
    log_level=logging.INFO,
    log_formatter=None,
    model_file_dir="/tmp/audio-separator-models/",
    output_dir=None,              # defaults to cwd
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
    mdx_params=None,
    vr_params=None,
    demucs_params=None,
    mdxc_params=None,
    ensemble_algorithm=None,
    ensemble_weights=None,
    ensemble_preset=None,
    info_only=False,
)
```

| Parameter | Type | Default | Description |
|:---------|:-----|:-------|:-----------|
| `model_file_dir` | `str` | `/tmp/audio-separator-models/` | Directory for cached models |
| `output_dir` | `str` | `None` (cwd) | Output directory |
| `output_format` | `str` | `WAV` | Output format (WAV, FLAC, MP3, OGG) |
| `output_bitrate` | `str` | `None` | Bitrate for lossy formats |
| `normalization_threshold` | `float` | `0.9` | Peak normalization (0.0–1.0) |
| `amplification_threshold` | `float` | `0.0` | Min peak amplification |
| `output_single_stem` | `str` | `None` | Output only this stem name |
| `invert_using_spec` | `bool` | `False` | Invert secondary stem |
| `sample_rate` | `int` | `44100` | Output sample rate |
| `use_soundfile` | `bool` | `False` | Use soundfile for output |
| `use_autocast` | `bool` | `False` | FP16 autocast (GPU only) |
| `use_directml` | `bool` | `False` | DirectML for AMD/Intel |
| `chunk_duration` | `float` | `None` | Chunk duration in seconds |
| `mdx_params` | `dict` | Default (see config) | MDX architecture params |
| `vr_params` | `dict` | Default (see config) | VR architecture params |
| `demucs_params` | `dict` | Default (see config) | Demucs architecture params |
| `mdxc_params` | `dict` | Default (see config) | MDXC/Roformer params |
| `ensemble_algorithm` | `str` | `None` | Ensemble algorithm name |
| `ensemble_weights` | `list` | `None` | Per-model weights |
| `ensemble_preset` | `str` | `None` | Named ensemble preset |
| `info_only` | `bool` | `False` | Skip hardware setup (for listing) |

## Key Methods

### `separate(audio_file_path, custom_output_names=None)`

Main method. Separates audio and returns output file paths.

```python
# Single file
output_files = separator.separate("song.mp3")

# Multiple files
output_files = separator.separate(["song1.mp3", "song2.mp3"])

# Custom stem names
output_files = separator.separate(
    "song.mp3",
    custom_output_names={"Vocals": "lead_vocals", "Instrumental": "backing_track"}
)
```

**Parameters:**
- `audio_file_path` — Path or list of paths
- `custom_output_names` — Dict mapping stem names to output filenames

**Returns:** `list[str]` — List of output file paths

### `load_model(model_filename=...)`

Load a model for inference. Auto-detects architecture from filename.

```python
# Single model
separator.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")

# Ensemble (list of models)
separator.load_model(["model1.ckpt", "model2.onnx"])

# Use with preset
separator = Separator(ensemble_preset="vocals_ensemble")
separator.load_model()  # Uses preset models automatically
```

### `list_models(model_type=None, filter_stem=None)`

List all available models with optional filtering. Merges models from `models.json` and the UVR `download_checks.json`.

```python
# All models grouped by architecture
models = separator.list_models()
# {"VR": {...}, "MDX": {...}, "Demucs": {...}, "MDXC": {...}}

# Only Roformer models
models = separator.list_models(model_type="MDXC")

# Only vocal-producing models
models = separator.list_models(filter_stem="vocals")

# Both filters combined
models = separator.list_models(model_type="MDXC", filter_stem="vocals")
```

**Returns:** `dict` — `{model_type: {display_name: {filename, scores, stems, ...}}}`

### `get_simplified_model_list(filter_sort_by=None)`

Get a simplified flat dict of all models.

```python
# All models
models = separator.get_simplified_model_list()

# Sorted by name
models = separator.get_simplified_model_list(filter_sort_by="name")

# Sorted by vocal SDR (descending)
models = separator.get_simplified_model_list(filter_sort_by="vocals")

# Sorted by filename
models = separator.get_simplified_model_list(filter_sort_by="filename")
```

**Returns:** `dict` — `{filename: {Name, Type, Stems, SDR}}`

### `download_model_and_data(model_filename)`

Download a model without performing separation.

```python
separator.download_model_and_data("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
```

### `list_ensemble_presets()`

List all available named ensemble presets.

```python
presets = separator.list_ensemble_presets()
for name, config in presets.items():
    print(f"{name}: {config['description']} ({len(config['models'])} models)")
```

## Advanced Examples

### Custom Architecture Parameters

```python
from separator import Separator
import config.variables as cfg

# Download performance
cfg.MAX_DOWNLOAD_WORKERS = 8
cfg.DOWNLOAD_CHUNK_SIZE = 524288

separator = Separator(
    model_file_dir="./models",
    output_dir="./output",
    output_format="FLAC",
    normalization_threshold=0.95,
    use_autocast=True,
    mdx_params={
        "segment_size": 512,
        "overlap": 0.5,
        "batch_size": 4,
    },
    vr_params={
        "window_size": 320,
        "aggression": 10,
        "enable_tta": True,
    },
    mdxc_params={
        "segment_size": 512,
        "overlap": 12,
        "pitch_shift": 2,
    },
)

output_files = separator.separate("long_song.mp3")
```

### Chunking Long Audio

```python
separator = Separator(
    chunk_duration=300,  # 5-minute chunks
)
output_files = separator.separate("podcast_2h.mp3")
```

### Ensemble with Multiple Models

```python
separator = Separator(
    ensemble_algorithm="median_wave",
    ensemble_weights=[0.5, 0.3, 0.2],
)
separator.load_model([
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "Kim_Vocal_2.onnx",
    "Mel-Roformer-Viperx-1053.ckpt",
])
output_files = separator.separate("song.mp3")
```

### Single Stem Output

```python
separator = Separator(output_single_stem="Vocals")
output_files = separator.separate("song.mp3")
# Returns only song_(Vocals).flac
```

### Spectrogram Inversion

```python
separator = Separator(invert_using_spec=True)
output_files = separator.separate("song.mp3")
```

## Using with the Notebook

In Google Colab, import directly from the cloned repo:

```python
import sys
sys.path.insert(0, "/content/vsep")

from separator import Separator
import config.variables as cfg

cfg.MAX_DOWNLOAD_WORKERS = 4

separator = Separator(
    model_file_dir="./models",
    output_dir="./output",
    use_soundfile=True,
)
separator.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
output_files = separator.separate("audio_file.wav")
```
