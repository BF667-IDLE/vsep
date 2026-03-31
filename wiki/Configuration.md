# Configuration

All configurable values are centralized in `config/variables.py`. Import and modify before creating the `Separator` instance.

## Repository URLs

| Variable | Default | Description |
|----------|---------|-------------|
| `UVR_PUBLIC_REPO_URL` | UVR model repo URL | Primary model download source |
| `UVR_VIP_REPO_URL` | UVR VIP repo URL | Paid subscriber models |
| `AUDIO_SEPARATOR_REPO_URL` | audio-separator repo URL | Fallback download source |
| `UVR_MODEL_DATA_URL_PREFIX` | UVR application data URL | Model parameter metadata |
| `UVR_VR_MODEL_DATA_URL` | VR model data URL | VR model hash → params mapping |
| `UVR_MDX_MODEL_DATA_URL` | MDX model data URL | MDX model hash → params mapping |

## Download Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_DOWNLOAD_WORKERS` | `4` | Parallel download threads |
| `DOWNLOAD_CHUNK_SIZE` | `262144` (256 KB) | Download chunk size in bytes |
| `DOWNLOAD_TIMEOUT` | `300` | HTTP request timeout in seconds |
| `HTTP_POOL_CONNECTIONS` | `10` | urllib3 connection pool size |
| `HTTP_POOL_MAXSIZE` | `10` | Maximum pool connections |

### Example: Speed Up Downloads

```python
import config.variables as cfg

cfg.MAX_DOWNLOAD_WORKERS = 8       # 8 parallel threads
cfg.DOWNLOAD_CHUNK_SIZE = 524288   # 512 KB chunks
cfg.DOWNLOAD_TIMEOUT = 600        # 10 minute timeout for slow connections
```

## Separator Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODEL_FILE_DIR` | `/tmp/audio-separator-models/` | Where models are cached |
| `DEFAULT_OUTPUT_FORMAT` | `WAV` | Output audio format |
| `DEFAULT_SAMPLE_RATE` | `44100` | Output sample rate in Hz |
| `DEFAULT_NORMALIZATION_THRESHOLD` | `0.9` | Peak normalization (0.0–1.0) |
| `DEFAULT_AMPLIFICATION_THRESHOLD` | `0.0` | Min peak amplification |
| `DEFAULT_OUTPUT_SINGLE_STEM` | `None` | If set, output only this stem |
| `DEFAULT_INVERT_USING_SPEC` | `False` | Invert secondary stem via spectrogram |
| `DEFAULT_USE_SOUNDFILE` | `False` | Use soundfile for output |
| `DEFAULT_USE_AUTOCAST` | `False` | FP16 autocast on GPU |
| `DEFAULT_USE_DIRECTML` | `False` | DirectML for AMD/Intel GPUs |
| `DEFAULT_CHUNK_DURATION` | `None` | Seconds per chunk (None = no chunking) |

### Example: Change Output Defaults

```python
import config.variables as cfg

cfg.DEFAULT_OUTPUT_FORMAT = "FLAC"
cfg.DEFAULT_SAMPLE_RATE = 48000
cfg.DEFAULT_NORMALIZATION_THRESHOLD = 0.95
```

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LOG_LEVEL` | `logging.INFO` | Default log level |
| `DEFAULT_LOG_FORMAT` | `%(asctime)s - %(levelname)s - %(module)s - %(message)s` | Log message format |

## Architecture-Specific Defaults

### MDX

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hop_length` | `1024` | STFT hop length |
| `segment_size` | `256` | Segment size for processing |
| `overlap` | `0.25` | Overlap ratio (0.001–0.999) |
| `batch_size` | `1` | Batch size for inference |
| `enable_denoise` | `False` | Built-in denoising |

### VR

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `1` | Batch size |
| `window_size` | `512` | Analysis window size |
| `aggression` | `5` | Vocal extraction intensity (-100 to 100) |
| `enable_tta` | `False` | Test-time augmentation |
| `enable_post_process` | `False` | Remove leftover vocal artifacts |
| `post_process_threshold` | `0.2` | Artifact detection threshold (0.1–0.3) |
| `high_end_process` | `False` | Mirror missing frequency range |

### Demucs

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segment_size` | `"Default"` | Segment size (`"Default"` = auto) |
| `shifts` | `2` | Random shift augmentations |
| `overlap` | `0.25` | Segment overlap ratio |
| `segments_enabled` | `True` | Enable segment-based processing |

### MDXC / Roformer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segment_size` | `256` | Segment size for processing |
| `override_model_segment_size` | `False` | Override model's default |
| `batch_size` | `1` | Batch size for inference |
| `overlap` | `8` | Overlap between windows (2–50) |
| `pitch_shift` | `0` | Pitch shift in semitones |

### Example: Override Architecture Params

```python
from separator import Separator

separator = Separator(
    mdx_params={"segment_size": 512, "overlap": 0.5, "batch_size": 4},
    vr_params={"window_size": 320, "aggression": 10},
    mdxc_params={"segment_size": 512, "overlap": 12},
)
```

## Ensemble

| Variable | Description |
|----------|-------------|
| `VALID_ENSEMBLE_ALGORITHMS` | List of 11 valid algorithm names |
| `DEFAULT_ENSEMBLE_ALGORITHM` | `avg_wave` |

### Valid Ensemble Algorithms

| Algorithm | Domain | Description |
|:----------|:-------|:-----------|
| `avg_wave` | Time | Average waveforms |
| `median_wave` | Time | Median waveform (removes outliers) |
| `min_wave` | Time | Minimum amplitude |
| `max_wave` | Time | Maximum amplitude |
| `avg_fft` | Frequency | Average in frequency domain |
| `median_fft` | Frequency | Median in frequency domain |
| `min_fft` | Frequency | Minimum magnitude spectrum |
| `max_fft` | Frequency | Maximum magnitude spectrum |
| `uvr_max_spec` | Frequency | UVR max spectral magnitude |
| `uvr_min_spec` | Frequency | UVR min spectral magnitude |
| `ensemble_wav` | Time | UVR's native waveform ensemble |

## Stem Name Mapping

The `STEM_NAME_MAP` dictionary normalizes stem names across different models:

| Alias | Normalized | Aliases |
|-------|:----------|:--------|
| Vocals | Vocals | — |
| Instrumental | Instrumental | `inst`, `karaoke`, `no_vocals` |
| Drums | Drums | — |
| Bass | Bass | — |
| Guitar | Guitar | — |
| Piano | Piano | — |
| Other | Other | — |
| Lead Vocals | Lead Vocals | — |
| Backing Vocals | Backing Vocals | — |
| Synthesizer | Synthesizer | — |
| Strings | Strings | — |
| Woodwinds | Woodwinds | — |
| Brass | Brass | — |

## Helper Functions

### `get_repo_url(is_vip=False)`

Returns the appropriate repository URL based on model type.

```python
from config.variables import get_repo_url
url = get_repo_url(is_vip=True)  # VIP repo URL
```

### `get_mdx_yaml_url(filename)`

Returns the full URL for an MDXC YAML config file.

```python
from config.variables import get_mdx_yaml_url
url = get_mdx_yaml_url("config_mel_band_roformer_vocals_gabox.yaml")
```

### `get_fallback_url(filename)`

Returns the fallback URL from audio-separator repo.

```python
from config.variables import get_fallback_url
url = get_fallback_url("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VSEP_MODEL_DIR` | Model cache directory | `/tmp/audio-separator-models/` |

```bash
# Use a custom model directory
export VSEP_MODEL_DIR=/mnt/data/vsep-models
python utils/cli.py song.mp3
```
