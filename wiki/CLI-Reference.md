# CLI Reference

## Basic Usage

```bash
python utils/cli.py <audio_file> [options]
```

## Separation

| Flag | Default | Description |
|------|---------|-------------|
| `-m`, `--model_filename` | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | Model file to use |
| `--extra_models` | — | Additional models for ensembling |
| `--output_format` | `FLAC` | Output format (WAV, FLAC, MP3, OGG, etc.) |
| `--output_bitrate` | — | Output bitrate for lossy formats (e.g. `320k`) |
| `--output_dir` | Current directory | Output directory |
| `--model_file_dir` | `/tmp/vsep-models/` | Model storage directory |
| `--download_model_only` | — | Download model without separating |

## Common Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--invert_spect` | off | Invert secondary stem via spectrogram |
| `--normalization` | `0.9` | Peak normalization threshold |
| `--amplification` | `0.0` | Minimum peak amplification |
| `--single_stem` | — | Output only one stem (Vocals, Instrumental, Drums, Bass, etc.) |
| `--sample_rate` | `44100` | Output sample rate |
| `--use_soundfile` | off | Use soundfile for output (avoids OOM) |
| `--use_autocast` | off | Enable FP16 autocast (GPU only) |
| `--chunk_duration` | — | Split audio into N-second chunks |
| `--custom_output_names` | — | JSON dict mapping stem names to output filenames |

## Model Listing

| Flag | Description |
|------|-------------|
| `-l`, `--list_models` | List all supported models |
| `--list_type` | Filter by architecture: `VR`, `MDX`, `Demucs`, `MDXC` |
| `--list_stem` | Filter by output stem: `vocals`, `drums`, `bass`, etc. |
| `--list_limit` | Show only first N models per group |
| `--list_format` | `pretty` (grouped table), `json` (raw), `categories` (task-based) |

### Examples

```bash
# Show all models grouped by architecture
python utils/cli.py --list_models

# Show only vocal models
python utils/cli.py --list_models --list_stem vocals

# Show only Roformer models in JSON
python utils/cli.py --list_models --list_type MDXC --list_format json

# Show models grouped by task category
python utils/cli.py --list_models --list_format categories

# Show top 5 models per architecture
python utils/cli.py --list_models --list_limit 5
```

## Ensemble

| Flag | Default | Description |
|------|---------|-------------|
| `--ensemble_algorithm` | — | Algorithm: `avg_wave`, `median_wave`, `min_wave`, `max_wave`, `avg_fft`, `median_fft`, `min_fft`, `max_fft`, `uvr_max_spec`, `uvr_min_spec`, `ensemble_wav` |
| `--ensemble_weights` | equal | Per-model weights (must match model count) |
| `--ensemble_preset` | — | Named preset (e.g. `vocal_balanced`) |
| `--list_presets` | — | List available ensemble presets |

### Examples

```bash
# Use a preset
python utils/cli.py song.mp3 --ensemble_preset vocals_ensemble

# Custom ensemble with two models
python utils/cli.py song.mp3 \
  -m model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  --extra_models Kim_Vocal_2.onnx \
  --ensemble_algorithm median_wave

# Weighted ensemble
python utils/cli.py song.mp3 \
  -m model1.ckpt --extra_models model2.onnx \
  --ensemble_weights 0.7 0.3
```

## MDX Architecture Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--mdx_segment_size` | `256` | Segment size (larger = better but slower) |
| `--mdx_overlap` | `0.25` | Overlap between windows (0.001–0.999) |
| `--mdx_batch_size` | `1` | Batch size (more RAM = faster) |
| `--mdx_hop_length` | `1024` | Hop length / stride |
| `--mdx_enable_denoise` | off | Enable denoising |

## VR Architecture Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--vr_batch_size` | `1` | Batch size |
| `--vr_window_size` | `512` | Window size (320 = best, 1024 = fast) |
| `--vr_aggression` | `5` | Vocal extraction intensity (-100 to 100) |
| `--vr_enable_tta` | off | Test-time augmentation (slow but better) |
| `--vr_high_end_process` | off | Mirror missing frequency range |
| `--vr_enable_post_process` | off | Remove leftover vocal artifacts |
| `--vr_post_process_threshold` | `0.2` | Post-process threshold (0.1–0.3) |

## Demucs Architecture Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--demucs_segment_size` | `Default` | Segment size (1–100) |
| `--demucs_shifts` | `2` | Random shift predictions |
| `--demucs_overlap` | `0.25` | Overlap between windows |
| `--demucs_segments_enabled` | `True` | Enable segment processing |

## MDXC / Roformer Architecture Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--mdxc_segment_size` | `256` | Segment size (larger = better but slower) |
| `--mdxc_override_model_segment_size` | off | Override model's default segment size |
| `--mdxc_overlap` | `8` | Overlap between windows (2–50) |
| `--mdxc_batch_size` | `1` | Batch size |
| `--mdxc_pitch_shift` | `0` | Pitch shift in semitones (-12 to 12) |

## Debug

| Flag | Description |
|------|-------------|
| `-v`, `--version` | Show version and exit |
| `-d`, `--debug` | Enable debug logging |
| `-e`, `--env_info` | Print environment info and exit |
| `--log_level` | Set log level: `info`, `debug`, `warning` |
