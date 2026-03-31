# Architecture

## System Overview

vsep is built around a central `Separator` class that orchestrates model downloading, loading, and inference across four separation architectures. The architecture is designed to be modular — each architecture has its own separator class with shared infrastructure for I/O, chunking, and ensembling.

```
┌─────────────────────────────────────────────────────┐
│                   Separator (Main)                   │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ list_models │  │ download     │  │ load_model      │  │
│  └────────────┘  └──────────────┘  └────────┬────────┘  │
│                                       │             │
│  ┌────────────────────────────────────┬──────────────┐ │
│  │              separate()              │              │ │
│  └────────────────────────────────────┼──────────────┘ │
│                                       │             │
│  ┌────────────┬──────────┬───────────┬─┴─────────────┐  │
│  │  MDX       │  VR      │  Demucs   │  MDXC         │  │
│  │ Separator  │Separator │ Separator │  Separator    │  │
│  └────────────┴──────────┴───────────┴─────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │         uvr_lib_v5 (inference library)            │  │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────────┐   │  │
│  │  │ mdxnet   │  │ vr_network│  │ demucs      │   │  │
│  │  └──────────┘  └───────────┘  └──────────────┘   │  │
│  │  ┌──────────────────────────────────────┐        │  │
│  │  │           roformer                   │        │  │
│  │  └──────────────────────────────────────┘        │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Architecture Detection

The architecture is auto-detected from the model filename extension:

| Extension | Architecture | Separator Class |
|:-----------|:------------|:---------------|
| `.pth` | VR Band Split | `VRSeparator` |
| `.onnx` | MDX-Net | `MDXSeparator` |
| `.yaml` (Demucs v4) | Demucs | `DemucsSeparator` |
| `.ckpt` (with specific config) | MDXC / Roformer | `MDXCSeparator` |

The detection logic in `Separator.load_model()` examines the file extension and, for `.ckpt` files, the associated YAML config to determine whether to use the MDXC TFC-TDF-Net or Roformer loader.

## Model Sources

Models are loaded from two sources that get merged at runtime:

1. **`models.json`** (local) — Curated model registry in the repo root with display names and download URLs. Contains VR, MDX, MDX23C, and Roformer models.
2. **`download_checks.json`** (remote) — Fetched from `TRvlvr/application_data` on GitHub. Contains the full UVR model list including Demucs, VR, MDX, VIP, and other models.

The `list_models()` method and `list_supported_model_files()` method merge both sources to provide a complete model catalog.

## Model Download Pipeline

```
User requests model filename
         │
         ▼
Check local cache (model_file_dir)
         │
    ┌────┴─────┐
    │ Found?    │
    └────┬─────┘
      Yes │  No
         │   │
         ▼   ▼
    Load  Fetch from models.json /
          download_checks.json
               │
         ▼
    Resolve download URLs
    (primary + fallback)
               │
         ▼
    Download with 4 parallel threads
    (HTTP Range resume support)
               │
         ▼
    Verify download (MD5 hash)
               │
         ▼
    Load model data from YAML/hash
               │
         ▼
    Ready for inference
```

Models are downloaded from multiple sources with automatic fallback:
1. **Primary**: UVR public model repo (`TRvlvr/model_repo`)
2. **Fallback**: Audio-separator repo (`nomadkaraoke/python-audio-separator`)
3. **UVR Data**: Application data from `TRvlvr/application_data`
4. **VIP**: Separate repo for paid subscriber models

## Hardware Acceleration

vsep automatically detects and uses the best available hardware:

```
CUDA available? ──→ Use ONNX GPU runtime + PyTorch CUDA
       │
MPS available? ──→ Use PyTorch MPS (Apple Silicon)
       │
DirectML available? ──→ Use ONNX DirectML
       │
       └──→ CPU (slowest)
```

The device selection happens in `setup_accelerated_inferencing_device()` which tries each backend in order and falls back gracefully.

## Audio Processing Pipeline

### Chunking

For long audio files, vsep splits the input into fixed-duration chunks:

```
Long audio file (e.g., 60 min)
         │
         ▼
    Split into N-second chunks
    (e.g., 10 min each = 6 chunks)
         │
         ▼
    Process each chunk independently
         │
         ▼
    Concatenate outputs
    (no overlap/crossfade)
```

### Overlap-Add (MDXC/Roformer)

The MDXC separator uses overlap-add processing where each segment is overlapped and blended for smoother results:
- Segment processed with overlap
- Overlapping regions are weighted and averaged
- Produces seamless output even at segment boundaries

## Ensemble Mode

Multiple models can be combined for higher-quality results:

```
Input audio
    │
    ├──→ Model A ──→ Stem A
    ├──→ Model B ──→ Stem B
    └──→ Model C ──→ Stem C
         │
         ▼
    Combine with algorithm
    (e.g., median_wave)
         │
         ▼
    Final output
```

11 ensemble algorithms are available, operating in either the time domain (waveform) or frequency domain (FFT).

## Configuration System

All configuration lives in `config/variables.py`:

| Category | Settings |
|----------|:---------|
| Repository URLs | UVR public, VIP, audio-separator repos |
| Downloads | Worker count, chunk size, timeout, connection pool |
| Logging | Default level, format string |
| Defaults | Model dir, output format, normalization, sample rate, etc. |
| Ensemble | Valid algorithms, stem name mapping |
| Architecture Params | Default hyperparams for MDX, VR, Demucs, MDXC |
| Helper Functions | `get_repo_url()`, `get_mdx_yaml_url()`, `get_fallback_url()` |

## File Structure

```
separator/
├── separator.py              # Main Separator class
├── common_separator.py       # Shared base class for architectures
├── ensembler.py              # Ensemble algorithm implementations
├── audio_chunking.py         # Chunk-based processing
├── architectures/
│   ├── mdx_separator.py      # MDX-Net (.onnx)
│   ├── vr_separator.py       # VR Band Split (.pth)
│   ├── demucs_separator.py   # Demucs v4 (.th/.yaml)
│   └── mdxc_separator.py     # MDXC/Roformer (.ckpt)
├── roformer/                 # Roformer model loader
│   ├── roformer_loader.py
│   ├── parameter_validator.py
│   └── configuration_normalizer.py
└── uvr_lib_v5/               # UVR inference library
    ├── mdxnet.py             # MDX-Net implementation
    ├── demucs/               # Demucs models
    ├── roformer/             # Roformer network
    └── vr_network/           # VR network layers

config/
├── variables.py              # All settings and defaults
└── __init__.py               # Package exports

utils/
└── cli.py                    # Command-line interface
```
