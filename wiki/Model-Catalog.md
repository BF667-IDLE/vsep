# Model Catalog

vsep supports **100+ models** from the Ultimate Vocal Remover ecosystem, organized into four architectures. Models are auto-downloaded on first use from the UVR model repository.

## Architecture Overview

| Architecture | Count | Extension | Description |
|:-------------|------:|:---------|:------------|
| **VR** (Band Split) | 29 | `.pth` | Classic UVR RNN models. Lightweight, supports TTA and post-processing. |
| **MDX-Net** | 39 | `.onnx` | Fast ONNX models. Includes community-trained and VIP variants. |
| **MDXC** (MDX23C + Roformer) | 83+ | `.ckpt` | State-of-the-art. Includes BS-Roformer, Mel-Band-Roformer, Bandit, SCNet, and others. |
| **Demucs** | 24 | `.th` / `.yaml` | Meta's hybrid transformer. Best for 4-stem separation (vocals, drums, bass, other). |

## Quick Recommendations

### Best Overall Vocal Extraction
| Model | Architecture | Why |
|-------|:-----------|:-----|
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | BS-Roformer | SDR 12.98 — highest SDR of any model |
| `Mel-Roformer-Viperx-1143` (ep_3005) | Mel-Band-Roformer | Excellent vocal clarity |
| `mel_band_roformer_kim_ft_unwa.ckpt` | Mel-Band-Roformer | Fine-tuned by unwa, very clean output |

### Best 4-Stem Separation
| Model | Architecture | Why |
|-------|:-----------|:-----|
| `htdemucs_ft.yaml` | Demucs v4 | Balanced quality across all 4 stems |
| `htdemucs_6s.yaml` | Demucs v4 | 6-stem version (adds piano, guitar) |

### Best Instrumental Extraction
| Model | Architecture | Why |
|-------|:-----------|:-----|
| `MDX23C-8KFFT-InstVoc_HQ.ckpt` | MDX23C | Clean instrumental with good vocal removal |
| `melband_roformer_inst_v2.ckpt` | Mel-Band-Roformer | Dedicated instrumental model, high quality |

### Best Karaoke
| Model | Architecture | Why |
|-------|:-----------|:-----|
| `mel_band_roformer_karaoke_becruily.ckpt` | Mel-Band-Roformer | Aggressive vocal removal for karaoke use |
| `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | Mel-Band-Roformer | Balanced karaoke quality |

### Best Post-Processing
| Model | Purpose | Architecture |
|-------|:---------|:------------|
| `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` | De-Reverb | Roformer |
| `dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt` | De-Reverb-Echo | Roformer |
| `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` | Denoise | Roformer |
| `UVR-DeNoise-Lite.pth` | Noise Removal (lightweight) | VR |

### Best Multi-Stem
| Model | Stems | Architecture |
|-------|:------|:------------|
| `htdemucs_6s.yaml` | vocals, drums, bass, other, piano, guitar | Demucs v4 |
| `model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt` | male vocals, female vocals | BS-Roformer |
| `scnet_checkpoint_musdb18.ckpt` | vocals, drums, bass, other | SCNet |

## Listing Models

Use the CLI to browse all models:

```bash
# By architecture
python utils/cli.py --list_models

# By task category
python utils/cli.py --list_models --list_format categories

# Only vocal models
python utils/cli.py --list_models --list_stem vocals

# Only Roformer/MDXC models
python utils/cli.py --list_models --list_type MDXC

# JSON output for scripting
python utils/cli.py --list_models --list_format json
```

## Model Sources

Models are loaded from two sources that get merged at runtime:

1. **`models.json`** — Local model registry in the repo root. Contains curated models with display names and download URLs.
2. **`download_checks.json`** — Fetched from the UVR application data repository. Contains the full UVR model list.

The `Separator.list_models()` method merges both sources and returns a unified list.

## Model File Structure

Models in `models.json` follow this format:

```json
{
  "vr_download_list": { "Display Name": "model_file.pth" },
  "mdx_download_list": { "Display Name": "model_file.onnx" },
  "mdx23c_download_list": { "Display Name": { "model.ckpt": "config.yaml" } },
  "roformer_download_list": { "Display Name": { "model.ckpt": "config.yaml" } }
}
```

Model performance scores (SDR, SIR, SAR, ISR) are stored in `models-scores.json`. Parameter metadata (segment size, hop length, etc.) is stored in `model-data.json`.

## VIP Models

Some MDX models are marked as VIP (require a paid subscription to Anjok07's Patreon). These are downloaded from a separate repository. vsep will display a reminder when using VIP models.

## Adding Custom Models

1. Place your `.ckpt`, `.onnx`, or `.pth` file in your `VSEP_MODEL_DIR`
2. If the model has a config YAML, place it in the same directory
3. For MDX models, add an entry to `model-data.json` with the model's MD5 hash as key
4. Pass the model filename directly: `python utils/cli.py song.mp3 -m my_custom_model.ckpt`
