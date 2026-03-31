# Configuration for vsep

This package contains all configuration variables and URL mappings for vsep.

## Purpose

The configuration system allows you to:
- **Easily switch download sources** - Change repository URLs without modifying code
- **Customize download behavior** - Adjust parallel workers, chunk sizes, timeouts
- **Use mirror repositories** - Override with alternative download sources
- **Centralize settings** - All configuration in one place

## Variables

### Repository URLs

- `UVR_PUBLIC_REPO_URL` - Primary source for public UVR models
- `UVR_VIP_REPO_URL` - Source for VIP (subscriber-only) models
- `AUDIO_SEPARATOR_REPO_URL` - vsep-specific models and configs
- `UVR_MODEL_DATA_URL_PREFIX` - Base URL for UVR model parameter data

### Download Settings

- `MAX_DOWNLOAD_WORKERS` - Number of parallel download threads (default: 4)
- `DOWNLOAD_CHUNK_SIZE` - Size of download chunks in bytes (default: 256KB)
- `DOWNLOAD_TIMEOUT` - Request timeout in seconds (default: 300)
- `HTTP_POOL_CONNECTIONS` - Connection pool size (default: 10)

## Usage

### Default Usage

The configuration is automatically used by the Separator class:

```python
from separator import Separator

separator = Separator()
separator.download_model_and_data("model_name.onnx")
```

### Custom Configuration

To use custom download sources or settings, modify the variables before importing Separator:

```python
# Override configuration variables
import config.variables as cfg

# Use a mirror repository
cfg.UVR_PUBLIC_REPO_URL = "https://mirror.example.com/uvr-models"
cfg.AUDIO_SEPARATOR_REPO_URL = "https://mirror.example.com/vsep"

# Adjust download settings for faster/slower connections
cfg.MAX_DOWNLOAD_WORKERS = 8  # More parallel downloads
cfg.DOWNLOAD_CHUNK_SIZE = 524288  # 512KB chunks
cfg.DOWNLOAD_TIMEOUT = 600  # 10 minute timeout

# Now import and use Separator
from separator import Separator

separator = Separator()
```

### Environment Variable Override (Advanced)

For even more flexibility, you can set environment variables and read them in your code:

```python
import os
import config.variables as cfg

# Override with environment variables if set
if os.getenv('UVR_MIRROR_URL'):
    cfg.UVR_PUBLIC_REPO_URL = os.getenv('UVR_MIRROR_URL')

if os.getenv('DOWNLOAD_WORKERS'):
    cfg.MAX_DOWNLOAD_WORKERS = int(os.getenv('DOWNLOAD_WORKERS'))
```

## File Structure

```
config/
├── __init__.py       # Package init with exports
├── variables.py      # All configuration variables
└── README.md         # This file
```

## Adding Custom Models

To add support for custom model repositories:

1. Add your repository URL to `variables.py`:
   ```python
   CUSTOM_REPO_URL = "https://your-repo.com/models"
   ```

2. Export it in `__init__.py`

3. Use it in your code:
   ```python
   from config import CUSTOM_REPO_URL
   download_url = f"{CUSTOM_REPO_URL}/{model_filename}"
   ```

## Troubleshooting

### Slow Downloads

Increase parallel workers and chunk size:
```python
cfg.MAX_DOWNLOAD_WORKERS = 8
cfg.DOWNLOAD_CHUNK_SIZE = 524288
```

### Download Timeouts

Increase timeout for large models:
```python
cfg.DOWNLOAD_TIMEOUT = 900  # 15 minutes
```

### Using Alternative Sources

If the primary repositories are unavailable:
```python
cfg.UVR_PUBLIC_REPO_URL = "https://alternative-source.com/uvr"
```
