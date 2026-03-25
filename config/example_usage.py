"""
Example: Custom Configuration for Audio Separator

This example shows how to customize download settings and use alternative repositories.
"""

# =============================================================================
# Example 1: Faster Downloads (for high-speed connections)
# =============================================================================

import config.variables as cfg

# Increase parallel downloads for faster throughput
cfg.MAX_DOWNLOAD_WORKERS = 8  # Default is 4

# Larger chunk sizes reduce I/O overhead
cfg.DOWNLOAD_CHUNK_SIZE = 524288  # 512KB instead of 256KB

# Now use the separator with optimized settings
from separator import Separator

separator = Separator()
# separator.download_model_and_data("model_name.onnx")


# =============================================================================
# Example 2: Using Mirror Repository
# =============================================================================

# Override repository URLs to use mirrors
cfg.UVR_PUBLIC_REPO_URL = "https://your-mirror.example.com/uvr-models"
cfg.AUDIO_SEPARATOR_REPO_URL = "https://your-mirror.example.com/audio-separator"

# The separator will now download from your mirror
# separator = Separator()
# separator.download_model_and_data("model_name.onnx")


# =============================================================================
# Example 3: Environment-Based Configuration
# =============================================================================

import os

# Read configuration from environment variables
if os.getenv("AUDIO_SEPARATOR_REPO_URL"):
    cfg.UVR_PUBLIC_REPO_URL = os.getenv("AUDIO_SEPARATOR_REPO_URL")

if os.getenv("MAX_DOWNLOAD_WORKERS"):
    cfg.MAX_DOWNLOAD_WORKERS = int(os.getenv("MAX_DOWNLOAD_WORKERS"))

# This allows configuration without code changes:
# Run with: AUDIO_SEPARATOR_REPO_URL=https://custom.com python your_script.py


# =============================================================================
# Example 4: Slower/Unstable Connection
# =============================================================================

# Reduce parallel workers to avoid overwhelming the connection
cfg.MAX_DOWNLOAD_WORKERS = 2

# Smaller chunks for better resume capability
cfg.DOWNLOAD_CHUNK_SIZE = 65536  # 64KB

# Longer timeout for slow connections
cfg.DOWNLOAD_TIMEOUT = 600  # 10 minutes


# =============================================================================
# Example 5: All Configuration Options
# =============================================================================

# Repository URLs
cfg.UVR_PUBLIC_REPO_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
cfg.UVR_VIP_REPO_URL = "https://github.com/Anjok0109/ai_magic/releases/download/v5"
cfg.AUDIO_SEPARATOR_REPO_URL = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"

# Download behavior
cfg.MAX_DOWNLOAD_WORKERS = 4  # Parallel download threads
cfg.DOWNLOAD_CHUNK_SIZE = 262144  # 256KB chunks
cfg.DOWNLOAD_TIMEOUT = 300  # 5 minute timeout

# Connection pooling
cfg.HTTP_POOL_CONNECTIONS = 10
cfg.HTTP_POOL_MAXSIZE = 10

# Model data filenames (usually don't need to change)
cfg.VR_MODEL_DATA_FILENAME = "vr_model_data.json"
cfg.MDX_MODEL_DATA_FILENAME = "mdx_model_data.json"
