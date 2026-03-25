"""
Configuration and URL mappings for audio-separator model downloads.

Centralizes all repository URLs, paths, and configuration to make it easy to:
- Switch to mirror/alternative download sources
- Customize download behavior
- Manage model repository mappings
"""

# =============================================================================
# MODEL REPOSITORY URLs
# =============================================================================

# Primary UVR model repository (public models)
UVR_PUBLIC_REPO_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"

# VIP models repository (Anjok07's paid subscriber models)
UVR_VIP_REPO_URL = "https://github.com/Anjok0109/ai_magic/releases/download/v5"

# Audio-separator specific models and configs
AUDIO_SEPARATOR_REPO_URL = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"

# UVR model data (for parameter lookups by hash)
UVR_MODEL_DATA_URL_PREFIX = "https://raw.githubusercontent.com/TRvlvr/application_data/main"
UVR_VR_MODEL_DATA_URL = f"{UVR_MODEL_DATA_URL_PREFIX}/vr_model_data/model_data_new.json"
UVR_MDX_MODEL_DATA_URL = f"{UVR_MODEL_DATA_URL_PREFIX}/mdx_model_data/model_data_new.json"

# =============================================================================
# MODEL PATH MAPPINGS
# =============================================================================

# MDXC YAML config file paths within UVR repo
MDXC_YAML_PATH_PREFIX = "mdx_model_data/mdx_c_configs"

# =============================================================================
# DOWNLOAD CONFIGURATION
# =============================================================================

# Number of parallel download threads
MAX_DOWNLOAD_WORKERS = 4

# Chunk size for downloads (256KB for faster downloads)
DOWNLOAD_CHUNK_SIZE = 262144  # 256 * 1024

# Request timeout in seconds (300s = 5 minutes for large files)
DOWNLOAD_TIMEOUT = 300

# Connection pool settings
HTTP_POOL_CONNECTIONS = 10
HTTP_POOL_MAXSIZE = 10

# =============================================================================
# MODEL DATA FILES
# =============================================================================

# Local model data filenames
VR_MODEL_DATA_FILENAME = "vr_model_data.json"
MDX_MODEL_DATA_FILENAME = "mdx_model_data.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_repo_url(is_vip=False):
    """
    Get the appropriate repository URL based on model type.
    
    Args:
        is_vip: True for VIP models, False for public models
        
    Returns:
        Repository URL string
    """
    return UVR_VIP_REPO_URL if is_vip else UVR_PUBLIC_REPO_URL


def get_mdx_yaml_url(filename):
    """
    Get the full URL for an MDXC YAML config file.
    
    Args:
        filename: YAML config filename
        
    Returns:
        Full URL to the YAML file
    """
    return f"{UVR_PUBLIC_REPO_URL}/{MDXC_YAML_PATH_PREFIX}/{filename}"


def get_fallback_url(filename):
    """
    Get the fallback URL from audio-separator repo.
    
    Args:
        filename: Model/config filename
        
    Returns:
        Fallback URL string
    """
    return f"{AUDIO_SEPARATOR_REPO_URL}/{filename}"
