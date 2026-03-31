"""
Configuration for vsep — audio stem separator.

Centralizes all defaults, constants, and settings so separator logic
stays clean. Everything users might want to tweak lives here.
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


# =============================================================================
# LOGGING
# =============================================================================

import logging

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"

# =============================================================================
# SEPARATOR DEFAULTS
# =============================================================================

DEFAULT_MODEL_FILE_DIR = "/tmp/audio-separator-models/"
DEFAULT_OUTPUT_FORMAT = "WAV"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_NORMALIZATION_THRESHOLD = 0.9
DEFAULT_AMPLIFICATION_THRESHOLD = 0.0
DEFAULT_OUTPUT_SINGLE_STEM = None
DEFAULT_INVERT_USING_SPEC = False
DEFAULT_USE_SOUNDFILE = False
DEFAULT_USE_AUTOCAST = False
DEFAULT_USE_DIRECTML = False
DEFAULT_CHUNK_DURATION = None

# =============================================================================
# ENSEMBLE
# =============================================================================

VALID_ENSEMBLE_ALGORITHMS = [
    "avg_wave", "median_wave", "min_wave", "max_wave",
    "avg_fft", "median_fft", "min_fft", "max_fft",
    "uvr_max_spec", "uvr_min_spec", "ensemble_wav",
]
DEFAULT_ENSEMBLE_ALGORITHM = "avg_wave"

# =============================================================================
# STEM NAME MAPPING
# =============================================================================

STEM_NAME_MAP = {
    "vocals": "Vocals",
    "instrumental": "Instrumental",
    "inst": "Instrumental",
    "karaoke": "Instrumental",
    "other": "Other",
    "no_vocals": "Instrumental",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano",
    "synthesizer": "Synthesizer",
    "strings": "Strings",
    "woodwinds": "Woodwinds",
    "brass": "Brass",
    "wind inst": "Wind Inst",
    "lead vocals": "Lead Vocals",
    "backing vocals": "Backing Vocals",
    "primary stem": "Primary Stem",
    "secondary stem": "Secondary Stem",
}

# =============================================================================
# ARCHITECTURE DEFAULT PARAMS
# =============================================================================

DEFAULT_MDX_PARAMS = {
    "hop_length": 1024,
    "segment_size": 256,
    "overlap": 0.25,
    "batch_size": 1,
    "enable_denoise": False,
}

DEFAULT_VR_PARAMS = {
    "batch_size": 1,
    "window_size": 512,
    "aggression": 5,
    "enable_tta": False,
    "enable_post_process": False,
    "post_process_threshold": 0.2,
    "high_end_process": False,
}

DEFAULT_DEMUCS_PARAMS = {
    "segment_size": "Default",
    "shifts": 2,
    "overlap": 0.25,
    "segments_enabled": True,
}

DEFAULT_MDXC_PARAMS = {
    "segment_size": 256,
    "override_model_segment_size": False,
    "batch_size": 1,
    "overlap": 8,
    "pitch_shift": 0,
}
