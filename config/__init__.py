"""Configuration package for audio-separator."""

from config.variables import (
    # Repository URLs
    UVR_PUBLIC_REPO_URL,
    UVR_VIP_REPO_URL,
    AUDIO_SEPARATOR_REPO_URL,
    UVR_MODEL_DATA_URL_PREFIX,
    UVR_VR_MODEL_DATA_URL,
    UVR_MDX_MODEL_DATA_URL,
    # Paths
    MDXC_YAML_PATH_PREFIX,
    # Download config
    MAX_DOWNLOAD_WORKERS,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_TIMEOUT,
    HTTP_POOL_CONNECTIONS,
    HTTP_POOL_MAXSIZE,
    # Model data files
    VR_MODEL_DATA_FILENAME,
    MDX_MODEL_DATA_FILENAME,
    # Helper functions
    get_repo_url,
    get_mdx_yaml_url,
    get_fallback_url,
)

__all__ = [
    # Repository URLs
    "UVR_PUBLIC_REPO_URL",
    "UVR_VIP_REPO_URL",
    "AUDIO_SEPARATOR_REPO_URL",
    "UVR_MODEL_DATA_URL_PREFIX",
    "UVR_VR_MODEL_DATA_URL",
    "UVR_MDX_MODEL_DATA_URL",
    # Paths
    "MDXC_YAML_PATH_PREFIX",
    # Download config
    "MAX_DOWNLOAD_WORKERS",
    "DOWNLOAD_CHUNK_SIZE",
    "DOWNLOAD_TIMEOUT",
    "HTTP_POOL_CONNECTIONS",
    "HTTP_POOL_MAXSIZE",
    # Model data files
    "VR_MODEL_DATA_FILENAME",
    "MDX_MODEL_DATA_FILENAME",
    # Helper functions
    "get_repo_url",
    "get_mdx_yaml_url",
    "get_fallback_url",
]
