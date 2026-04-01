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

# =============================================================================
# MODEL CATALOG  (display name  →  filename)
# =============================================================================
# Flat lookup used by the Colab notebook and CLI so users pick by readable
# name and never need to type raw filenames.  Models here come from
# models.json; the full runtime catalog (UVR download_checks + local) is
# built dynamically by Separator.list_supported_model_files().
# =============================================================================

MODEL_CATALOG = {
    # ── VR ──────────────────────────────────────────────────────────────────
    "VR De-Reverb (aufr33-jarredou)": "UVR-De-Reverb-aufr33-jarredou.pth",
    "VR BVE-4B SN 44100-2": "UVR-BVE-4B_SN-44100-2.pth",
    # ── MDX ─────────────────────────────────────────────────────────────────
    "MDX-Net Inst HQ 5": "UVR-MDX-NET-Inst_HQ_5.onnx",
    # ── MDX23C ──────────────────────────────────────────────────────────────
    "MDX23C De-Reverb (aufr33-jarredou)": "MDX23C-De-Reverb-aufr33-jarredou.ckpt",
    "MDX23C DrumSep (aufr33-jarredou)": "MDX23C-DrumSep-aufr33-jarredou.ckpt",
    # ── Roformer / MDXC ────────────────────────────────────────────────────
    "Mel Roformer Karaoke Aufr33 Viperx (SDR 10.20)": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "MelBand Roformer Karaoke (Gabox)": "mel_band_roformer_karaoke_gabox.ckpt",
    "MelBand Roformer Karaoke V2 (Gabox)": "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "MelBand Roformer Karaoke (becruily)": "mel_band_roformer_karaoke_becruily.ckpt",
    "BS Roformer Karaoke (frazer-becruily)": "bs_roformer_karaoke_frazer_becruily.ckpt",
    "BS Roformer Karaoke (anvuew)": "bs_roformer_karaoke_anvuew.ckpt",
    "Denoise Mel Roformer Aufr33 (SDR 28.00)": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    "Denoise Mel Roformer Aufr33 Aggressive (SDR 27.98)": "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
    "MelBand Roformer Denoise Debleed (Gabox)": "mel_band_roformer_denoise_debleed_gabox.ckpt",
    "Mel Roformer Crowd Aufr33 Viperx (SDR 8.71)": "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
    "BS Roformer De-Reverb": "deverb_bs_roformer_8_384dim_10depth.ckpt",
    "Vocals MelBand Roformer (Kimberley Jensen)": "vocals_mel_band_roformer.ckpt",
    "MelBand Roformer Kim FT (unwa)": "mel_band_roformer_kim_ft_unwa.ckpt",
    "MelBand Roformer Kim FT 2 (unwa)": "mel_band_roformer_kim_ft2_unwa.ckpt",
    "MelBand Roformer Kim FT 2 Bleedless (unwa)": "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
    "MelBand Roformer Kim FT 3 (unwa)": "mel_band_roformer_kim_ft3_unwa.ckpt",
    "MelBand Roformer Kim Inst V1 Plus (Unwa)": "melband_roformer_inst_v1_plus.ckpt",
    "MelBand Roformer Kim Inst V1E (Unwa)": "melband_roformer_inst_v1e.ckpt",
    "MelBand Roformer Kim Inst V1E Plus (Unwa)": "melband_roformer_inst_v1e_plus.ckpt",
    "BS Roformer Vocals Revive (Unwa)": "bs_roformer_vocals_revive_unwa.ckpt",
    "BS Roformer Vocals Revive V2 (Unwa)": "bs_roformer_vocals_revive_v2_unwa.ckpt",
    "BS Roformer Vocals Revive V3e (Unwa)": "bs_roformer_vocals_revive_v3e_unwa.ckpt",
    "MelBand Roformer Vocals (becruily)": "mel_band_roformer_vocals_becruily.ckpt",
    "MelBand Roformer Instrumental (becruily)": "mel_band_roformer_instrumental_becruily.ckpt",
    "MelBand Roformer Vocals Fullness (Aname)": "mel_band_roformer_vocal_fullness_aname.ckpt",
    "BS Roformer Vocals (Gabox)": "bs_roformer_vocals_gabox.ckpt",
    "MelBand Roformer Vocals (Gabox)": "mel_band_roformer_vocals_gabox.ckpt",
    "MelBand Roformer Vocals V2 (Gabox)": "mel_band_roformer_vocals_v2_gabox.ckpt",
    "MelBand Roformer Vocals FV1 (Gabox)": "mel_band_roformer_vocals_fv1_gabox.ckpt",
    "MelBand Roformer Vocals FV2 (Gabox)": "mel_band_roformer_vocals_fv2_gabox.ckpt",
    "MelBand Roformer Vocals FV3 (Gabox)": "mel_band_roformer_vocals_fv3_gabox.ckpt",
    "MelBand Roformer Vocals FV4 (Gabox)": "mel_band_roformer_vocals_fv4_gabox.ckpt",
    "MelBand Roformer Vocals FV5 (Gabox)": "mel_band_roformer_vocals_fv5_gabox.ckpt",
    "MelBand Roformer Vocals FV6 (Gabox)": "mel_band_roformer_vocals_fv6_gabox.ckpt",
    "MelBand Roformer Vocals FV7b (Gabox)": "mel_band_roformer_vocals_fv7b_gabox.ckpt",
    "MelBand Roformer Instrumental (Gabox)": "mel_band_roformer_instrumental_gabox.ckpt",
    "MelBand Roformer Instrumental 2 (Gabox)": "mel_band_roformer_instrumental_2_gabox.ckpt",
    "MelBand Roformer Instrumental 3 (Gabox)": "mel_band_roformer_instrumental_3_gabox.ckpt",
    "MelBand Roformer Instrumental Bleedless V1 (Gabox)": "mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt",
    "MelBand Roformer Instrumental Bleedless V2 (Gabox)": "mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
    "MelBand Roformer Instrumental Bleedless V3 (Gabox)": "mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt",
    "MelBand Roformer Instrumental Fullness V1 (Gabox)": "mel_band_roformer_instrumental_fullness_v1_gabox.ckpt",
    "MelBand Roformer Instrumental Fullness V2 (Gabox)": "mel_band_roformer_instrumental_fullness_v2_gabox.ckpt",
    "MelBand Roformer Instrumental Fullness V3 (Gabox)": "mel_band_roformer_instrumental_fullness_v3_gabox.ckpt",
    "MelBand Roformer Instrumental Fullness V4 (Gabox)": "mel_band_roformer_instrumental_fullness_v4_gabox.ckpt",
    "MelBand Roformer Instrumental Fullness Noisy V4 (Gabox)": "mel_band_roformer_instrumental_fullness_noise_v4_gabox.ckpt",
    "MelBand Roformer INSTV5 (Gabox)": "mel_band_roformer_instrumental_instv5_gabox.ckpt",
    "MelBand Roformer INSTV5N (Gabox)": "mel_band_roformer_instrumental_instv5n_gabox.ckpt",
    "MelBand Roformer INSTV6 (Gabox)": "mel_band_roformer_instrumental_instv6_gabox.ckpt",
    "MelBand Roformer INSTV6N (Gabox)": "mel_band_roformer_instrumental_instv6n_gabox.ckpt",
    "MelBand Roformer INSTV7 (Gabox)": "mel_band_roformer_instrumental_instv7_gabox.ckpt",
    "MelBand Roformer INSTV7N (Gabox)": "mel_band_roformer_instrumental_instv7n_gabox.ckpt",
    "MelBand Roformer INSTV8 (Gabox)": "mel_band_roformer_instrumental_instv8_gabox.ckpt",
    "MelBand Roformer INSTV8N (Gabox)": "mel_band_roformer_instrumental_instv8n_gabox.ckpt",
    "MelBand Roformer Instrumental FV7z (Gabox)": "mel_band_roformer_instrumental_fv7z_gabox.ckpt",
    "MelBand Roformer Instrumental FV8 (Gabox)": "mel_band_roformer_instrumental_fv8_gabox.ckpt",
    "MelBand Roformer Instrumental FV8b (Gabox)": "mel_band_roformer_instrumental_fv8b_gabox.ckpt",
    "MelBand Roformer Instrumental FVX (Gabox)": "mel_band_roformer_instrumental_fvx_gabox.ckpt",
    "MelBand Roformer De-Reverb (anvuew) (SDR 19.17)": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "MelBand Roformer De-Reverb Less Aggressive (anvuew) (SDR 18.81)": "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
    "MelBand Roformer De-Reverb Mono (anvuew)": "dereverb_mel_band_roformer_mono_anvuew.ckpt",
    "MelBand Roformer De-Reverb Big (Sucial)": "dereverb_big_mbr_ep_362.ckpt",
    "MelBand Roformer De-Reverb Super Big (Sucial)": "dereverb_super_big_mbr_ep_346.ckpt",
    "MelBand Roformer De-Reverb-Echo (Sucial) (SDR 10.02)": "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
    "MelBand Roformer De-Reverb-Echo V2 (Sucial) (SDR 13.48)": "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
    "MelBand Roformer De-Reverb-Echo Fused (Sucial)": "dereverb_echo_mbr_fused.ckpt",
    "MelBand Roformer Kim SYHFT (SYH99999)": "MelBandRoformerSYHFT.ckpt",
    "MelBand Roformer Kim SYHFT V2 (SYH99999)": "MelBandRoformerSYHFTV2.ckpt",
    "MelBand Roformer Kim SYHFT V2.5 (SYH99999)": "MelBandRoformerSYHFTV2.5.ckpt",
    "MelBand Roformer Kim SYHFT V3 (SYH99999)": "MelBandRoformerSYHFTV3Epsilon.ckpt",
    "MelBand Roformer Kim Big SYHFT V1 (SYH99999)": "MelBandRoformerBigSYHFTV1.ckpt",
    "MelBand Roformer Kim Big Beta 4 FT (unwa)": "melband_roformer_big_beta4.ckpt",
    "MelBand Roformer Kim Big Beta 5e FT (unwa)": "melband_roformer_big_beta5e.ckpt",
    "MelBand Roformer Big Beta 6 (unwa)": "melband_roformer_big_beta6.ckpt",
    "MelBand Roformer Big Beta 6X (unwa)": "melband_roformer_big_beta6x.ckpt",
    "BS Roformer Chorus Male-Female (Sucial) (SDR 24.13)": "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt",
    "BS Roformer Male-Female (aufr33) (SDR 7.29)": "bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt",
    "MelBand Roformer Aspiration (Sucial) (SDR 18.98)": "aspiration_mel_band_roformer_sdr_18.9845.ckpt",
    "MelBand Roformer Aspiration Less Aggressive (Sucial) (SDR 18.12)": "aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt",
    "MelBand Roformer Bleed Suppressor V1 (unwa-97chris)": "mel_band_roformer_bleed_suppressor_v1.ckpt",
    "BS Roformer Vocals Resurrection (unwa)": "bs_roformer_vocals_resurrection_unwa.ckpt",
    "BS Roformer Instrumental Resurrection (unwa)": "bs_roformer_instrumental_resurrection_unwa.ckpt",
    "BS Roformer Instrumental Resurrection (Gabox)": "bs_roformer_instrumental_resurrection_gabox.ckpt",
    "BS Roformer SW (jarredou)": "BS-Roformer-SW.ckpt",
}
