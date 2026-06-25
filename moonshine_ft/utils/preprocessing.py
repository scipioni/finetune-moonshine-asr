"""
Audio preprocessing utilities for Moonshine ASR.
"""

import re
import unicodedata

import numpy as np

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize transcript for WER computation and training targets.

    Lowercases, strips punctuation, and collapses whitespace.
    Preserves Italian accented characters (à è é ì ò ù).
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def normalize_audio(audio_data: np.ndarray, target_rms: float = 0.075) -> np.ndarray:
    """
    Normalize audio to target RMS amplitude.

    Args:
        audio_data: Input audio array
        target_rms: Target RMS value (default: 0.075, matches training data)

    Returns:
        Normalized audio array
    """
    rms = np.sqrt(np.mean(audio_data**2))
    if rms > 0.001:  # Avoid division by very small numbers
        scale_factor = target_rms / rms
        normalized = audio_data * scale_factor
        return np.clip(normalized, -1.0, 1.0)
    return audio_data


def pad_audio(
    audio_data: np.ndarray,
    target_duration: float = 2.0,
    sample_rate: int = 16000,
    mode: str = "center"
) -> np.ndarray:
    """
    Pad audio to target duration.

    Args:
        audio_data: Input audio array
        target_duration: Target duration in seconds
        sample_rate: Audio sample rate (default: 16000)
        mode: Padding mode - "center" (default), "start", or "end"

    Returns:
        Padded audio array
    """
    target_samples = int(target_duration * sample_rate)
    current_samples = len(audio_data)

    if current_samples >= target_samples:
        return audio_data

    pad_total = target_samples - current_samples

    if mode == "center":
        # Center the audio with silence padding on both sides
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
    elif mode == "start":
        # Pad at the start
        pad_left = pad_total
        pad_right = 0
    elif mode == "end":
        # Pad at the end
        pad_left = 0
        pad_right = pad_total
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    return np.pad(audio_data, (pad_left, pad_right), mode='constant', constant_values=0)


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Note: This is a simple implementation. For production use, consider using
    librosa.resample or scipy.signal.resample for higher quality.

    Args:
        audio_data: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio_data

    # Simple resampling using linear interpolation
    duration = len(audio_data) / orig_sr
    target_length = int(duration * target_sr)

    indices = np.linspace(0, len(audio_data) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)

    return resampled
