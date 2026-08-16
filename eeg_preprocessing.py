"""
EEG preprocessing: band-pass + notch filtering,
and window-level artifact flagging.
"""

from typing import Dict, List, Sequence, pd

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from config import (
    BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
    NOTCH_FREQ, NOTCH_Q, EEG_CHANNELS,
    ARTIFACT_MAD_K, ARTIFACT_FLAT_UV,
)


def _bandpass_filter(signal: np.ndarray, fs: float,
                     low: float = BANDPASS_LOW,
                     high: float = BANDPASS_HIGH,
                     order: int = BANDPASS_ORDER) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter to a single-channel signal.
    Returns the filtered signal (same shape as input).
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def _notch_filter(signal: np.ndarray, fs: float,
                  freq: float = NOTCH_FREQ, q: float = NOTCH_Q) -> np.ndarray:
    """
    Apply a zero-phase notch filter to a single-channel signal to suppress
    electrical line noise at freq.
    Returns the filtered signal (same shape as input).
    """
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, signal)


def preprocess_eeg(df: pd.DataFrame, fs: float) -> pd.DataFrame:
    """
    Apply 1-40 Hz zero-phase Butterworth band-pass and a 50 Hz notch filter
    to every EEG channel.
    Returns a new DataFrame (same shape/columns as input).
    """
    out = df.copy()
    for ch in EEG_CHANNELS:
        sig = df[ch].to_numpy(dtype=float)
        sig = _bandpass_filter(sig, fs)
        sig = _notch_filter(sig, fs)
        out[ch] = sig
    return out


def compute_artifact_thresholds(window_ptp_by_channel: Dict[str, Sequence[float]],
                                k: float = ARTIFACT_MAD_K) -> Dict[str, float]:
    """
    Given a dict {channel: array of per-window peak-to-peak amplitudes} for one
    recording, compute a robust per-channel upper threshold = median + k*MAD.
    This adapts to each channel's own noise floor
    instead of using one fixed cutoff.
    """
    thresholds = {}
    for ch, vals in window_ptp_by_channel.items():
        vals = np.asarray(vals, dtype=float)
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        thresholds[ch] = med + k * 1.4826 * mad
    return thresholds


def is_artifact_window(window_df: pd.DataFrame,
                       thresholds: Dict[str, float],
                       channels: List[str] = EEG_CHANNELS,
                       flat_thresh: float = ARTIFACT_FLAT_UV) -> bool:
    """
    Flags a window if ANY channel has peak-to-peak amplitude above its
    recording-specific robust threshold (movement / poor contact) or below
    flat_thresh (disconnected electrode). See compute_artifact_thresholds().
    """
    for ch in channels:
        ptp = window_df[ch].max() - window_df[ch].min()
        if ptp > thresholds[ch] or ptp < flat_thresh:
            return True
    return False
