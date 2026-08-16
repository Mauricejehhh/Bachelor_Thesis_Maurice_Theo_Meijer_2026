"""
Segment continuous EEG into overlapping windows and compute per-band spectral
power (Welch's method) for each window.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid

from config import (
    EEG_WINDOW_SEC, EEG_WINDOW_OVERLAP, BANDS,
    EEG_CHANNELS, FRONTAL_CHANNELS,
)
from eeg_preprocessing import is_artifact_window, compute_artifact_thresholds


def _band_power(sig: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    """
    Integrate the Welch PSD of `sig` within `band` (Hz)
    using the trapezoidal rule.
    """
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 2)))
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.nan
    return trapezoid(psd[mask], freqs[mask])


def extract_window_features(eeg_df: pd.DataFrame, fs: float,
                            window_sec: float = EEG_WINDOW_SEC,
                            overlap: float = EEG_WINDOW_OVERLAP) -> pd.DataFrame:
    """
    Slide a window_sec-long, `overlap`-overlapping window across the
    preprocessed EEG and compute band power per channel and per band.

    Returns a DataFrame with one row per EEG window:
        t_start, t_end, t_center, artifact,
        <channel>_<band>_power  for every channel x band,
        frontal_<band>_power    (mean of AF7/AF8),
        tp9_<band>_power, tp10_<band>_power (kept separate, per methodology)
    """
    win_n = int(round(window_sec * fs))
    step_n = int(round(win_n * (1 - overlap)))
    n_samples = len(eeg_df)

    starts = list(range(0, n_samples - win_n + 1, step_n))

    ptp_by_channel = {ch: [] for ch in EEG_CHANNELS}
    for start in starts:
        window = eeg_df.iloc[start:start + win_n]
        for ch in EEG_CHANNELS:
            ptp_by_channel[ch].append(window[ch].max() - window[ch].min())
    thresholds = compute_artifact_thresholds(ptp_by_channel)

    rows = []
    for start in starts:
        end = start + win_n
        window = eeg_df.iloc[start:end]

        row = {
            "t_start": window["t"].iloc[0],
            "t_end": window["t"].iloc[-1],
            "t_center": (window["t"].iloc[0] + window["t"].iloc[-1]) / 2.0,
            "artifact": is_artifact_window(window, thresholds),
        }

        for ch in EEG_CHANNELS:
            sig = window[ch].to_numpy(dtype=float)
            for band_name, band_range in BANDS.items():
                row[f"{ch}_{band_name}_power"] = _band_power(sig, fs, band_range)

        for band_name in BANDS:
            row[f"frontal_{band_name}_power"] = np.mean(
                [row[f"{ch}_{band_name}_power"] for ch in FRONTAL_CHANNELS]
            )

        rows.append(row)

    return pd.DataFrame(rows)
