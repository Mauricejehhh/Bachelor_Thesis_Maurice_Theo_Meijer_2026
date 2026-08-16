"""
Combine EEG spectral features with the behavioural fatigue onset:
- identify the EEG time range corresponding to the combined baseline phase
- baseline-normalise every EEG window; subtract participant's own baseline mean
- extract the PRE_FATIGUE_SEC interval preceding fatigue onset
- express pre-fatigue EEG windows on a common "time-to-onset" axis for
  group-level alignment (fatigue onset = t = 0)
"""
from typing import Tuple

import pandas as pd

from config import BANDS, EEG_CHANNELS, BASELINE_BLOCKS, PRE_FATIGUE_SEC

POWER_COLS = (
    [f"{ch}_{b}_power" for ch in EEG_CHANNELS for b in BANDS]
    + [f"frontal_{b}_power" for b in BANDS]
)


def get_baseline_eeg_window(exp_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Elapsed-time range (seconds, EEG/EXP shared clock) spanned by the combined
    baseline phase (practice + no_feedback), using each trial's onset plus its
    duration to include the final trial fully.
    """
    baseline = exp_df[exp_df["block"].isin(BASELINE_BLOCKS)]
    t_start = baseline["time_in_experiment_s"].min()
    last = baseline.loc[baseline["time_in_experiment_s"].idxmax()]
    t_end = last["time_in_experiment_s"] + (last["trial_duration"] / 1000.0)
    return t_start, t_end


def compute_baseline_eeg_power(features_df: pd.DataFrame,
                               baseline_t_start: float,
                               baseline_t_end: float) -> pd.Series:
    """
    Mean power per channel/band across artifact-free EEG windows whose center
    falls inside the baseline time range.
    Returns a Series indexed by POWER_COLS.
    """
    mask = (
        (features_df["t_center"] >= baseline_t_start)
        & (features_df["t_center"] <= baseline_t_end)
        & (~features_df["artifact"])
    )
    baseline_windows = features_df[mask]
    if baseline_windows.empty:
        raise ValueError("No artifact-free windows in the baseline period.")
    return baseline_windows[POWER_COLS].mean()


def normalize_features(features_df: pd.DataFrame, baseline_power: pd.Series) -> pd.DataFrame:
    """
    Subtract the participant's baseline mean power from
    every EEG window (per column).
    """
    out = features_df.copy()
    for col in POWER_COLS:
        out[col] = out[col] - baseline_power[col]
    return out


def extract_prefatigue_interval(features_norm_df: pd.DataFrame, onset_time_s: float,
                                pre_sec: float = PRE_FATIGUE_SEC) -> pd.DataFrame:
    """
    EEG windows (already baseline-normalised) whose center falls within
    [onset_time_s - pre_sec, onset_time_s], excluding artifact windows.
    Adds a 'time_to_onset' column (<= 0, seconds; 0 = fatigue onset).
    """
    mask = (
        (features_norm_df["t_center"] >= onset_time_s - pre_sec)
        & (features_norm_df["t_center"] <= onset_time_s)
        & (~features_norm_df["artifact"])
    )
    pre = features_norm_df[mask].copy()
    pre["time_to_onset"] = pre["t_center"] - onset_time_s
    return pre
