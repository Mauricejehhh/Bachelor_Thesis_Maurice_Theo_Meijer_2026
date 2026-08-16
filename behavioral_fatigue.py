"""
Behavioural analysis: baseline estimation and sliding-window fatigue-onset
detection, following the thesis's Section "Behavioural Fatigue Detection".
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    BASELINE_BLOCKS, MAIN_BLOCKS, BEH_WINDOW_SIZE, BEH_WINDOW_STEP,
    FATIGUE_CONSEC_WINDOWS,
)


def compute_baseline_performance(exp_df: pd.DataFrame) -> Dict[str, float]:
    """
    Baseline = average performance across practice + no_feedback blocks.
    Returns dict: error_rate, mean_rt, rt_std
    """
    baseline = exp_df[exp_df["block"].isin(BASELINE_BLOCKS)]
    error_rate = baseline["is_error"].mean()

    go_correct = baseline[(baseline["go_type"] == "Go") & (baseline["correct"] == "correct")]
    mean_rt = go_correct["rt"].mean()
    rt_std = go_correct["rt"].std()

    return {"error_rate": error_rate, "mean_rt": mean_rt, "rt_std": rt_std,
            "n_trials": len(baseline)}


def sliding_window_performance(exp_df: pd.DataFrame, window_size: int = BEH_WINDOW_SIZE,
                               step: int = BEH_WINDOW_STEP,
                               blocks: List[str] = MAIN_BLOCKS) -> pd.DataFrame:
    """
    Compute error rate / mean RT / RT variability over overlapping windows of
    `window_size` consecutive trials (step = `step` trial) within the main
    experimental blocks, in chronological order.

    Returns a DataFrame with one row per window:
        window_idx, trial_start_idx, trial_end_idx,
        t_start_s, t_end_s (time_in_experiment of first/last trial in window),
        error_rate, mean_rt, rt_std
    """
    main = exp_df[exp_df["block"].isin(blocks)].sort_values("time_in_experiment_s").reset_index(drop=True)
    n_trials = len(main)

    rows = []
    idx = 0
    window_idx = 0
    while idx + window_size <= n_trials:
        chunk = main.iloc[idx: idx + window_size]
        go_correct = chunk[(chunk["go_type"] == "Go") & (chunk["correct"] == "correct")]

        rows.append({
            "window_idx": window_idx,
            "trial_start_idx": idx,
            "trial_end_idx": idx + window_size - 1,
            "t_start_s": chunk["time_in_experiment_s"].iloc[0],
            "t_end_s": chunk["time_in_experiment_s"].iloc[-1],
            "error_rate": chunk["is_error"].mean(),
            "mean_rt": go_correct["rt"].mean() if len(go_correct) else np.nan,
            "rt_std": go_correct["rt"].std() if len(go_correct) > 1 else np.nan,
        })
        idx += step
        window_idx += 1

    return pd.DataFrame(rows)


def detect_fatigue_onset(windows_df: pd.DataFrame, baseline_error_rate: float,
                         consec: int = FATIGUE_CONSEC_WINDOWS) -> Dict[str, object]:
    """
    First of `consec` consecutive windows whose error_rate exceeds the
    participant's baseline error rate.

    Returns a dict with onset info and 'fatigue_detected': True, or
    {'fatigue_detected': False} if the criterion was never met.
    """
    above = (windows_df["error_rate"] > baseline_error_rate).to_numpy()

    run_len = 0
    for i, flag in enumerate(above):
        run_len = run_len + 1 if flag else 0
        if run_len == consec:
            onset_window_idx = i - consec + 1
            onset_row = windows_df.iloc[onset_window_idx]
            return {
                "onset_window_idx": int(onset_window_idx),
                "onset_trial_idx": int(onset_row["trial_start_idx"]),
                "onset_time_s": float(onset_row["t_start_s"]),
                "fatigue_detected": True,
            }
    return {"fatigue_detected": False}
