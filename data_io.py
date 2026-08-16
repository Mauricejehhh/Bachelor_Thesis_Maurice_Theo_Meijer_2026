"""
Loading and light cleaning of the raw CSV files.

Expected file naming convention:
    EEG-Data-P<id>.csv
    EXP-Data-P<id>.csv

No explicit clock-alignment procedure is required: both streams start together,
so we align them on elapsed time since each recording's own start.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import EEG_CHANNELS


def load_eeg(path: Union[str, Path], target_fs: Optional[float] = None) -> Tuple[pd.DataFrame, float]:
    """
    Load a Muse EEG CSV and resample onto a uniform time grid.

    Returns
    -------
    df : DataFrame with columns ['t', 'TP9', 'AF7', 'AF8', 'TP10']
         't' is elapsed time in seconds from the first sample (0-based).
    fs : float, the (measured) uniform sampling rate used for resampling.
    """
    raw = pd.read_csv(path, sep=";")
    raw = raw.sort_values("timestamps").reset_index(drop=True)

    ts = raw["timestamps"].to_numpy(dtype=float)
    t_elapsed = ts - ts[0]

    dt = np.median(np.diff(ts))
    measured_fs = 1.0 / dt
    fs = target_fs if target_fs is not None else measured_fs

    n_samples = int(np.floor(t_elapsed[-1] * fs)) + 1
    t_uniform = np.arange(n_samples) / fs

    df = pd.DataFrame({"t": t_uniform})
    for ch in EEG_CHANNELS:
        df[ch] = np.interp(t_uniform, t_elapsed, raw[ch].to_numpy(dtype=float))

    return df, fs


def load_exp(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the behavioural/experiment CSV.

    Returns
    -------
    df : DataFrame, one row per trial, with numeric columns coerced and
         time columns converted to seconds ('time_in_experiment_s',
         'trial_onset_s').
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]

    numeric_cols = [
        "block_trial_counter", "current_number", "hand_pos", "responded",
        "rt", "step_size", "subject_nr", "t0", "time_in_block",
        "time_in_experiment", "trial_counter", "trial_duration", "trial_onset",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time_in_experiment_s"] = df["time_in_experiment"] / 1000.0
    df["trial_onset_s"] = df["trial_onset"] / 1000.0

    df["is_error"] = df["correct"].isin(["omission", "commission"])
    df = df.sort_values("time_in_experiment_s").reset_index(drop=True)
    return df


def discover_participants(data_dir: Union[str, Path]) -> List[Tuple[str, Path, Path]]:
    """
    Find matching (eeg_path, exp_path, participant_id) triples in a directory,
    based on filenames 'EEG-Data-P<id>.csv' / 'EXP-Data-P<id>.csv'.
    """
    data_dir = Path(data_dir)
    eeg_files = {}
    exp_files = {}
    pat_eeg = re.compile(r"EEG-Data-P(\w+)\.csv$", re.IGNORECASE)
    pat_exp = re.compile(r"EXP-Data-P(\w+)\.csv$", re.IGNORECASE)

    for f in data_dir.glob("*.csv"):
        m = pat_eeg.search(f.name)
        if m:
            eeg_files[m.group(1)] = f
        m = pat_exp.search(f.name)
        if m:
            exp_files[m.group(1)] = f

    participants = []
    for pid in sorted(set(eeg_files) & set(exp_files)):
        participants.append((pid, eeg_files[pid], exp_files[pid]))

    missing = set(eeg_files) ^ set(exp_files)
    if missing:
        print(f"[data_io] Warning: unmatched EEG/EXP files for id(s): {sorted(missing)}")

    return participants
