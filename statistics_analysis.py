"""
Statistical analysis for the attention-fatigue EEG thesis.

For each participant, calculate descriptive baseline and pre-fatigue means.
Baseline = combined practice + no-feedback phase.
Pre-fatigue = final 30 s preceding the operational behavioural fatigue onset.

For group-level inference, use one value per participant:
pre-fatigue mean - baseline mean and test whether the group-level
mean difference differs from zero.

The participant is therefore the unit of statistical inference.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from config import BANDS, PRIMARY_BANDS, SHAPIRO_ALPHA, ALPHA_LEVEL


def _normality_ok(x: Sequence[float]) -> bool:
    """Return True when Shapiro-Wilk does not reject normality."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < 3:
        return False

    try:
        _, p = stats.shapiro(x)
        return p > SHAPIRO_ALPHA
    except Exception:
        return False


def _descriptive_comparison(baseline_windows_power: Sequence[float],
                            prefatigue_windows_power: Sequence[float],
                            label: str) -> Dict[str, object]:
    """
    Calculate descriptive baseline and pre-fatigue means for one participant.

    No inferential test is performed here. EEG windows are not treated as
    independent observations because they overlap and are nested within the
    same participant.
    """
    a = np.asarray(baseline_windows_power, dtype=float)
    b = np.asarray(prefatigue_windows_power, dtype=float)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    mean_baseline = np.mean(a) if len(a) else np.nan
    mean_prefatigue = np.mean(b) if len(b) else np.nan
    difference = (
        mean_prefatigue - mean_baseline
        if np.isfinite(mean_baseline) and np.isfinite(mean_prefatigue)
        else np.nan
    )

    return {
        "label": label,
        "test": "descriptive_only",
        "n_baseline_windows": len(a),
        "n_prefatigue_windows": len(b),
        "mean_baseline": mean_baseline,
        "mean_prefatigue": mean_prefatigue,
        "difference_prefatigue_minus_baseline": difference,
        "statistic": np.nan,
        "p_value": np.nan,
        "significant": np.nan,
    }


def per_participant_stats(baseline_df: pd.DataFrame, prefatigue_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce descriptive baseline-vs-pre-fatigue results for one participant.

    This preserves the existing output structure used by run_pipeline.py,
    but removes the scientifically inappropriate Welch/Mann-Whitney tests.
    """
    results = []

    channel_labels = {
        "frontal": "Frontal (AF7/AF8 avg)",
        "TP9": "TP9",
        "TP10": "TP10",
    }

    for ch_key, ch_label in channel_labels.items():
        for band in BANDS:
            col = f"{ch_key}_{band}_power"

            res = _descriptive_comparison(
                baseline_df[col],
                prefatigue_df[col],
                f"{ch_label} {band}",
            )

            res["channel"] = ch_key
            res["band"] = band
            res["outcome_type"] = (
                "primary"
                if (ch_key == "frontal" and band in PRIMARY_BANDS)
                else "exploratory"
            )

            results.append(res)

    return pd.DataFrame(results)


def group_level_stats(participant_means_df: pd.DataFrame, value_col: str,
                      group_col: Optional[str] = None) -> Dict[str, object]:
    """
    Test whether the participant-level baseline-normalised pre-fatigue
    values differ from zero.

    The input must contain one observation per participant. This is the
    appropriate unit of inference for the group-level EEG analysis.

    A one-sample t-test is used when the participant-level values satisfy
    the normality criterion. Otherwise, a Wilcoxon signed-rank test is used.
    """
    x = participant_means_df[value_col].dropna().to_numpy(dtype=float)

    if len(x) < 2:
        return {
            "n": len(x),
            "test": "insufficient_participants",
            "p_value": np.nan,
            "mean": np.mean(x) if len(x) else np.nan,
            "sd": np.std(x, ddof=1) if len(x) > 1 else np.nan,
            "statistic": np.nan,
            "significant": np.nan,
        }

    if _normality_ok(x):
        stat, p = stats.ttest_1samp(x, 0)
        test_name = "one-sample t-test"
    else:
        stat, p = stats.wilcoxon(x)
        test_name = "Wilcoxon signed-rank"

    return {
        "n": len(x),
        "test": test_name,
        "mean": np.mean(x),
        "sd": np.std(x, ddof=1),
        "statistic": stat,
        "p_value": p,
        "significant": p < ALPHA_LEVEL,
    }
