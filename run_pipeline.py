"""
End-to-end pipeline for the attention-fatigue EEG thesis.
"""
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))

from config import BANDS, PRIMARY_BANDS, PRE_FATIGUE_SEC, MAIN_BLOCKS
from data_io import load_eeg, load_exp, discover_participants
from eeg_preprocessing import preprocess_eeg
from eeg_features import extract_window_features
from behavioral_fatigue import (
    compute_baseline_performance, sliding_window_performance, detect_fatigue_onset,
)
from fatigue_eeg_analysis import (
    get_baseline_eeg_window, compute_baseline_eeg_power, normalize_features,
    extract_prefatigue_interval,
)
from statistics_analysis import per_participant_stats, group_level_stats


def process_participant(pid: str, eeg_path: Union[str, Path], exp_path: Union[str, Path],
                        out_dir: Union[str, Path]) -> Tuple[Dict[str, object],
                                                             Optional[pd.DataFrame],
                                                             Optional[Dict[str, object]]]:
    """
    Run the full single-participant pipeline (steps 1-7 in the module
    docstring): load, filter, behavioural fatigue detection, EEG spectral
    features, baseline normalisation, per-participant stats, and figures.

    Returns (result, stats_df, p_means):
        result   : dict of summary fields for this participant (always returned)
        stats_df : per-participant baseline-vs-prefatigue stats DataFrame,
                   or None if no fatigue onset was detected
        p_means  : dict of participant-level band/channel differences for the
                   group-level test, or None if no fatigue onset was detected
    """
    print(f"\n=== Participant {pid} ===")
    out_dir = Path(out_dir) / f"P{pid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg_df, fs = load_eeg(eeg_path)
    exp_df = load_exp(exp_path)
    print(f"  EEG: {len(eeg_df)} samples @ {fs:.2f} Hz ({eeg_df['t'].iloc[-1]:.1f}s)")
    print(f"  EXP: {len(exp_df)} trials, blocks={exp_df['block'].unique().tolist()}")

    eeg_clean = preprocess_eeg(eeg_df, fs)

    baseline_perf = compute_baseline_performance(exp_df)
    windows_perf = sliding_window_performance(exp_df, blocks=MAIN_BLOCKS)
    onset = detect_fatigue_onset(windows_perf, baseline_perf["error_rate"])
    windows_perf.to_csv(out_dir / "behavioral_windows.csv", index=False)

    print(f"  Baseline error rate: {baseline_perf['error_rate']:.3f} "
          f"(mean RT {baseline_perf['mean_rt']:.1f} ms, n={baseline_perf['n_trials']})")
    if onset["fatigue_detected"]:
        print(f"  Fatigue onset detected at t={onset['onset_time_s']:.1f}s "
              f"(trial idx {onset['onset_trial_idx']})")
    else:
        print("  No sustained fatigue onset detected for this participant.")

    features = extract_window_features(eeg_clean, fs)
    b_start, b_end = get_baseline_eeg_window(exp_df)
    main_t_start = exp_df.loc[exp_df["block"].isin(MAIN_BLOCKS), "time_in_experiment_s"].min()
    main_t_end = exp_df.loc[exp_df["block"].isin(MAIN_BLOCKS), "time_in_experiment_s"].max()
    features["phase"] = np.select(
        [
            (features["t_center"] >= b_start) & (features["t_center"] <= b_end),
            (features["t_center"] >= main_t_start) & (features["t_center"] <= main_t_end),
        ],
        ["baseline", "main"],
        default="other",
    )
    n_artifact = features["artifact"].sum()
    print(f"  EEG windows: {len(features)} total, {n_artifact} flagged as artifact "
          f"({100 * n_artifact / len(features):.1f}%)")
    features.to_csv(out_dir / "eeg_window_features.csv", index=False)

    result = {
        "participant": pid, "fs": fs, "n_eeg_windows": len(features),
        "n_artifact_windows": int(n_artifact), **baseline_perf, **onset,
    }

    if not onset["fatigue_detected"]:
        pd.DataFrame([result]).to_csv(out_dir / "summary.csv", index=False)
        return result, None, None

    baseline_power = compute_baseline_eeg_power(features, b_start, b_end)
    features_norm = normalize_features(features, baseline_power)
    prefatigue = extract_prefatigue_interval(features_norm, onset["onset_time_s"])
    prefatigue.to_csv(out_dir / "prefatigue_eeg_normalized.csv", index=False)

    prefatigue_raw = extract_prefatigue_interval(features, onset["onset_time_s"])
    prefatigue_raw.to_csv(out_dir / "prefatigue_eeg_raw.csv", index=False)

    baseline_raw = features[
        (features["t_center"] >= b_start) & (features["t_center"] <= b_end)
        & (~features["artifact"])
    ]

    print(f"  Pre-fatigue window: {len(prefatigue_raw)} EEG windows in the "
          f"final {PRE_FATIGUE_SEC:.0f}s before onset")

    stats_df = per_participant_stats(baseline_raw, prefatigue_raw)
    stats_df.to_csv(out_dir / "baseline_vs_prefatigue_stats.csv", index=False)
    print(f"  Baseline: {len(baseline_raw)} clean windows | Pre-fatigue: {len(prefatigue_raw)} clean windows")
    primary = stats_df[stats_df["outcome_type"] == "primary"]
    cols = [c for c in ["label", "test", "mean_baseline", "mean_prefatigue",
                         "difference_prefatigue_minus_baseline"] if c in primary.columns]
    print(primary[cols].to_string(index=False))

    p_means = {"participant": pid}
    for ch_key in ["frontal", "TP9", "TP10"]:
        for band in BANDS:
            row = stats_df[(stats_df["channel"] == ch_key) & (stats_df["band"] == band)]
            p_means[f"{ch_key}_{band}_power"] = row["difference_prefatigue_minus_baseline"].iloc[0]

    pd.DataFrame([result]).to_csv(out_dir / "summary.csv", index=False)

    make_participant_figure(pid, windows_perf, baseline_perf, onset, prefatigue, out_dir)

    make_baseline_prefatigue_figure(pid, baseline_raw, prefatigue_raw, "theta", out_dir)
    make_baseline_prefatigue_figure(pid, baseline_raw, prefatigue_raw, "alpha", out_dir)

    return result, stats_df, p_means


def make_participant_figure(pid: str, windows_perf: pd.DataFrame, baseline_perf: Dict[str, object],
                            onset: Dict[str, object], prefatigue: Optional[pd.DataFrame],
                            out_dir: Union[str, Path]) -> None:
    """
    Save a two-panel diagnostic figure for one participant: behavioural error
    rate over time (top) and baseline-normalised frontal spectral power in
    the pre-fatigue interval (bottom, only if a fatigue onset was detected).
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    ax = axes[0]
    ax.plot(windows_perf["t_start_s"], windows_perf["error_rate"], color="steelblue")
    ax.axhline(baseline_perf["error_rate"], color="gray", linestyle="--", label="baseline error rate")
    if onset["fatigue_detected"]:
        ax.axvline(onset["onset_time_s"], color="crimson", linestyle=":", label="fatigue onset")
    ax.set_xlabel("Time in experiment (s)")
    ax.set_ylabel("Error rate (20-trial window)")
    ax.set_title(f"Participant {pid}: behavioural error rate over time")
    ax.legend()

    ax = axes[1]
    if prefatigue is not None and len(prefatigue):
        for band, color in zip(BANDS, ["tab:orange", "tab:green", "tab:purple"]):
            ax.plot(prefatigue["time_to_onset"], prefatigue[f"frontal_{band}_power"],
                     marker="o", label=f"frontal {band}", color=color)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("Time to fatigue onset (s)")
        ax.set_ylabel("Baseline-normalised power")
        ax.set_title(f"Participant {pid}: frontal spectral power before fatigue onset")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No fatigue onset detected", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / f"P{pid}_diagnostics.png", dpi=150)
    plt.close(fig)


def make_baseline_prefatigue_figure(pid: str, baseline_raw: pd.DataFrame, prefatigue: pd.DataFrame,
                                    band: str, out_dir: Union[str, Path]) -> None:
    """
    Frontal {band} baseline vs. pre-fatigue figure.

    Shows the per-window distribution (boxplot) of frontal <band> power
    during the baseline period (practice + no-feedback) against the final
    PRE_FATIGUE_SEC window preceding the detected fatigue onset, with the
    per-condition mean overlaid. This is the direct visual counterpart to
    the descriptive comparison computed in per_participant_stats().
    """
    col = f"frontal_{band}_power"

    baseline_vals = baseline_raw[col].dropna().to_numpy(dtype=float)
    prefatigue_vals = prefatigue[col].dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(4.5, 5))

    data = [baseline_vals, prefatigue_vals]
    labels = ["Baseline", "Pre-fatigue"]
    bp = ax.boxplot(data, tick_labels=labels, widths=0.5, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "crimson",
                                "markeredgecolor": "crimson"})

    for i, vals in enumerate(data, start=1):
        if len(vals):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=12, alpha=0.4,
                       color="steelblue", zorder=1)

    ax.set_ylabel(f"Frontal {band} power (AF7/AF8 avg)")
    ax.set_title(f"Participant {pid}: frontal {band} baseline vs. pre-fatigue")
    fig.tight_layout()
    fig.savefig(out_dir / f"P{pid}_frontal_{band}_baseline_vs_prefatigue.png", dpi=150)
    plt.close(fig)


def _group_descriptive_only(participant_means_df: pd.DataFrame, value_col: str) -> Dict[str, object]:
    """
    Descriptive-only group-level summary (no inferential test), for
    exploratory outcomes. Per the methodology, only the predefined primary
    outcomes (frontal theta and frontal alpha power) are tested against
    zero; beta-band activity and the TP9/TP10 electrodes are exploratory
    and are interpreted descriptively, not tested for significance.
    """
    x = participant_means_df[value_col].dropna().to_numpy(dtype=float)
    return {
        "n": len(x),
        "test": "descriptive_only",
        "mean": np.mean(x) if len(x) else np.nan,
        "sd": np.std(x, ddof=1) if len(x) > 1 else np.nan,
        "statistic": np.nan,
        "p_value": np.nan,
        "significant": np.nan,
    }


def group_analysis(participant_means: List[Dict[str, object]], out_dir: Union[str, Path]) -> None:
    """
    Run group-level statistics (step 8-9 in the module docstring) across all
    participants with a detected fatigue onset. Only the predefined primary
    outcomes (frontal theta and alpha power) are tested against zero
    (one-sample t-test / Wilcoxon signed-rank); the remaining channel/band
    combinations (TP9, TP10, and beta) are summarised descriptively only.
    Saves a group participant-means CSV and a group-level stats CSV.
    """
    out_dir = Path(out_dir)
    df = pd.DataFrame(participant_means)
    if df.empty:
        print("\n[group_analysis] No participants with detected fatigue onset -- skipping group stats.")
        return
    df.to_csv(out_dir / "group_participant_means.csv", index=False)

    rows = []
    for ch_key in ["frontal", "TP9", "TP10"]:
        for band in BANDS:
            col = f"{ch_key}_{band}_power"
            is_primary = (ch_key == "frontal" and band in PRIMARY_BANDS)
            if is_primary:
                res = group_level_stats(df, col)
            else:
                res = _group_descriptive_only(df, col)
            res.update({
                "channel": ch_key, "band": band,
                "outcome_type": "primary" if is_primary else "exploratory",
            })
            rows.append(res)
    group_stats_df = pd.DataFrame(rows)
    group_stats_df.to_csv(out_dir / "group_level_stats.csv", index=False)

    print(f"\n=== Group-level results (n={len(df)} participant(s) with fatigue onset) ===")
    cols = [c for c in ["channel", "band", "outcome_type", "n", "test", "mean", "p_value"]
            if c in group_stats_df.columns]
    print(group_stats_df[cols].to_string(index=False))

    if len(df) < 2:
        print("\nNOTE: group-level inferential stats require >=2 participants with a "
              "detected fatigue onset. With n=1 these numbers are descriptive only.")


def main() -> None:
    """
    Parse CLI arguments, discover participants, run the per-participant
    pipeline for each, then run group-level analysis across all participants
    with a detected fatigue onset.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    participants = discover_participants(args.data_dir)
    if not participants:
        print(f"No matching EEG-Data-P*/EXP-Data-P* pairs found in {args.data_dir}")
        return

    all_results = []
    participant_means = []
    for pid, eeg_path, exp_path in participants:
        result, stats_df, p_means = process_participant(pid, eeg_path, exp_path, args.out_dir)
        all_results.append(result)
        if p_means is not None:
            participant_means.append(p_means)

    pd.DataFrame(all_results).to_csv(Path(args.out_dir) / "all_participants_summary.csv", index=False)
    group_analysis(participant_means, args.out_dir)


if __name__ == "__main__":
    main()
