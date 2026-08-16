# Attention-Fatigue EEG Pipeline

This repository implements the full analysis pipeline described in the Methodology section of my Bachelor Thesis. This pipeline can be summarized as: preprocessing → behavioural fatigue-onset detection → EEG spectral features → baseline normalisation → pre-fatigue extraction → statistics → figures.

## How to run

First, you want to make sure all the requirements are installed:
```bash
pip install -r requirements.txt
```

Once that is done, you can run the following command to execute the pipeline:
```bash
python run_pipeline.py --data-dir /path/to/folder/with/all/csvs --out-dir /path/to/results
```

Put every participant's `EEG-Data-P<id>.csv` and `EXP-Data-P<id>.csv` in the
same `--data-dir`. The script auto-discovers and pairs them, runs the full
pipeline per participant, then runs group-level statistics across everyone
who showed a detected fatigue onset.


## What each module does

| File | Role |
|---|---|
| `config.py` | Every numeric parameter as described in the Methodology. |
| `data_io.py` | Loads + resamples EEG onto a uniform time grid; loads/cleans the trial log; pairs participant files. |
| `eeg_preprocessing.py` | 1–40 Hz zero-phase Butterworth band-pass, 50 Hz notch, and artifact flagging. |
| `eeg_features.py` | 4 s / 50%-overlap windowing + Welch PSD → theta/alpha/beta power per channel and per window. |
| `behavioral_fatigue.py` | Baseline performance (practice+no_feedback), 20-trial/1-trial-step sliding window, fatigue-onset detection (3 consecutive windows above baseline error rate). |
| `fatigue_eeg_analysis.py` | Baseline EEG power, per-participant normalisation, 30s pre-fatigue-onset extraction, time-to-onset alignment. |
| `statistics_analysis.py` | Per-participant baseline-vs-pre-fatigue comparison (descriptive only — individual EEG windows are not treated as independent observations, so no test is run at this level). Group-level inferential testing (one-sample t-test if normality holds, otherwise Wilcoxon signed-rank) is applied only to the predefined primary outcomes (frontal theta and alpha power); TP9, TP10, and beta-band power are exploratory and reported descriptively only. |
| `run_pipeline.py` | Orchestrates everything: per-participant CSVs + diagnostic figures, group-level statistics, and the group-level figures (behavioural trajectories, baseline-vs-pre-fatigue comparisons, primary/exploratory change, temporal evolution). |

## Outputs (per participant, in `<out-dir>/P<id>/`)

- `behavioral_windows.csv` — every 20-trial sliding window's error rate/RT/RT-SD
- `eeg_window_features.csv` — every 4 s EEG window's band power per channel + artifact flag
- `summary.csv` — baseline performance, fatigue onset time, and `eeg_valid` (whether the baseline period had at least one artifact-free EEG window)
- `prefatigue_eeg_normalized.csv` — baseline-normalised power for the 30 s before onset (only if a fatigue onset was detected and EEG is valid)
- `prefatigue_eeg_raw.csv` — the same pre-fatigue window on raw (un-normalised) power, same scale as baseline (only if fatigue onset detected and EEG valid)
- `baseline_vs_prefatigue_stats.csv` — per-participant descriptive baseline-vs-pre-fatigue comparison
- `P<id>_diagnostics.png` — error-rate timeline + frontal power before onset
- `P<id>_frontal_theta_baseline_vs_prefatigue.png`, `P<id>_frontal_alpha_baseline_vs_prefatigue.png` — per-participant baseline vs. pre-fatigue boxplots for the primary frontal outcomes

Participants without a detected fatigue onset, or whose baseline EEG period has no artifact-free windows, only get `behavioral_windows.csv`, `eeg_window_features.csv`, and `summary.csv` — they're excluded from all downstream EEG-based analyses and figures.

## Outputs (group-level, at the top of `<out-dir>/`)

- `all_participants_summary.csv` — one row per participant, all participants
- `group_participant_means.csv`, `group_level_stats.csv` — participant-level primary/exploratory differences and the corresponding group-level tests (inferential for frontal theta/alpha, descriptive-only for TP9, TP10, and beta)
- `group_behavioral_trajectories.png` — behavioural error-rate trajectories for every participant with a detected onset and valid EEG data, aligned to their own fatigue onset (t=0)
- `group_frontal_theta_baseline_vs_prefatigue.png`, `group_frontal_alpha_baseline_vs_prefatigue.png` — participant-level frontal theta/alpha power, baseline vs. pre-fatigue
- `group_primary_change.png` — participant-level change (pre-fatigue minus baseline) in frontal theta/alpha power
- `group_exploratory_change.png` — the same, exploratory outcomes only (TP9, TP10, frontal beta)
- `group_temporal_evolution.png` — group mean ± SE of baseline-normalised frontal theta/alpha power over the 30 s preceding onset, time-to-onset axis (0 = onset)
