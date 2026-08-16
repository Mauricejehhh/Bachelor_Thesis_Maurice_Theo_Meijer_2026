# Attention-Fatigue EEG Pipeline

This repository implements the full analysis pipeline described in the Methodology section of my Bachelor Thesis. This pipeline can be summarized as: preprocessing → behavioural fatigue-onset detection → EEG spectral features → baseline normalisation → pre-fatigue extraction → statistics.

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
| `statistics_analysis.py` | Per-participant baseline-vs-pre-fatigue tests (normality-checked: t-test or Mann-Whitney/Wilcoxon) and group-level one-sample tests on normalised power. |
| `run_pipeline.py` | Orchestrates everything, saves CSVs + a diagnostic figure per participant, then group summary. |

## Outputs (per participant, in `<out-dir>/P<id>/`)

- `behavioral_windows.csv` — every 20-trial sliding window's error rate/RT/RT-SD
- `eeg_window_features.csv` — every 4 s EEG window's band power per channel + artifact flag
- `summary.csv` — baseline performance + fatigue onset time
- `prefatigue_eeg_normalized.csv` — baseline-normalised power for the 30 s before onset (only if fatigue detected)
- `baseline_vs_prefatigue_stats.csv` — per-participant statistical tests
- `P<id>_diagnostics.png` — error-rate timeline + frontal power before onset

And at the top level: `all_participants_summary.csv`, `group_participant_means.csv`,
`group_level_stats.csv`.
