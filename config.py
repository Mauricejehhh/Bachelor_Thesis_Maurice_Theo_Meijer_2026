"""
Central configuration for the attention-fatigue EEG pipeline.
"""
FS_NOMINAL = 256
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FRONTAL_CHANNELS = ["AF7", "AF8"]
TEMPOROPARIETAL_CHANNELS = ["TP9", "TP10"]


BANDPASS_LOW = 1.0
BANDPASS_HIGH = 40.0
BANDPASS_ORDER = 4
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0

ARTIFACT_MAD_K = 5.0
ARTIFACT_FLAT_UV = 0.5

EEG_WINDOW_SEC = 4.0
EEG_WINDOW_OVERLAP = 0.5

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
PRIMARY_BANDS = ["theta", "alpha"]
EXPLORATORY_BANDS = ["beta"]

BEH_WINDOW_SIZE = 20
BEH_WINDOW_STEP = 1
FATIGUE_CONSEC_WINDOWS = 3

BASELINE_BLOCKS = ["practice", "no_feedback"]
MAIN_BLOCKS = ["block1", "block2", "block3"]

PRE_FATIGUE_SEC = 30.0

ALPHA_LEVEL = 0.05
SHAPIRO_ALPHA = 0.05
