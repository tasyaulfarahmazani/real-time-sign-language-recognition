import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# MODEL PATH
STATIC_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "static",
    "best_rf_model.joblib"
)

DYNAMIC_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "dynamic",
    "dynamic_gru.keras"
)

DYNAMIC_LABELS_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "dynamic",
    "dynamic_labels.npy"
)

HAND_LANDMARKER = os.path.join(
    BASE_DIR,
    "assets",
    "hand_landmarker.task"
)
print(HAND_LANDMARKER)

SEQUENCE_LENGTH = 30

STATIC_THRESHOLD = 40
DYNAMIC_THRESHOLD = 60

PREDICTION_COOLDOWN = 15

ALPHABET_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "static",
    "alphabet_rf.joblib"
)

NUMBER_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "static",
    "number_rf.joblib"
)