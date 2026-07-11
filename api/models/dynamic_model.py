import numpy as np

from tensorflow.keras.models import load_model
from api.config import (
    DYNAMIC_MODEL_PATH,
    DYNAMIC_LABELS_PATH
)

model = load_model(DYNAMIC_MODEL_PATH)

labels = np.load(
    DYNAMIC_LABELS_PATH,
    allow_pickle=True
)