import joblib
from api.config import (
    ALPHABET_MODEL_PATH,
    NUMBER_MODEL_PATH
)

alphabet_bundle = joblib.load(ALPHABET_MODEL_PATH)
number_bundle = joblib.load(NUMBER_MODEL_PATH)

alphabet_model = alphabet_bundle["model"]
alphabet_encoder = alphabet_bundle["label_encoder"]

number_model = number_bundle["model"]
number_encoder = number_bundle["label_encoder"]