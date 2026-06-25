import numpy as np

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from api.config import SEQUENCE_LENGTH

from api.services.feature_extractor import (
    decode_frame,
    extract_features_static,
    extract_features_dynamic
)

from api.services.hand_detector import detector

from api.services.predictor import (
    predict_static,
    predict_dynamic
)

from api.services.session_manager import (
    get_session
)

router = APIRouter()


# =====================================================
# STATIC (HURUF / ANGKA)
# =====================================================
@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = "letter"
):
    contents = await file.read()

    frame, mp_image = decode_frame(contents)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Gambar tidak valid"
        )

    result = detector.detect(mp_image)

    print("HANDS:", len(result.hand_landmarks))

    if not result.hand_landmarks:
        return {
            "prediction": "",
            "confidence": 0
        }

    features = extract_features_static(
        result.hand_landmarks,
        result.handedness
    )

    print("FEATURES:", features.shape)

    prediction, confidence = predict_static(
        features,
        mode
    )

    print("PRED:", prediction)
    print("CONF:", confidence)

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    }


# =====================================================
# DYNAMIC (KATA)
# =====================================================
@router.post("/predict-dynamic")
async def predict_dynamic_route(
    file: UploadFile = File(...),
    session_id: str = "default"
):
    contents = await file.read()

    frame, mp_image = decode_frame(contents)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Gambar tidak valid"
        )

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return {
            "prediction": "",
            "confidence": 0
        }

    features = extract_features_dynamic(
        result.hand_landmarks
    )

    print("DYNAMIC FEATURES:", features.shape)

    session = get_session(session_id)

    session["buffer"].append(features)

    print("BUFFER:", len(session["buffer"]))

    if len(session["buffer"]) < SEQUENCE_LENGTH:
        return {
            "prediction": "",
            "confidence": 0
        }

    sequence = np.array(
        session["buffer"],
        dtype=np.float32
    )

    sequence = np.expand_dims(
        sequence,
        axis=0
    )

    print("SEQUENCE:", sequence.shape)

    prediction, confidence = predict_dynamic(
        sequence
    )

    print("WORD:", prediction)
    print("CONF:", confidence)

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    }