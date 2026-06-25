from json import encoder

from api.models.static_model import (
    alphabet_model,
    alphabet_encoder,
    number_model,
    number_encoder
)

from api.models.dynamic_model import (
    model as dynamic_model,
    labels as dynamic_labels
)


def predict_static(features, mode):

    if mode == "letter":
        model = alphabet_model
        encoder = alphabet_encoder

    elif mode == "number":
        model = number_model
        encoder = number_encoder

    else:
        model = alphabet_model
        encoder = alphabet_encoder

    pred_idx = model.predict([features])[0]

    print("PRED IDX:", pred_idx)
    print("PRED IDX TYPE:", type(pred_idx))

    pred = encoder.inverse_transform([int(pred_idx)])[0]

    print("PRED LABEL:", pred)

    confidence = 0

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([features])[0]
        confidence = float(max(proba))

    return str(pred), confidence


def predict_dynamic(sequence):

    pred = dynamic_model.predict(
        sequence,
        verbose=0
    )[0]

    idx = pred.argmax()

    return (
        str(dynamic_labels[idx]),
        float(pred[idx])
    )