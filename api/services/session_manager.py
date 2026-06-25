from collections import deque

from api.config import (
    SEQUENCE_LENGTH,
    PREDICTION_COOLDOWN
)

session_data = {}

def get_session(session_id):

    if session_id not in session_data:

        session_data[session_id] = {
            "buffer": deque(maxlen=SEQUENCE_LENGTH),
            "cooldown": 0,
            "last_pred": None,
            "last_conf": 0
        }

    return session_data[session_id]


def clear_session(session_id):

    if session_id in session_data:
        del session_data[session_id]