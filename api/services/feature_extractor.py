import cv2
import numpy as np
import mediapipe as mp

def extract_features_static(hand_landmarks_list, handedness_list) -> np.ndarray:
    features = np.zeros(126)

    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        if hand_idx >= 2:
            break

        label = handedness_list[hand_idx][0].category_name  # 'Left' / 'Right'

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        )  # (21, 3)

        # Normalisasi 1: Translation
        landmarks -= landmarks[0].copy()

        # Normalisasi 2: Scale
        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        if max_dist > 0:
            landmarks /= max_dist

        flat = landmarks.flatten()  # (63,)

        if label == "Left":
            features[0:63] = flat
        elif label == "Right":
            features[63:126] = flat

    return features


# ============================================================
# FEATURE EXTRACTION — DYNAMIC
#
# FIX: Format HARUS identik dengan collect_dynamic.py:
#   for lm in hand_landmarks.landmark:
#       landmarks.extend([lm.x, lm.y, lm.z])
#
# Yaitu: x0,y0,z0, x1,y1,z1, ..., x20,y20,z20  (per-landmark interleaved)
# Tangan pertama yang dideteksi -> slot [0:63]
# Tangan kedua                  -> slot [63:126]
# Tidak pakai Left/Right — konsisten dengan training
# ============================================================
def extract_features_dynamic(hand_landmarks_list) -> np.ndarray:
    keypoints = np.zeros(126, dtype=np.float32)

    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        if hand_idx >= 2:
            break

        # Format identik dengan collect_dynamic.py:
        # extend([lm.x, lm.y, lm.z]) per landmark
        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks, dtype=np.float32)  # (63,)

        if hand_idx == 0:
            keypoints[0:63] = landmarks
        else:
            keypoints[63:126] = landmarks

    return keypoints


# ============================================================
# HELPER: decode gambar
# ============================================================
def decode_frame(contents: bytes):
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return frame, mp_image