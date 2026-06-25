import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

from api.config import HAND_LANDMARKER

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_LANDMARKER
    ),
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE
)

detector = HandLandmarker.create_from_options(options)