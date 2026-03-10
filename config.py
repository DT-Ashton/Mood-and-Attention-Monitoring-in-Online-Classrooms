import time

LANDMARKER_MODEL_PATH = "models/face_landmarker.task"
EMOTION_MODEL_PATH = "models/EfficentNetV2_emotion_model.pth"

# EAR threshold for eye closure detection
EAR_THRESHOLD = 0.21

# Number of consecutive frames to detect blink
BLINK_CONSEC_FRAMES = 3

# Head pose thresholds (degrees)
YAW_THRESHOLD = 20
PITCH_THRESHOLD = 15

# Visualization flags
SHOW_LANDMARKS = True
SHOW_DETAILS_METRICS = True

# Logging settings
LOGGING_ENABLED = False
LOG_PATH = "logs/"
LOG_FILE = time.strftime("%d-%m-%Y %H.%M.%S") +".csv"