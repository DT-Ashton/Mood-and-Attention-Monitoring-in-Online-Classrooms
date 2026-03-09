import cv2
import mediapipe as mp
import time

from modules.face_landmarker import load_face_landmarker
from modules.head_pose import get_head_pose
from modules.eye_features import BlinkRateCalculator, average_ear
from modules.decision_model import attention_decision
from utils.temporal_smoothing import TemporalSmoother
from utils.visualization import draw_landmarks, draw_eye_mesh

# CONFIG
MODEL_PATH = "models/face_landmarker.task"

# EAR threshold for eye closure detection
EAR_THRESHOLD = 0.21

# Number of consecutive frames to detect blink
BLINK_CONSEC_FRAMES = 3

# Head pose thresholds (degrees)
YAW_THRESHOLD = 20
PITCH_THRESHOLD = 15

# Temporal smoothing window
SMOOTHING_WINDOW = 5

# Visualization flags
SHOW_LANDMARKS = True
SHOW_EYE_MESH = False
SHOW_DETAILS_OVERLAY = True

# Logging settings
LOGGING_ENABLED = False
LOG_PATH = "logs/"
LOG_FILE = time.strftime("%d-%m-%Y %H.%M.%S") +".csv"

# MAIN PIPELINE
def run_pipeline():
    landmarker = load_face_landmarker(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    yaw_smoother = TemporalSmoother(window_size=SMOOTHING_WINDOW)
    pitch_smoother = TemporalSmoother(window_size=SMOOTHING_WINDOW)
    ear_smoother = TemporalSmoother(window_size=SMOOTHING_WINDOW)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(time.time()*1000)

        blink_rate_calc = BlinkRateCalculator(EAR_THRESHOLD, BLINK_CONSEC_FRAMES)


        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            h, w, _ = frame.shape

            # Convert landmarks to numpy coordinates
            coords = [(lm.x * w, lm.y * h) for lm in landmarks]

            if SHOW_LANDMARKS:
                draw_landmarks(frame, coords)

            if SHOW_EYE_MESH:
                draw_eye_mesh(frame, coords)

            # Compute EAR
            ear = average_ear(coords)
            ear = ear_smoother.smooth(ear)

            # Blink detection
            blink_rate = blink_rate_calc.update(ear)

            # Head pose
            if result.facial_transformation_matrixes:
                matrix = result.facial_transformation_matrixes[0]
                pitch, yaw, roll = get_head_pose(matrix)
                pitch = pitch_smoother.smooth(pitch)
                yaw = yaw_smoother.smooth(yaw)
            else:
                yaw, pitch = 0, 0

            # Attention decision
            state = attention_decision(yaw, pitch, ear, blink_rate)

            if SHOW_DETAILS_OVERLAY:
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}", (30, 170),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
                
            cv2.putText(frame, state, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
                
            if LOGGING_ENABLED:
                with open(LOG_PATH + LOG_FILE, "a") as f:
                    f.write(f"{timestamp},{yaw:.2f},{pitch:.2f},{ear:.2f},{blink_rate:.1f},{state}\n")

        cv2.imshow("Student Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()