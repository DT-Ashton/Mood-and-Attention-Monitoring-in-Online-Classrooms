import cv2
import mediapipe as mp
import time

from modules.face_landmarker import FaceLandmarkerWrapper
from modules.head_pose import get_head_pose
from modules.eye_features import BlinkRateCalculator, compute_avg_ear
from modules.emotion_model import EmotionRecognizer
from modules.attention_model import AttentionModel
from utils.temporal_smoothing import EMAFilter
from utils.visualization import draw_landmarks, draw_metrics
from config import *

# MAIN PIPELINE
def run_pipeline():
    landmarker = FaceLandmarkerWrapper(LANDMARKER_MODEL_PATH)
    emotion_model = EmotionRecognizer(EMOTION_MODEL_PATH, device="cuda")
    blink_rate_calc = BlinkRateCalculator(EAR_THRESHOLD, BLINK_CONSEC_FRAMES)
    attention_model = AttentionModel()

    cap = cv2.VideoCapture(0)

    yaw_smoother = EMAFilter()
    pitch_smoother = EMAFilter()
    ear_smoother = EMAFilter()
    emotion_filter = EMAFilter()

    frame_id = 0
    emotion = "Unknown"
    conf = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        timestamp = int(time.time()*1000)
        frame_id += 1

        landmarker.detect_async(frame, timestamp)
        result = landmarker.get_latest_result()

        if result and result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # Convert landmarks to numpy coordinates
            h, w, _ = frame.shape
            coords = [(lm.x * w, lm.y * h) for lm in landmarks]

            # Compute EAR
            ear = compute_avg_ear(coords)
            ear = ear_smoother.update(ear)

            # Blink detection
            blink_rate = blink_rate_calc.update(ear)

            # Head pose
            if result.facial_transformation_matrixes:
                matrix = result.facial_transformation_matrixes[0]
                pitch, yaw, roll = get_head_pose(matrix)
                pitch = pitch_smoother.update(pitch)
                yaw = yaw_smoother.update(yaw)
            else:
                yaw, pitch = 0, 0

            # Face alignment every 10 frames for emotion recognition
            if frame_id % 10 == 0:
                cropped_face = emotion_model.crop_face(frame, coords)
                if cropped_face is not None and cropped_face.size > 0:
                    emotion, conf = emotion_model.predict(cropped_face)
                    conf = emotion_filter.update(conf)

            # Attention decision
            state = attention_model.update(ear=ear, blink_rate=blink_rate, yaw=yaw, pitch=pitch, emotion=emotion)

            if SHOW_DETAILS_METRICS:
                draw_metrics(frame, state, ear=ear, blink_rate=blink_rate, yaw=yaw, pitch=pitch, emotion=emotion, confidence=conf)
            else:
                draw_metrics(frame, state)

            if SHOW_LANDMARKS:
                draw_landmarks(frame, coords)
                
            if LOGGING_ENABLED:
                with open(LOG_PATH + LOG_FILE, "a") as f:
                    f.write(f"{timestamp},{state},{yaw:.2f},{pitch:.2f},{ear:.2f},{blink_rate:.1f},{emotion},{conf:.2f}\n")

        cv2.imshow("Student Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()