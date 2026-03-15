import time
import cv2

from modules.face_landmarker import FaceLandmarkerWrapper
from modules.head_pose import get_head_pose
from modules.eye_features import BlinkRateCalculator, compute_avg_ear
from modules.emotion_model import EmotionRecognizer
from modules.attention_model import AttentionModel
from utils.temporal_smoothing import EMAFilter
from utils.visualization import draw_landmarks, draw_metrics
from config.config_loader import load_config

cfg = load_config()

class MonitoringPipeline:
    """
    Main processing pipeline for mood and attention monitoring.

    This class processes video frames and returns annotated frames
    with emotion and attention predictions.
    """
    def __init__(self):
        self.landmarker = FaceLandmarkerWrapper(cfg['models']['landmarker'])
        self.emotion_model = EmotionRecognizer(
            cfg["models"]["emotion"]["architecture"],
            cfg["models"]["emotion"]["path"],
            cfg["models"]["device"]
        )
        self.attention_model = AttentionModel(
            window_size=cfg['attention']['window_size'],
            ear_threshold=cfg['attention']['ear_threshold'],
            yaw_threshold=cfg['attention']['yaw_threshold'],
            pitch_threshold=cfg['attention']['pitch_threshold'],
            blink_rate_drowsy=cfg['attention']['blink_rate_drowsy']
        )
        self.blink_rate_calc = BlinkRateCalculator(
            ear_threshold=cfg['attention']['ear_threshold'],
            consec_frames=cfg['attention']['blink_consec_frames']
        )
        self.show_metrics = cfg['visualization']['show_metrics']
        self.show_landmarks = cfg['visualization']['show_landmarks']
        self.logging_enabled = cfg['logging']['enabled']
        self.log_file = cfg['logging']['path'] + time.strftime("%d-%m-%Y %H.%M.%S") + ".csv"

        self.yaw_smoother = EMAFilter(cfg['smoothing']['ema_alpha_pose'])
        self.pitch_smoother = EMAFilter(cfg['smoothing']['ema_alpha_pose'])
        self.ear_smoother = EMAFilter(cfg['smoothing']['ema_alpha_ear'])
        self.emotion_filter = EMAFilter(cfg['smoothing']['ema_alpha_emotion'])

        self.skipped_frames = cfg['pipeline']['emotion_inference_interval']
        self.frame_id = 0
        self.emotion = "Unknown"
        self.conf = 0.0

    def process_frame(self, frame):
        """
        Process a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame from webcam or video.

        Returns
        -------
        np.ndarray
            Annotated output frame.
        """
        frame = cv2.flip(frame, 1)
        timestamp = int(time.time() * 1000)

        self.frame_id += 1

        # Run Face Landmarker
        self.landmarker.detect_async(frame, timestamp)
        result = self.landmarker.get_latest_result()

        if result and result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # Convert landmarks to pixel coordinates
            h, w, _ = frame.shape
            coords = [(lm.x * w, lm.y * h) for lm in landmarks]

            ear = compute_avg_ear(coords)
            ear = self.ear_smoother.update(ear)

            blink_rate = self.blink_rate_calc.update(ear)

            if result.facial_transformation_matrixes:
                matrix = result.facial_transformation_matrixes[0]
                pitch, yaw, roll = get_head_pose(matrix)
                pitch = self.pitch_smoother.update(pitch)
                yaw = self.yaw_smoother.update(yaw)
            else:
                yaw, pitch = 0, 0

            if self.frame_id % self.skipped_frames == 0:
                cropped_face = self.emotion_model.crop_face(frame, coords)
                if cropped_face is not None and cropped_face.size > 0:
                    self.emotion, self.conf = self.emotion_model.predict(cropped_face)
                    self.conf = self.emotion_filter.update(self.conf)

            state = self.attention_model.update(ear=ear, blink_rate=blink_rate, yaw=yaw, pitch=pitch, emotion=self.emotion)

            if self.show_metrics:
                draw_metrics(frame, state, ear=ear, blink_rate=blink_rate, yaw=yaw, pitch=pitch, emotion=self.emotion, confidence=self.conf)
            else:
                draw_metrics(frame, state)

            if self.show_landmarks:
                draw_landmarks(frame, coords)

            if self.logging_enabled:
                with open(self.log_file, "a") as f:
                    f.write(f"{timestamp},{state},{yaw:.2f},{pitch:.2f},{ear:.2f},{blink_rate:.1f},{self.emotion},{self.conf:.2f}\n")
        return frame