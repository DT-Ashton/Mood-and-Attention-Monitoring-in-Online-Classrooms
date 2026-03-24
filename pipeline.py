import time
import cv2

from modules.face_landmarker import FaceLandmarkerLiveStreamWrapper, FaceLandmarkerImageWrapper
from modules.head_pose import get_head_pose
from modules.eye_features import BlinkRateCalculator, compute_avg_ear
from modules.emotion_model import EmotionRecognizer
from modules.attention_model import AttentionModel
from utils.temporal_smoothing import EMAFilter
from utils.visualization import draw_landmarks, draw_metrics
from config.config_loader import load_config

class LiveStreamMonitoringPipeline:
    """
    Processing pipeline for real-time monitoring.

    This class processes video frames and returns annotated frames
    with emotion and attention predictions.
    """
    def __init__(self, config):
        self.landmarker = FaceLandmarkerLiveStreamWrapper(config['models']['landmarker'])
        self.emotion_model = EmotionRecognizer(
            config["models"]["emotion"]["architecture"],
            config["models"]["emotion"]["path"],
            config["models"]["device"]
        )
        self.attention_model = AttentionModel(
            window_size=config['attention']['window_size'],
            ear_threshold=config['attention']['ear_threshold'],
            yaw_threshold=config['attention']['yaw_threshold'],
            pitch_threshold=config['attention']['pitch_threshold'],
            blink_rate_drowsy=config['attention']['blink_rate_drowsy']
        )
        self.blink_rate_calc = BlinkRateCalculator(
            ear_threshold=config['attention']['ear_threshold'],
            consec_frames=config['attention']['blink_consec_frames']
        )
        self.show_metrics = config['visualization']['show_metrics']
        self.show_landmarks = config['visualization']['show_landmarks']
        self.logging_enabled = config['logging']['enabled']
        self.log_file = config['logging']['path'] + time.strftime("%d-%m-%Y %H.%M.%S") + ".csv"

        self.yaw_smoother = EMAFilter(config['smoothing']['ema_alpha_pose'])
        self.pitch_smoother = EMAFilter(config['smoothing']['ema_alpha_pose'])
        self.ear_smoother = EMAFilter(config['smoothing']['ema_alpha_ear'])
        self.emotion_filter = EMAFilter(config['smoothing']['ema_alpha_emotion'])

        # Internal states
        self.skipped_frames = config['pipeline']['emotion_inference_interval']
        self.frame_id = 0
        self.emotion = "Unknown"
        self.state = "Unknown"
        self.conf = 0.0

        # Session statistics
        self.config = config
        self.focused_frames = 0
        self.disengaged_frames = 0
        self.distracted_frames = 0
        self.drowsy_frames = 0
        self.total_frames = 0

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

        self.frame_id += 1
        
        # Monotonically increasing timestamp (33ms per frame ≈ 30fps)
        timestamp = self.frame_id * 33

        # Run Face Landmarker
        self.landmarker.detect_async(frame, timestamp)
        result = self.landmarker.get_latest_result()

        if result and result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # Convert landmarks to pixel coordinates
            h, w, _ = frame.shape
            coords = [(lm.x * w, lm.y * h) for lm in landmarks]

            # Compute eye aspect ratio and blink rate
            ear = compute_avg_ear(coords)
            ear = self.ear_smoother.update(ear)
            blink_rate = self.blink_rate_calc.update(ear)

            # Head pose estimation
            if result.facial_transformation_matrixes:
                matrix = result.facial_transformation_matrixes[0]
                pitch, yaw, roll = get_head_pose(matrix)
                pitch = self.pitch_smoother.update(pitch)
                yaw = self.yaw_smoother.update(yaw)
            else:
                yaw, pitch = 0, 0

            # Emotion inference every N frames
            if self.frame_id % self.skipped_frames == 0:
                cropped_face = self.emotion_model.crop_face(frame, coords)
                if cropped_face is not None and cropped_face.size > 0:
                    self.emotion, self.conf = self.emotion_model.predict(cropped_face)
                    self.conf = self.emotion_filter.update(self.conf)

            # Attention state update
            self.state = self.attention_model.update(ear=ear, blink_rate=blink_rate, yaw=yaw, pitch=pitch, emotion=self.emotion)

            self.total_frames += 1

            # Session statistics
            if self.state == "Focused":
                self.focused_frames += 1
            elif self.state == "Disengaged":
                self.disengaged_frames += 1
            elif self.state == "Distracted":
                self.distracted_frames += 1
            elif self.state == "Drowsy":
                self.drowsy_frames += 1

            # Draw visualizations
            if self.show_metrics:
                draw_metrics(frame, self.state, ear=ear, blink_rate=blink_rate, yaw=yaw, pitch=pitch, emotion=self.emotion, confidence=self.conf)
            else:
                draw_metrics(frame, self.state)

            if self.show_landmarks:
                draw_landmarks(frame, coords)

            # Logging
            if self.logging_enabled:
                with open(self.log_file, "a") as f:
                    f.write(f"{timestamp},{self.state},{yaw:.2f},{pitch:.2f},{ear:.2f},{blink_rate:.1f},{self.emotion},{self.conf:.2f}\n")
        
        return frame

    def get_session_stats(self):
        if self.total_frames == 0:
            return {
                "focused": 0,
                "disengaged": 0,
                "distracted": 0,
                "drowsy": 0
            }

        return {
            "focused": round(self.focused_frames / self.total_frames * 100, 1),
            "disengaged": round(self.disengaged_frames / self.total_frames * 100, 1),
            "distracted": round(self.distracted_frames / self.total_frames * 100, 1),
            "drowsy": round(self.drowsy_frames / self.total_frames * 100, 1)
        }
    

class ImageMonitoringPipeline:
    """
    Processing pipeline for single image inference.

    This class processes a single image and returns an annotated image
    """
    def __init__(self, config):
        self.landmarker = FaceLandmarkerImageWrapper(config['models']['landmarker'])
        self.emotion_model = EmotionRecognizer(
            config["models"]["emotion"]["architecture"],
            config["models"]["emotion"]["path"],
            config["models"]["device"]
        )
        self.attention_model = AttentionModel(
            window_size=0,
            ear_threshold=config['attention']['ear_threshold'],
            yaw_threshold=config['attention']['yaw_threshold'],
            pitch_threshold=config['attention']['pitch_threshold'],
            blink_rate_drowsy=1
        )
        self.show_metrics = config['visualization']['show_metrics']
        self.show_landmarks = config['visualization']['show_landmarks']
        self.emotion = "Unknown"
        self.state = "Unknown"
        self.conf = 0.0
    
    def process_image(self, image):
        """
        Process a single image.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        
        Returns
        -------
        np.ndarray
            Annotated output image.
        """
        # Run Face Landmarker
        self.landmarker.detect(image)
        result = self.landmarker.get_latest_result()

        if result and result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # Convert landmarks to pixel coordinates
            h, w, _ = image.shape
            coords = [(lm.x * w, lm.y * h) for lm in landmarks]

            # Compute eye aspect ratio
            ear = compute_avg_ear(coords)

            # Head pose estimation
            if result.facial_transformation_matrixes:
                matrix = result.facial_transformation_matrixes[0]
                pitch, yaw, roll = get_head_pose(matrix)
                pitch = self.pitch_smoother.update(pitch)
                yaw = self.yaw_smoother.update(yaw)
            else:
                yaw, pitch = 0, 0

            # Emotion inference
            cropped_face = self.emotion_model.crop_face(image, coords)
            if cropped_face is not None and cropped_face.size > 0:
                self.emotion, self.conf = self.emotion_model.predict(cropped_face)

            # Attention state update
            self.state = self.attention_model.update(ear=ear, blink_rate=0, yaw=yaw, pitch=pitch, emotion=self.emotion)

            # Draw visualizations
            if self.show_metrics:
                draw_metrics(image, self.state, ear=ear, yaw=yaw, pitch=pitch, emotion=self.emotion, confidence=self.conf)
            else:
                draw_metrics(image, self.state)

            if self.show_landmarks:
                draw_landmarks(image, coords)

        return image