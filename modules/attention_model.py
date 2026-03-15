from collections import deque
import numpy as np

class AttentionModel:
    """
    Rule-based attention classifier using temporal feature fusion.

    Parameters
    ----------
    window_size : int
        Number of frames used for temporal smoothing.

    ear_thresh : float
        EAR threshold for eye closure detection.

    yaw_thresh : float
        Yaw threshold for distraction detection.

    pitch_thresh : float
        Pitch threshold.

    blink_rate_drowsy : float
        Blink rate threshold for drowsiness detection (blinks per minute).
    """

    def __init__(self, window_size=30, ear_threshold=0.23, yaw_threshold=15, pitch_threshold=15, blink_rate_drowsy=20):
        self.window_size = window_size
        self.ear_threshold = ear_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.blink_rate_drowsy = blink_rate_drowsy


        # Temporal buffers
        self.ear_buffer = deque(maxlen=window_size)
        self.blink_buffer = deque(maxlen=window_size)
        self.yaw_buffer = deque(maxlen=window_size)
        self.pitch_buffer = deque(maxlen=window_size)
        self.emotion_buffer = deque(maxlen=window_size)

    def update(self, ear: float, blink_rate: float, yaw: float, pitch: float, emotion: str) -> str:
        """
        Update model with new frame features.

        Parameters
        ----------
        ear : float
            Eye Aspect Ratio.

        blink_rate : float
            Blink rate (blinks per minute).

        yaw : float
            Head yaw angle.

        pitch : float
            Head pitch angle.

        emotion : str
            Detected emotion label.

        Returns
        -------
        str
            Estimated attention state.
        """
        self.ear_buffer.append(ear)
        self.blink_buffer.append(blink_rate)
        self.yaw_buffer.append(yaw)
        self.pitch_buffer.append(pitch)
        self.emotion_buffer.append(emotion)

        if len(self.ear_buffer) < self.window_size:
            return "Analyzing"

        return self._decision()

    def _decision(self) -> str:
        """
        Decision logic based on temporal statistics.

        Returns
        -------
        str
            Estimated attention state: "Focused", "Distracted", "Drowsy", "Confused", "Disengaged".
        """
        ear_mean = np.mean(self.ear_buffer)
        blink_mean = np.mean(self.blink_buffer)
        yaw_mean = np.mean(self.yaw_buffer)
        pitch_mean = np.mean(self.pitch_buffer)

        # dominant emotion in window
        emotion_counts = {}

        for e in self.emotion_buffer:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        if ear_mean < self.ear_threshold and blink_mean < self.blink_rate_drowsy:
            return "Drowsy"
        
        if ear_mean < self.ear_threshold and pitch_mean > self.pitch_threshold:
            return "Drowsy"

        if abs(yaw_mean) > self.yaw_threshold:
            return "Distracted"
        
        if abs(pitch_mean) > self.pitch_threshold:
            return "Distracted"

        if (dominant_emotion == "fear" and abs(yaw_mean) < self.yaw_threshold and ear_mean > self.ear_threshold):
            return "Confused"

        if (dominant_emotion == "sad" and abs(yaw_mean) < self.yaw_threshold and ear_mean > self.ear_threshold):
            return "Disengaged"

        if dominant_emotion in ["happy", "neutral"]:
            return "Focused"

        return "Focused"