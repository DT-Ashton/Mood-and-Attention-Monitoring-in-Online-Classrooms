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
    """

    def __init__(self, window_size=30, ear_thresh=0.23, yaw_thresh=20, pitch_thresh=20):
        self.window_size = window_size
        self.ear_thresh = ear_thresh
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh

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

        if ear_mean < self.ear_thresh and blink_mean < 8:
            return "Drowsy"
        
        if ear_mean < self.ear_thresh and pitch_mean > self.pitch_thresh:
            return "Drowsy"

        if abs(yaw_mean) > self.yaw_thresh:
            return "Distracted"
        
        if abs(pitch_mean) > self.pitch_thresh:
            return "Distracted"

        if (dominant_emotion == "fear" and abs(yaw_mean) < self.yaw_thresh and ear_mean > self.ear_thresh):
            return "Confused"

        if (dominant_emotion == "sad" and abs(yaw_mean) < self.yaw_thresh and ear_mean > self.ear_thresh):
            return "Disengaged"

        if dominant_emotion in ["happy", "neutral"]:
            return "Focused"

        return "Focused"