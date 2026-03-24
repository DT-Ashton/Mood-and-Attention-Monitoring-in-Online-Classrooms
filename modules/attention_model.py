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

    temporal_mode : bool
        If True, use temporal buffers for smoothing. If False, process single frame.
    """

    def __init__(self, window_size=30, ear_threshold=0.23, yaw_threshold=15, pitch_threshold=15, blink_rate_drowsy=20, temporal_mode=True):
        self.ear_threshold = ear_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.blink_rate_drowsy = blink_rate_drowsy
        self.temporal_mode = temporal_mode

        if temporal_mode:
            # Temporal buffers
            self.window_size = window_size
            self.ear_buffer = deque(maxlen=window_size)
            self.blink_buffer = deque(maxlen=window_size)
            self.yaw_buffer = deque(maxlen=window_size)
            self.pitch_buffer = deque(maxlen=window_size)
            self.emotion_buffer = deque(maxlen=window_size)
        else:
            # For single frame, store current values
            self.current_ear = 0.0
            self.current_blink_rate = 0.0
            self.current_yaw = 0.0
            self.current_pitch = 0.0
            self.current_emotion = "neutral"

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
        if self.temporal_mode:
            self.ear_buffer.append(ear)
            self.blink_buffer.append(blink_rate)
            self.yaw_buffer.append(yaw)
            self.pitch_buffer.append(pitch)
            self.emotion_buffer.append(emotion)

            if len(self.ear_buffer) < self.window_size:
                return "Analyzing"

            return self._decision()
        else:
            # Single frame mode
            self.current_ear = ear
            self.current_blink_rate = blink_rate
            self.current_yaw = yaw
            self.current_pitch = pitch
            self.current_emotion = emotion
            return self._decision_single()

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

        return self._classify_attention(ear_mean, blink_mean, yaw_mean, pitch_mean, dominant_emotion, use_blink_rate=True)

    def _decision_single(self) -> str:
        """
        Decision logic for single frame.

        Returns
        -------
        str
            Estimated attention state.
        """
        return self._classify_attention(
            self.current_ear,
            self.current_blink_rate,
            self.current_yaw,
            self.current_pitch,
            self.current_emotion,
            use_blink_rate=False
        )

    def _classify_attention(self, ear, blink_rate, yaw, pitch, emotion, use_blink_rate=True) -> str:
        """
        Core attention classification logic.

        Parameters
        ----------
        ear : float
        blink_rate : float
        yaw : float
        pitch : float
        emotion : str
        use_blink_rate : bool
            Whether to use blink_rate in classification (True for live stream, False for image).

        Returns
        -------
        str
            Attention state.
        """
        if emotion in ["sad", "neutral"] and ear < self.ear_threshold and use_blink_rate and blink_rate < self.blink_rate_drowsy:
            return "Drowsy"
        
        if emotion in ["sad", "neutral"] and ear < self.ear_threshold and pitch > self.pitch_threshold:
            return "Drowsy"

        if abs(yaw) > self.yaw_threshold:
            return "Distracted"
        
        if abs(pitch) > self.pitch_threshold:
            return "Distracted"

        if emotion in ["angry", "fear"] and abs(yaw) < self.yaw_threshold and ear > self.ear_threshold:
            return "Disengaged"

        if emotion in ["happy", "neutral"]:
            return "Focused"

        return "Focused"