import numpy as np
import time

def compute_ear(eye_indices: list) -> float:
    """
    Compute Eye Aspect Ratio (EAR).

    EAR is calculated using distances between vertical
    and horizontal eye landmarks.

    Parameters
    ----------
    eye_indices : list
        List of eye landmark indices.

    Returns
    -------
    float
        Eye Aspect Ratio value.
    """

    vertical1 = np.linalg.norm(eye_indices[1] - eye_indices[5])
    vertical2 = np.linalg.norm(eye_indices[2] - eye_indices[4])
    horizontal = np.linalg.norm(eye_indices[0] - eye_indices[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear

def average_ear(landmarks: list) -> float:
    """
    Compute average EAR for both eyes.

    Parameters
    ----------
    landmarks : list
        List of facial landmarks.

    Returns
    -------
    float
        Average Eye Aspect Ratio value.
    """

    # MediaPipe eye landmark indices (FaceMesh)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]

    left_ear = compute_ear(landmarks, left_eye)
    right_ear = compute_ear(landmarks, right_eye)

    return (left_ear + right_ear) / 2.0


class BlinkRateCalculator:
    """
    Estimates blink rate in terms of blinks per minute.
    """

    def __init__(self, ear_threshold=0.21, consec_frames=3):
        """
        Initialize the blink rate calculator.

        Parameters
        ----------
        ear_threshold : float
            Threshold below which the eye is considered closed.

        consec_frames : int
            Number of consecutive frames required to register a blink.
        """

        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames

        self.frame_counter = 0
        self.total_blinks = 0

        self.start_time = time.time()

    def update(self, ear: float) -> tuple:
        """
        Update blink detector with a new EAR value.

        Parameters
        ----------
        ear : float
            Eye Aspect Ratio for the current frame.

        Returns
        -------
        tuple
            (blink_detected, blink_rate)
        """

        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.total_blinks += 1
            self.frame_counter = 0

        elapsed_time = time.time() - self.start_time

        if elapsed_time > 0:
            blink_rate = (self.total_blinks / elapsed_time) * 60
        else:
            blink_rate = 0

        return blink_rate