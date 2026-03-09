import cv2
import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def draw_landmarks(frame: np.ndarray, landmarks: list) -> None:
    """
    Draw facial landmarks on the frame.

    Parameters
    ----------
    frame : ndarray
        Video frame.
    landmarks : list
        List of facial landmarks.
    """

    for lm in landmarks:
        cv2.circle(frame, tuple(lm), 1, (0,255,0), -1)

def draw_eye_mesh(frame, points):
    for idx in LEFT_EYE:
        cv2.circle(frame, tuple(points[idx]), 2, (255,0,0), -1)

    for idx in RIGHT_EYE:
        cv2.circle(frame, tuple(points[idx]), 2, (255,0,0), -1)