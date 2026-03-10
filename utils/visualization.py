import cv2
import numpy as np

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
        cv2.circle(frame, tuple(map(int, lm)), 1, (0,255,0), -1)

def draw_metrics(frame, ear=None, blink_rate=None, yaw=None, pitch=None, attention=None, emotion=None, confidence=None):
    """
    Draw monitoring metrics on the frame.

    Parameters
    ----------
    frame : np.ndarray
        Input video frame.

    ear : float, optional
        Eye Aspect Ratio value.

    blink_rate : float, optional
        Blink rate (blinks per minute).

    yaw : float, optional
        Head yaw angle.

    pitch : float, optional
        Head pitch angle.

    attention : str, optional
        Attention state (Focused / Distracted / Drowsy).

    emotion : str, optional
        Predicted emotion label.

    confidence : float, optional
        Emotion prediction confidence.
    """

    y_offset = 30
    line_gap = 30

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (255, 0, 0)
    thickness = 2

    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, y_offset), font, scale, color, thickness)
        y_offset += line_gap

    if blink_rate is not None:
        cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}", (20, y_offset), font, scale, color, thickness)
        y_offset += line_gap

    if yaw is not None:
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, y_offset), font, scale, color, thickness)
        y_offset += line_gap

    if pitch is not None:
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, y_offset), font, scale, color, thickness)
        y_offset += line_gap

    if attention is not None:
        cv2.putText(frame, f"Attention: {attention}", (20, y_offset), font, scale, color, thickness)
        y_offset += line_gap

    if emotion is not None:
        if confidence is not None:
            text = f"Emotion: {emotion} ({confidence:.2f})"
        else:
            text = f"Emotion: {emotion}"

        cv2.putText(frame, text, (20, y_offset), font, scale, color, thickness)