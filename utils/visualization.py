import cv2
import numpy as np

def draw_landmarks(frame: np.ndarray, landmarks: list|np.ndarray, color=(0,255,0)):
    """
    Fast drawing of facial landmarks using NumPy vectorization.

    Parameters
    ----------
    frame : np.ndarray
        Video frame.

    landmarks : list or np.ndarray
        List of (x,y) landmark coordinates.

    color : tuple
        Landmark color (B, G, R).
    """
    pts = np.array(landmarks, dtype=np.int32)

    h, w, _ = frame.shape

    x = np.clip(pts[:,0], 0, w-1)
    y = np.clip(pts[:,1], 0, h-1)

    frame[y, x] = color

def draw_metrics(frame: np.ndarray, state: str, ear: float|None=None, blink_rate: float|None=None, yaw: float|None=None, pitch: float|None=None, emotion: str|None=None, confidence: float|None=None):
    """
    Draw monitoring metrics on the frame.

    Parameters
    ----------
    frame : np.ndarray
        Input video frame.

    state : str, optional
        Attention state ("Focused", "Distracted", "Drowsy", "Confused", "Disengaged").

    ear : float, optional
        Eye Aspect Ratio value.

    blink_rate : float, optional
        Blink rate (blinks per minute).

    yaw : float, optional
        Head yaw angle.

    pitch : float, optional
        Head pitch angle.

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
    thickness = 3

    cv2.putText(frame, f"Attention: {state}", (20, y_offset), font, scale, (0, 255, 0), thickness)
    y_offset += line_gap

    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, y_offset), font, scale, color, thickness)
        y_offset += line_gap

    if blink_rate is not None:
        cv2.putText(frame, f"Blink Rate: {blink_rate:.1f}/min", (20, y_offset), font, scale, color, thickness)
        y_offset += line_gap

    if yaw is not None:
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, y_offset), font, scale, color, thickness)
        y_offset += line_gap

    if pitch is not None:
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, y_offset), font, scale, color, thickness)
        y_offset += line_gap

    if emotion is not None:
        if confidence is not None:
            text = f"Emotion: {emotion} ({confidence:.2f})"
        else:
            text = f"Emotion: {emotion}"

        cv2.putText(frame, text, (20, y_offset), font, scale, color, thickness)