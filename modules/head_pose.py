import numpy as np
import math

def get_head_pose(matrix: np.ndarray) -> tuple:
    """
    Convert the facial transformation matrix
    provided by MediaPipe into Euler angles representing
    head orientation (pitch, yaw, roll).

    Parameters
    ----------
    matrix : np.ndarray
        4x4 facial transformation matrix from MediaPipe.

    Returns
    -------
    tuple
        Pitch, Yaw, Roll angles in degrees.
    """

    R = matrix[:3,:3]
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2,1], R[2,2])
        yaw = math.atan2(-R[2,0], sy)
        roll = math.atan2(R[1,0], R[0,0])
    else:
        pitch = math.atan2(-R[1,2], R[1,1])
        yaw = math.atan2(-R[2,0], sy)
        roll = 0

    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    roll = math.degrees(roll)

    return yaw, pitch, roll