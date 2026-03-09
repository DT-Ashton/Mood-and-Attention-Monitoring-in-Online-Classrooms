def attention_decision(yaw: float, pitch: float, ear: float, blink_rate: float,
                       yaw_threshold=20,
                       pitch_threshold=15,
                       ear_threshold=0.2):
    """
    Rule-based decision model to determine attention state.

    Parameters
    ----------
    yaw : float
        Head yaw angle.
    pitch : float
        Head pitch angle.
    ear : float
        Eye Aspect Ratio.
    blink_rate : float
        Blink frequency.

    Returns
    -------
    str
        Attention state (Focused, Distracted, or Drowsy).
    """

    if ear < ear_threshold:
        return "Drowsy"

    if abs(yaw) > yaw_threshold or abs(pitch) > pitch_threshold:
        return "Distracted"

    return "Focused"