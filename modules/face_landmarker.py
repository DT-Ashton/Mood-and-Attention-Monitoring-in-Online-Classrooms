import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

def load_face_landmarker(model_path: str):
    """
    Load MediaPipe Face Landmarker model.

    Parameters
    ----------
    model_path : str
        Path to the .task model file.

    Returns
    -------
    vision.FaceLandmarker
        Initialized Face Landmarker instance.
    """

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )

    landmarker = vision.FaceLandmarker.create_from_options(options)

    return landmarker