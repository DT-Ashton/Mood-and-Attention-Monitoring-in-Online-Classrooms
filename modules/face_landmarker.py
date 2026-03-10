import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

class FaceLandmarkerWrapper:
    """
    Wrapper for MediaPipe FaceLandmarker in LIVE_STREAM mode.

    Parameters
    ----------
    model_path : str
        Path to the .task model file.
    """
    def __init__(self, model_path: str):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            output_facial_transformation_matrixes=True,
            result_callback=self._callback
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.latest_result = None

    def _callback(self, result, output_image, timestamp_ms):
        """
        Callback function called asynchronously by MediaPipe.
        """
        self.latest_result = result

    def get_latest_result(self):
        return self.latest_result
    
    def detect_async(self, frame, timestamp):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, timestamp)