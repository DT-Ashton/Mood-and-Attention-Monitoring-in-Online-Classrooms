import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

class FaceLandmarkerLiveStreamWrapper:
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

        Parameters
        ----------
        result : mediapipe.tasks.python.vision.FaceLandmarkerResult
            Detection result containing face landmarks and transformations.

        output_image : mediapipe.Image
            The image with landmarks drawn (if enabled).

        timestamp_ms : int
            Timestamp of the processed frame in milliseconds.
        """
        self.latest_result = result

    def get_latest_result(self):
        return self.latest_result
    
    def detect_async(self, frame, timestamp):
        """
        Asynchronously detect face landmarks in a single image frame.
        Parameters
        ----------
        frame : numpy.ndarray
            Input image frame in RGB format.
            
        timestamp : int
            Timestamp of the frame in milliseconds.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, timestamp)


class FaceLandmarkerImageWrapper:
    """
    Wrapper for MediaPipe FaceLandmarker in IMAGE mode.

    Parameters
    ----------
    model_path : str
        Path to the .task model file.
    """
    def __init__(self, model_path: str):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        Synchronously detect face landmarks in a single image frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Input image frame in RGB format.

        Returns
        -------
        mediapipe.tasks.python.vision.FaceLandmarkerResult
            Detection result containing face landmarks and transformations.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.landmarker.detect(mp_image)
        return result