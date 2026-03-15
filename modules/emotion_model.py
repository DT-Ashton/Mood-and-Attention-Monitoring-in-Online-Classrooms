import torch
import torchvision.models as models
import cv2
import numpy as np

class EmotionRecognizer:
    """
    Wrapper class for emotion recognition using a fine-tuned EfficientNetV2 model.

    Parameters
    model_path : str
        Path to the trained .pt model file ("resnet18", "mobilenet_v3_large", "efficientnet_v2_s").

    device : str
        Device used for inference ("cpu" or "cuda").
    """
    def __init__(self, architecture: str, model_path: str, device="cpu"):
        self.device = device
        self.labels = ['anger', 'fear', 'happy', 'neutral', 'sad']
        self.input_size = 224

        if architecture == "resnet18":
            self.model = models.resnet18(weights=None)
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(in_features, len(self.labels))
        elif architecture == "mobilenet_v3_large":
            self.model = models.mobilenet_v3_large(weights=None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(in_features, len(self.labels))
        else:
            self.model = models.efficientnet_v2_s(weights=None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(in_features, len(self.labels))

        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()

    def crop_face(self, frame: np.ndarray, landmarks: list, margin=20) -> np.ndarray:
        """
        Crop the face region from the frame based on landmarks.

        Parameters
        ----------
        frame : numpy.ndarray
            The input frame.

        landmarks : list
            List of facial landmarks.
        
        margin : int, optional
            Margin to add around the bounding box. Default is 20 pixels.
        
        Returns
        -------
        numpy.ndarray
            Cropped face image.
        """
        x_coords = [p[0] for p in landmarks]
        y_coords = [p[1] for p in landmarks]

        x1 = max(0, int(min(x_coords))-margin)
        y1 = max(0, int(min(y_coords))-margin)
        x2 = min(frame.shape[1], int(max(x_coords))+margin)
        y2 = min(frame.shape[0], int(max(y_coords))+margin)

        face_crop = frame[y1:y2, x1:x2]
        return face_crop
    
    def preprocess(self, face: np.ndarray) -> torch.Tensor:
        """
        Preprocess cropped face image before feeding into the CNN.

        Parameters
        ----------
        face : np.ndarray
            Cropped face image.

        Returns
        -------
        torch.Tensor
            Preprocessed face image as a PyTorch tensor.
        """
        face = cv2.resize(face, (self.input_size, self.input_size))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2,0,1))
        face = torch.tensor(face).unsqueeze(0)
        return face.to(self.device)

    def predict(self, face: np.ndarray) -> tuple:
        """
        Perform emotion prediction.

        Parameters
        ----------
        face : np.ndarray
            Cropped face image.

        Returns
        -------
        emotion : str
            Predicted emotion label.

        confidence : float
            Softmax probability of the predicted emotion.
        """
        if face is None or face.size == 0:
            return "Unknown", 0.0

        x = self.preprocess(face)

        with torch.no_grad():
            logits = self.model(x)

        prob = torch.nn.functional.softmax(logits, dim=1)
        confidence, pred = torch.max(prob, 1)
        emotion = self.labels[pred.item()]
        return emotion, confidence.item()