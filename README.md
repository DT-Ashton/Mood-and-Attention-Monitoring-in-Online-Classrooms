# Mood and Attention Monitoring in Online Classrooms
This project builds a simple real-time system that helps monitor a student's **emotion** and **attention level** using a webcam.

The system integrates **facial landmark analysis, head pose estimation, eye-based features, and deep learning-based emotion recognition** into a pipeline. A rule-based decision model is then applied to infer the overall attention level of the student.

## 🚀 Key Features

### 🎥 Real-time Monitoring
- Works directly with webcam input  
- Processes video frame by frame (also support image)
- Shows results instantly  

### 😊 Emotion Recognition
- Uses a trained CNN model (EfficientNetV2, ResNet18, MobileNetV3)
  - Happy  
  - Neutral  
  - Sad  
  - Angry  
  - Fear  
- Also returns a confidence score  

### 👁️ Eye-based Features
- Calculates **Eye Aspect Ratio (EAR)**  
- Detects when the eyes are closed  
- Tracks **blink rate over time**  

### 🧭 Head Pose Estimation
- Estimates head movement using MediaPipe  
- Measures:
  - Yaw (left/right)  
  - Pitch (up/down)  
  - Roll (tilt)  
- Helps determine if the student is looking at the screen  

### ⏱️ Smoothing (Stable Results)
- Uses EMA (Exponential Moving Average)  
- Reduces noise from frame-to-frame changes  
- Makes the system more stable  

### 🧠 Attention Detection
- Combines:
  - Head pose  
  - Eye features  
  - Blink rate  
  - Emotion  
- Classifies attention into:
  - Focused  
  - Distracted  
  - Drowsy  
  - Disengaged  

### 📊 Session Statistics
- Tracks attention over time  
- Shows:
  - % Focused  
  - % Distracted  
  - % Drowsy  

### 🌐 Web Dashboard
- Built with Flask  
- Shows live webcam stream (also support image uploading)
- Displays:
  - Emotion  
  - Attention state (with colors)  
  - Session statistics  
- Allows turning on/off:
  - Landmarks  
  - Metrics

---

## Dataset
OFFICAL DATASET : https://www.kaggle.com/datasets/lorddemon/face-emotion-balance-dataset-american-asian
=> Unlimited accesses can be used for any purpose.

## 📂 Project Structure:
```
Mood-and-Attention-Monitoring-in-Online-Classrooms/
│
├── app.py                      # Web demo
├── main.py                     # CLI demo
├── pipeline.py                 # System pipeline
│
├── config/                     # System config
│
├── models/                     # Fine-tuned CNN models and Mediapipe Face Mesh
│
├── modules/
│   ├── attention_model.py
│   ├── emotion_model.py
│   ├── eye_features.py
│   ├── face_landmarker.py
│   └── head_pose.py
│
├── notebooks/
│   ├── training/               # Training scripts
│   └── benchmark.ipynb         # Fine-tuned models benchmark
│
├── templates/
│   └── index.html              # Demo web interface
│
├── utils/
│   ├── temporal_smoothing.py
│   └── visualization.py
│   
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ Running the Project
1. Clone the repository

```bash
git clone https://github.com/DT-Ashton/Mood-and-Attention-Monitoring-in-Online-Classrooms.git
cd https://github.com/DT-Ashton/Mood-and-Attention-Monitoring-in-Online-Classrooms.git
```
2. Create and activate a virtual environment

```bash
python -m venv venv
```

&emsp;&emsp;Windows:  
&emsp;&emsp;&emsp; ```.venv\Scripts\Activate.ps1``` (powershell)  
&emsp;&emsp;&emsp; ```.venv\Scripts\activate``` (command prompt)  

&emsp;&emsp;macOS/Linux:  ```source .venv/bin/activate```  

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run project:

- Run CLI demo: 
```bash
python main.py
```

- Run web demo: 
```bash
python app.py
```

**Note:** Change project settings in `config/config.yaml`
