# Mood and Attention Monitoring in Online Classrooms

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
