from flask import Flask, render_template, Response, jsonify, request
import cv2
from pipeline import MonitoringPipeline
from config.config_loader import load_config

app = Flask(__name__)

cfg = load_config()
pipeline = MonitoringPipeline(config=cfg)

cap = cv2.VideoCapture(cfg["pipeline"]["webcam_id"])
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

def generate_frames():
    while True:
        if not cap.isOpened():
            continue

        success, frame = cap.read()
        if not success:
            break

        frame = pipeline.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/metrics")
def metrics():
    stats = pipeline.get_session_stats()

    return jsonify({
        "state": pipeline.state,
        "emotion": pipeline.emotion,
        "stats": stats
    })

@app.route('/toggle', methods=['POST'])
def toggle():
    option = request.json["option"]
    value = request.json["value"]

    if option == "landmarks":
        pipeline.show_landmarks = value

    if option == "metrics":
        pipeline.show_metrics = value

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)