from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import os
import base64
import numpy as np
from pipeline import LiveStreamMonitoringPipeline, ImageMonitoringPipeline
from config.config_loader import load_config

app = Flask(__name__)

cfg = load_config()
pipeline = None
cap = None

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_webcam_pipeline():
    global pipeline, cap
    if pipeline is None or not isinstance(pipeline, LiveStreamMonitoringPipeline):
        pipeline = LiveStreamMonitoringPipeline(config=cfg)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(cfg['pipeline']['webcam_id'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def init_image_pipeline():
    global pipeline
    if pipeline is None or not isinstance(pipeline, ImageMonitoringPipeline):
        pipeline = ImageMonitoringPipeline(config=cfg)
    return pipeline


def generate_frames():
    global cap, pipeline
    if cap is None or not cap.isOpened():
        return
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = pipeline.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', mode='home')


@app.route('/camera')
def camera():
    init_webcam_pipeline()
    return render_template('index.html', mode='camera')


@app.route('/video_feed')
def video_feed():
    if cap is None or not cap.isOpened():
        return "Webcam is not available", 503

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap, pipeline
    if cap is not None:
        cap.release()
        cap = None
    if isinstance(pipeline, LiveStreamMonitoringPipeline):
        pipeline = None
    return jsonify({'status': 'camera_stopped'})


@app.route('/metrics')
def metrics():
    global pipeline
    if pipeline is None or not isinstance(pipeline, LiveStreamMonitoringPipeline):
        return jsonify({'error': 'Metrics only available for webcam'}), 400

    stats = pipeline.get_session_stats()
    return jsonify({
        'state': pipeline.state,
        'emotion': pipeline.emotion,
        'stats': stats
    })


@app.route('/toggle', methods=['POST'])
def toggle():
    global pipeline
    if pipeline is None or not isinstance(pipeline, LiveStreamMonitoringPipeline):
        return jsonify({'error': 'Toggle only available for webcam'}), 400

    option = request.json.get('option')
    value = request.json.get('value')

    if option == 'landmarks':
        pipeline.show_landmarks = value
    if option == 'metrics':
        pipeline.show_metrics = value

    return jsonify({'status': 'ok'})


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('index.html', mode='upload', result=None)

    # POST
    if 'file' not in request.files:
        return render_template('index.html', mode='upload', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', mode='upload', error='No selected file')

    if not allowed_file(file.filename):
        return render_template('index.html', mode='upload', error='File type not allowed')

    image_data = file.read()
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    with open(filepath, 'wb') as f:
        f.write(image_data)

    # Prepare image pipeline and process
    init_image_pipeline()

    frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return render_template('index.html', mode='upload', error='Failed to read image')

    processed_frame = pipeline.process_image(frame)

    ret, buffer = cv2.imencode('.jpg', processed_frame)
    if not ret:
        return render_template('index.html', mode='upload', error='Failed to encode image')

    result_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    return render_template(
        'index.html',
        mode='upload',
        result=result_base64,
        state=pipeline.state,
        pitch=round(pipeline.pitch, 2),
        yaw=round(pipeline.yaw, 2),
        ear=round(pipeline.ear, 2),
        emotion=pipeline.emotion,
        conf = round(pipeline.conf, 2)
    )


if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)