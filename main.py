import cv2
from pipeline import MonitoringPipeline
from config.config_loader import load_config

cfg = load_config()

def main():
    pipeline = MonitoringPipeline(config=cfg)
    cap = cv2.VideoCapture(cfg["pipeline"]["webcam_id"])

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = pipeline.process_frame(frame)

        cv2.imshow("Student Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()