import cv2
from pipeline import MonitoringPipeline

def main():
    pipeline = MonitoringPipeline()
    cap = cv2.VideoCapture(0)

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