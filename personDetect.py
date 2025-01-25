from ultralytics import YOLO
import cv2
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_person.py <path_to_video>")
        exit(1)

    video_path = sys.argv[1]
    model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 Nano model

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit(1)

    frame_skip = 10  # Process every nth frame for efficiency
    frame_count = 0
    person_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Use YOLO to predict objects in the current frame
        results = model.predict(frame, conf=0.5, show=False)
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                class_name = result.names[int(cls)]
                print(f"Frame {frame_count}: Detected {class_name} with confidence {conf:.2f}")

                if class_name == "person":  # Check if a person is detected
                    person_detected = True

        if person_detected:
            break

    cap.release()

    if person_detected:
        print("The video contains a person.")
        exit(0)
    else:
        print("No person detected in the video.")
        exit(1)

if __name__ == "__main__":
    main()

