from ultralytics import YOLO
import cv2
import os
import sys

def extract_persons_from_videos(input_dir, output_dir, classNumber, confidence_threshold=0.5):
    # Load the pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")  # YOLOv8 Nano model (lightweight and efficient)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for video_name in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_name)

        # Skip non-video files
        if not video_name.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {video_name}")
            continue

        print(f"Processing video: {video_name}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_name}")
            continue

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_name = f"{os.path.splitext(video_name)[0]}_frame{frame_count}.jpg"
            image_path = os.path.join(images_dir, frame_name)

            # Use YOLO to predict objects in the current frame
            results = model.predict(frame, conf=confidence_threshold, show=False)

            # Collect annotations for all detected objects
            annotations = []
            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    class_name = result.names[int(cls)]
                    if class_name == "person":
                        # Normalize bounding box coordinates for YOLO format
                        img_h, img_w, _ = frame.shape
                        x1, y1, x2, y2 = map(int, box)
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h

                        # Placeholder class ID (3) for later manual labeling
                        annotations.append(f"{classNumber} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Save the frame if at least one person is detected
            if annotations:
                cv2.imwrite(image_path, frame)

                # Save the annotation file
                label_name = f"{os.path.splitext(video_name)[0]}_frame{frame_count}.txt"
                label_path = os.path.join(labels_dir, label_name)
                with open(label_path, "w") as f:
                    f.write("\n".join(annotations))

        cap.release()

    print("Processing complete.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        scriptName= os.path.basename(__file__)
        print(f"Usage: python3 {scriptName} <path_to_video_collection> <path_to_extracted_data class_number")
        exit(1)
    inputDir= sys.argv[1]
    outputDir= sys.argv[2]
    classNumber= sys.argv[3]
    extract_persons_from_videos(inputDir, outputDir, classNumber)

