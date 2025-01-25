import cv2
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_person.py <path_to_video>")
        exit(1)

    video_path = sys.argv[1]

    # Load pre-trained model for detecting people (MobileNet-SSD example)
    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",  # Replace with the path to your prototxt file
        "res10_300x300_ssd_iter_140000.caffemodel"  # Replace with the path to your model weights
    )

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        exit(1)

    frame_skip = 30  # Process every nth frame for efficiency
    frame_count = 0
    person_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every nth frame
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Prepare the frame for detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Check if any detection is a person with confidence > 50%
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                print("Person detected!")
                person_detected = True
                break

        if person_detected:
            break

    cap.release()

    if person_detected:
        print("The video contains a person.")
    else:
        print("No person detected in the video.")

if __name__ == "__main__":
    main()

