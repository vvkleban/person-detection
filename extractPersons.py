#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
import os
import argparse
import sys

def extract_persons_from_videos(input_dir, output_dir, classNumber, confidence_threshold=0.85, frame_skip=1):
    # Load the pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")  # YOLOv8 Nano model (lightweight and efficient)

    # Create directories for images and labels
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Loop over all videos in the given input directory
    for videoName in os.listdir(input_dir):
        videoPath = os.path.join(input_dir, videoName)

        # Skip non-video files
        if not videoName.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {videoName}")
            continue

        print(f"Processing video: {videoName}")
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print(f"Error: Could not open video {videoName}")
            continue

        frameCount = 0
        extractedImages = []
        extractedLabels = []
        multiplePersonsInAFrame = False

        # Loop over every frame in the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frameCount += 1

            # Skip frames based on the specified interval
            if frameCount % frame_skip != 0:
                continue

            personCount = 0
            # Use YOLO to predict objects in the current frame
            results = model.predict(frame, conf=confidence_threshold, show=False)

            #Loop over basically a list of one element, as it is one result per image.
            #TODO: reanalyze the API - perhaps YOLO can handle videos without  the need for cv2
            for result in results:
                # Loop over detected boxes and classes
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    class_name = result.names[int(cls)]
                    if class_name == "person":
                        personCount += 1
                        if personCount > 1:
                            multiplePersonsInAFrame = True
                            # Show the frame for debugging when multiple persons are detected
                            model.predict(frame, conf=confidence_threshold, show=True)
                            break
                        # Normalize bounding box coordinates for YOLO format
                        img_h, img_w, _ = frame.shape
                        x1, y1, x2, y2 = map(int, box)
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        # Prepare file names for the image and label
                        imageName = f"{os.path.splitext(videoName)[0]}_frame{frameCount}.jpg"
                        labelName = f"{os.path.splitext(videoName)[0]}_frame{frameCount}.txt"
                        imagePath = os.path.join(images_dir, imageName)
                        labelPath = os.path.join(labels_dir, labelName)
                        extractedImages.append(imagePath)
                        cv2.imwrite(imagePath, frame)
                        with open(labelPath, "w") as f:
                            f.write(f"{classNumber} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        extractedLabels.append(labelPath)
                else:
                    # Continue the outer loop if the inner loop wasn't broken
                    continue
                # Break out of the result loop if a break was encountered inside the inner loop
                break
            else:
                # Continue to the next frame if the result loop was not broken
                continue
            # Break out of the frame loop if the result loop was broken
            break

        cap.release()
        # If more than one person was detected in any frame, delete all extracted frames
        # and mark the video as bad by renaming it.
        if multiplePersonsInAFrame:
            print(f"Multiple persons detected in {videoName}, deleting extracted frames and renaming video.")
            for file in extractedImages + extractedLabels:
                if os.path.exists(file):
                    os.remove(file)
            os.rename(videoPath, videoPath + ".bad")

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract persons from videos using YOLOv8.")
    parser.add_argument("input_dir", help="Path to the video collection.")
    parser.add_argument("output_dir", help="Path to save the extracted data (images and labels).")
    parser.add_argument("class_number", type=int, help="Class number for person annotation.")
    parser.add_argument("-c", "--confidence", type=float, default=0.85,
                        help="Confidence threshold for YOLO detection (default: 0.85).")
    parser.add_argument("-f", "--frame-skip", type=int, default=1,
                        help="Process every N-th frame (default: 1, i.e. every frame).")
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    extract_persons_from_videos(
        args.input_dir,
        args.output_dir,
        args.class_number,
        confidence_threshold=args.confidence,
        frame_skip=args.frame_skip
    )

