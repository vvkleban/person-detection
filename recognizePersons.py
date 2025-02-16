#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
from ultralytics import YOLO

class VideoParsingError(Exception):
    pass

def removeSuffixes(filename, suffixes):
    """Removes specific suffixes from a filename before the file extension."""
    name, ext = os.path.splitext(filename)
    for suffix in suffixes:
        name = name.replace(suffix, "")
    return name + ext

class VideoAnalyzer:
    def __init__(self, model_path):
        # Load the model dynamically from the given path
        self.model = YOLO(model_path)
        # Extract class names dynamically
        self.class_names = self.model.names
        self.suffixes = {name: f"_{name}" for name in self.class_names.values()}

    def isPersonInVideo(self, videoPath, frame_skip=20, confidence_threshold=0.9):
        """Analyzes the video for the presence of trained objects and returns suffixes."""
        print(f"Analyzing {videoPath} with frame_skip={frame_skip}, confidence_threshold={confidence_threshold}")
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            raise VideoParsingError(f"Error: Could not open video {videoPath}")

        frame_count = 0
        detected_classes = set()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Use YOLO to predict objects in the current frame
                results = self.model.predict(frame, conf=confidence_threshold, show=False, device="cuda:0")

                for result in results:
                    for cls in result.boxes.cls:
                        class_name = self.class_names[int(cls)]  # Convert class ID to class name
                        print(f"Frame {frame_count}: Detected {class_name}")
                        detected_classes.add(class_name)

                # Stop early if all known classes are detected
                if set(self.suffixes.keys()).issubset(detected_classes):
                    break

            cap.release()

            # Construct suffix based on detected classes
            suffix = "".join([self.suffixes[name] for name in detected_classes if name in self.suffixes])
            return suffix

        except Exception as e:
            raise VideoParsingError(f"Failed parsing video {videoPath}: {e}") from e

def main():
    """Parses command-line arguments and runs video analysis."""
    parser = argparse.ArgumentParser(description="YOLO-based video object detection and renaming.")

    # Positional arguments
    parser.add_argument("model_path", help="Path to the YOLO model (e.g., best.pt)")
    parser.add_argument("video_path", help="Path to the video collection directory")

    # Optional arguments
    parser.add_argument("-s", "--skip", type=int, default=20, help="Frame skip interval (default: 20)")
    parser.add_argument("-c", "--confidence", type=float, default=0.9, help="Confidence threshold (default: 0.9)")

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Initialize the video analyzer with the provided model
    videoAnalyzer = VideoAnalyzer(args.model_path)

    # Process each video in the directory
    for videoName in sorted(os.listdir(args.video_path)):
        if not videoName.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {videoName}")
            continue

        videoPath = os.path.join(args.video_path, videoName)
        cleanedName = removeSuffixes(videoName, videoAnalyzer.suffixes.values())
        cleanedPath = os.path.join(args.video_path, cleanedName)
        if cleanedName != videoName:
            print(f"Renaming {videoName} to {cleanedName}")
            os.rename(videoPath, cleanedPath)
            videoName = cleanedName
            videoPath = cleanedPath

        try:
            suffix = videoAnalyzer.isPersonInVideo(videoPath, args.skip, args.confidence)
            if suffix:
                newName = os.path.splitext(videoName)[0] + suffix + os.path.splitext(videoName)[1]
                print(f"Renaming {videoName} to {newName}")
                os.rename(videoPath, os.path.join(args.video_path, newName))
        except Exception as e:
            print(f"Error processing {videoName}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

