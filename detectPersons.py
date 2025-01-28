from ultralytics import YOLO
import cv2
import sys
import os
import shutil

class VideoParsingError(Exception):
    pass

class VideoAnalyzer:

    model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 Nano model

    def isPersonInVideo(self, videoPath, frame_skip=10, confidence_threshold=0.5):
        print(f"Analyzing {videoPath}")
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            raise VideoParsingError(f"Error: Could not open video {videoPath}")

        frame_count = 0
        person_detected = False

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
                #results = self.model.predict(frame, conf=confidence_threshold, show=False, device="cpu")
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
                return True
            else:
                print("No person detected in the video.")
                return False

        except Exception as e:
            raise VideoParsingError("Failed parsin video {videoPath}: {e}") from e

def main():
    if len(sys.argv) < 2:
        scriptName= os.path.basename(__file__)
        print(f"Usage: python3 {scriptName} <path_to_video_collection>")
        exit(1)

    inputDir= sys.argv[1]
    videoAnalyzer= VideoAnalyzer()

    person = os.path.join(inputDir, "Person")
    noPerson = os.path.join(inputDir, "noPerson")
    os.makedirs(person, exist_ok=True)
    os.makedirs(noPerson, exist_ok=True)

    for videoName in os.listdir(inputDir):
        if not videoName.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {videoName}")
            continue

        videoPath = os.path.join(inputDir, videoName)
        try:
            if videoAnalyzer.isPersonInVideo(videoPath):
                print(f"Moving {videoName} to {person}\n")
                shutil.move(videoPath, os.path.join(person, videoName))
            else:
                print(f"Moving {videoName} to {noPerson}\n")
                shutil.move(videoPath, os.path.join(noPerson, videoName))
        except Exception as e:
            print(f"Failed moving {videoName}: {e} \n", file=sys.stderr)


if __name__ == "__main__":
    main()

