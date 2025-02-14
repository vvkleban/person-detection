from ultralytics import YOLO
import cv2
import sys
import os

class VideoParsingError(Exception):
    pass

def removeSuffixes(filename, suffixes):
    """
    Removes specific suffixes from a filename before the file extension.
    """
    name, ext = os.path.splitext(filename)
    for suffix in suffixes:
        name= name.replace(suffix, "")
    return name + ext

class VideoAnalyzer:

    model = YOLO("/home/vova/runs/detect/train10/weights/best.pt")
    suffixes= ["_Vova", "_Sha", "_Leo"]

    def isPersonInVideo(self, videoPath, frame_skip=20, confidence_threshold=0.9):
        print(f"Analyzing {videoPath}")
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            raise VideoParsingError(f"Error: Could not open video {videoPath}")

        frame_count = 0
        vova= False
        sha= False
        leo= False

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
                        if class_name == "Vova":  # Check if a person is detected
                            vova = True
                        elif class_name == "Sha":
                            sha= True
                        elif class_name == "Leo":
                            leo= True

                if vova and sha and leo:
                    break

            cap.release()

            suffix= ""
            if vova:
                suffix += self.suffixes[0]
            if sha:
                suffix += self.suffixes[1]
            if leo:
                suffix += self.suffixes[2]

            return suffix

        except Exception as e:
            raise VideoParsingError("Failed parsing video {videoPath}: {e}") from e

def main():
    if len(sys.argv) < 2:
        scriptName= os.path.basename(__file__)
        print(f"Usage: python3 {scriptName} <path_to_video_collection>")
        exit(1)

    inputDir= sys.argv[1]
    videoAnalyzer= VideoAnalyzer()

    for videoName in sorted(os.listdir(inputDir)):
        if not videoName.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {videoName}")
            continue

        videoPath = os.path.join(inputDir, videoName)
        cleanedName = removeSuffixes(videoName, videoAnalyzer.suffixes)
        cleanedPath = os.path.join(inputDir, cleanedName)
        if cleanedName != videoName:
            print(f"Moving {videoName} to {cleanedName}\n")
            os.rename(videoPath, cleanedPath)
            videoName= cleanedName
            videoPath= cleanedPath

        try:
            suffix= videoAnalyzer.isPersonInVideo(videoPath)
            if suffix:
                newName = os.path.splitext(videoName)[0] + suffix + os.path.splitext(videoName)[1]
                print(f"Moving {videoName} to {newName}")
                os.rename(videoPath, os.path.join(inputDir, newName))
        except Exception as e:
            print(f"Failed moving {videoName}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

