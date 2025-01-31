from ultralytics import YOLO
import cv2
import os
import sys

def extract_persons_from_videos(input_dir, output_dir, classNumber, confidence_threshold=0.9):
    # Load the pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")  # YOLOv8 Nano model (lightweight and efficient)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Loop over all videos in a given path
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
        multiplePersonsInAFrame= False

        # Loop over every frame of a given video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frameCount += 1
            personCount= 0
            # Use YOLO to predict objects in the current frame
            results = model.predict(frame, conf=confidence_threshold, show=False)

            #Loop over basically a list of one element, as it is one result per image.
            #TODO: reanalyze the API - perhaps YOLO can handle videos without  the need for cv2
            for result in results:

                # Loop over all boxes in a given image
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    class_name = result.names[int(cls)]
                    if class_name == "person":
                        personCount += 1
                        if personCount > 1:
                            multiplePersonsInAFrame= True
                            model.predict(frame, conf=confidence_threshold, show=True)
                            break
                        # Normalize bounding box coordinates for YOLO format
                        img_h, img_w, _ = frame.shape
                        x1, y1, x2, y2 = map(int, box)
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        # Save the result into the training library
                        imageName = f"{os.path.splitext(videoName)[0]}_frame{frameCount}.jpg"
                        labelName = f"{os.path.splitext(videoName)[0]}_frame{frameCount}.txt"
                        imagePath = os.path.join(images_dir, imageName)
                        labelPath = os.path.join(labels_dir, labelName)
                        extractedImages.append(imagePath)
                        cv2.imwrite(imagePath, frame)
                        with open(labelPath, "w") as f:
                            f.write(f"{classNumber} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        extractedLabels.append(labelPath)
                else: # "else" is for the inner "for" loop - not for the "if"
                    continue # continue outer "for" loop as usual if the inner one wasn't broken
                break # break the outer "for" loop if the inner one was broken
            else: # Now the same logic for the "while" loop
                continue
            break

        cap.release()
        # If in any frame of the video more than one person was detected, delete all
        # frames created so far and ban the video from our library by renaming it
        if multiplePersonsInAFrame:
            print(f"Multiple persons detected in {videoName}, deleting extracted frames and renaming video.")
            for file in extractedImages + extractedLabels:
                if os.path.exists(file):
                    os.remove(file)
            os.rename(videoPath, videoPath + ".bad")

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

