Person recognition in your camera videos

Overview

Sort your home videos into the ones containing people and not. Furthermore mark the ones containing people with the custom names of people in the video. For this to work, you need to train your custom network with the videos containing only that one person

Features

- Test your Torch CUDA readiness (torchDiag.py)
- Sort the videos into the ones containing people and the ones that don't (extractPersons.py)
- Prepare training data for your custom network (extractPersons.py)
- Add suffixes containing people's names into the names of the videos (recognizePersons.py)
- Remove all previously added suffixes (removeSuffixes.py)

Installation

$ git clone https://github.com/vvkleban/person-detection
$ cd person-detection
# Activate your virtual python environment
(venv) $ python -m pip install -U ultralytics opencv-python-headless
(venv) $ curl -OL https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt

Usage

Sort videos into folders corresponding to whether people were detected in them
./detectPersons.py /your/path/to/videos
This script will create Person and noPerson subfolders inside /your/path/to/videos and move video files correspondingly

Custom person detection

1. If you want your videos to be marked with the names of the people in them, you should train your custom network with a custom training data. Create folders (one per person you wish to recognize), which will contain the videos of that single person. After ./detectPersons.py finishes its job, you need to manually browse the videos in Person folder and find those that contain _only one_ person in them. All person YOLO boxes from the frames of the video will be used as training data for your custom data.

2. Prepate your personDataset.yaml with the names of the people your custom network will be classifying your videos into

3. Prepare your training data with ./extractPersons.py:

usage: extractPersons.py [-h] [-c CONFIDENCE] [-f FRAME_SKIP] input_dir output_dir class_number

Extract persons from videos using YOLOv8.

positional arguments:
  input_dir             Path to the video collection.
  output_dir            Path to save the extracted data (images and labels).
  class_number          Class number for person annotation.

options:
  -h, --help            show this help message and exit
  -c CONFIDENCE, --confidence CONFIDENCE
                        Confidence threshold for YOLO detection (default: 0.85).
  -f FRAME_SKIP, --frame-skip FRAME_SKIP
                        Process every N-th frame (default: 1, i.e. every frame).

Your input_dir will be the one containing all videos of a particular person. Your output_dir will contain extracted labeled images of that person for training data. Make sure it corresponds to the path you've specified in personDataset.yaml as train or val keys. class_number option is an integer specifying index in the names array of personDataset.yaml. NOTE that it starts with 0. YOLO documentation suggests minimum of 10,000 images per class. However the more images you have, the longer your training will take. In order to adjust the number of images produced, you can use frame_skip option. By default ./extractPersons.py will use every video frame with a person as a training image. However if your videos have too many frames, you can make the script pick up every frame_skip frame i.e. if frame_skip is 2, it will pick every second frame

4. Train your custom network:
(venv) $ yolo task=detect mode=train model=yolov8n.pt data=personDataset.yaml epochs=150 imgsz=640
This will train your custom network as directed in personDataset.yaml. epochs option sets how many training iterations of your network yolo will perform. You should custom pick this value based on the convergence of loss variables. Please read YOLO documentation for more information on this.

5. After your training is done, yolo script will indicate the location of your custom networks. Usually it's ~/runs/detect/train<whatever number of times you ran yolo trining>/weights/best.pt You can use this network with recognizePersons.py script to mark your videos with people's names:

usage: recognizePersons.py [-h] [-s SKIP] [-c CONFIDENCE] model_path video_path

YOLO-based video object detection and renaming.

positional arguments:
  model_path            Path to the YOLO model (e.g., best.pt)
  video_path            Path to the video collection directory

options:
  -h, --help            show this help message and exit
  -s SKIP, --skip SKIP  Frame skip interval (default: 20)
  -c CONFIDENCE, --confidence CONFIDENCE
                        Confidence threshold (default: 0.9)

6. Iterate
You might not be happy with the performance of your custom network for the first few iterations. You would want to refine yout model using the videos recognizePersons.py misattributes. When preparing your additional training data, it would be wise to strip your new video files of their person suffixes before adding them to your training data in order to avoid duplicates. You can do it with removeSuffixes.py:

usage: removeSuffixes.py [-h] yaml_path video_path

Remove suffixes from filenames based on YOLO class names.

positional arguments:
  yaml_path   Path to the YOLO training YAML file
  video_path  Path to the video collection directory

options:
  -h, --help  show this help message and exit

