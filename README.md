# Person Recognition in Your Camera Videos

## Overview

Sort your home videos into those containing people and those that don't. Furthermore, mark the ones containing people with custom names. For this to work, you need to train your custom network with videos containing only one person.

## Features

- **Test your Torch CUDA readiness** (`torchDiag.py`)
- **Sort videos** into those with and without people (`detectPersons.py`)
- **Prepare training data** for your custom network (`extractPersons.py`)
- **Add names as suffixes** to video filenames (`recognizePersons.py`)
- **Remove all previously added suffixes** (`removeSuffixes.py`)

## Installation

```bash
$ git clone https://github.com/vvkleban/person-detection
$ cd person-detection
# Activate your virtual python environment
(venv) $ python -m pip install -U ultralytics opencv-python-headless
(venv) $ curl -OL https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt
```

## Usage

### Sort Videos by Person Detection

```bash
./detectPersons.py /your/path/to/videos
```

This script will create `Person` and `noPerson` subfolders inside `/your/path/to/videos` and move video files accordingly.

### Custom Person Detection

#### 1. Prepare Training Data

1. Manually review videos in the `Person` folder and select ones containing only a single person.
2. Organize these videos into separate folders for each person.
3. Create a `personDataset.yaml` file with the names of the people for classification.

#### 2. Extract Training Data

Run the following command to extract labeled images:

```bash
usage: extractPersons.py [-h] [-c CONFIDENCE] [-f FRAME_SKIP] input_dir output_dir class_number
```

**Arguments:**

- `input_dir`: Path to video collection.
- `output_dir`: Path to save extracted images and labels.
- `class_number`: Index in `personDataset.yaml` (starting from 0).

**Options:**

- `-c, --confidence CONFIDENCE`: Confidence threshold (default: 0.85).
- `-f, --frame-skip FRAME_SKIP`: Process every N-th frame (default: 1).

Example:

```bash
./extractPersons.py /videos/person1 /training_data/person1 0
```

Your **input_dir** will be the one containing all videos of a particular person. Your **output_dir** will contain extracted labeled images of that person for training data. Make sure it corresponds to the path you've specified in personDataset.yaml as train or val keys. **class_number** option is an integer specifying index in the names array of personDataset.yaml. **NOTE** that it starts with 0. YOLO documentation suggests minimum of 10,000 images per class. However the more images you have, the longer your training will take. In order to adjust the number of images produced, you can use **frame_skip** option. By default ./extractPersons.py will use every video frame with a person as a training image. However if your videos have too many frames, you can make the script pick up every frame_skip frame e.g. if frame_skip is 2, it will pick every second frame

#### 3. Train Your Custom Model

```bash
(venv) $ yolo task=detect mode=train model=yolov8n.pt data=personDataset.yaml epochs=150 imgsz=640
```
This will train your custom network as directed in personDataset.yaml. **epochs** option sets how many training iterations of your network yolo will perform. You should custom pick this value based on the convergence of loss variables. Please read YOLO documentation for more information on this.

After your training is done, yolo script will indicate the location of your custom networks. Usually it's ~/runs/detect/train<whatever number of times you ran yolo trining>/weights/best.pt

#### 4. Recognize People in Videos

```bash
usage: recognizePersons.py [-h] [-s SKIP] [-c CONFIDENCE] model_path video_path
```

**Arguments:**

- `model_path`: Path to trained model (e.g., `best.pt`).
- `video_path`: Directory containing videos.

**Options:**

- `-s, --skip SKIP`: Frame skip interval (default: 20).
- `-c, --confidence CONFIDENCE`: Confidence threshold (default: 0.9).

Example:

```bash
(venv) $ ./recognizePersons.py best.pt /videos_to_process
```

#### 5. Iterate and Improve

If results are unsatisfactory, refine your model using misattributed videos. Before adding new videos to training data, strip them of previously added person suffixes:

```bash
usage: removeSuffixes.py [-h] yaml_path video_path
```

Example:

```bash
(venv) $ ./removeSuffixes.py personDataset.yaml /videos_to_process

