#!/usr/bin/env python3

import os
import sys
import argparse
import yaml

def load_classes_from_yaml(yaml_path):
    """Loads class names from a YOLO training YAML file."""
    try:
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
            return data.get("names", [])
    except Exception as e:
        print(f"Error reading YAML file: {e}", file=sys.stderr)
        sys.exit(1)

def removeSuffixes(filename, suffixes):
    """
    Removes specific suffixes from a filename before the file extension.
    """
    name, ext = os.path.splitext(filename)
    for suffix in suffixes:
        name = name.replace(suffix, "")
    return name + ext

def main():
    """Parses command-line arguments and processes video filenames."""
    parser = argparse.ArgumentParser(description="Remove suffixes from filenames based on YOLO class names.")

    # Positional arguments
    parser.add_argument("yaml_path", help="Path to the YOLO training YAML file")
    parser.add_argument("video_path", help="Path to the video collection directory")

    args = parser.parse_args()

    # Load class names from YAML
    class_names = load_classes_from_yaml(args.yaml_path)
    suffixes = [f"_{name}" for name in class_names]

    # Process video files
    for videoName in sorted(os.listdir(args.video_path)):
        if not videoName.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {videoName}")
            continue

        videoPath = os.path.join(args.video_path, videoName)
        cleanedName = removeSuffixes(videoName, suffixes)
        cleanedPath = os.path.join(args.video_path, cleanedName)

        if cleanedName != videoName:
            print(f"Renaming {videoName} to {cleanedName}")
            os.rename(videoPath, cleanedPath)

if __name__ == "__main__":
    main()

