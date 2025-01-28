
import sys
import os

def removeSuffixes(filename, suffixes):
    """
    Removes specific suffixes from a filename before the file extension.
    """
    name, ext = os.path.splitext(filename)
    for suffix in suffixes:
        name= name.replace(suffix, "")
    return name + ext

def main():
    if len(sys.argv) < 2:
        scriptName= os.path.basename(__file__)
        print(f"Usage: python3 {scriptName} <path_to_video_collection>")
        exit(1)

    inputDir= sys.argv[1]
    suffixes= ["_Vova", "_Sha", "_Leo"]

    for videoName in sorted(os.listdir(inputDir)):
        if not videoName.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {videoName}")
            continue

        videoPath = os.path.join(inputDir, videoName)
        cleanedName = removeSuffixes(videoName, suffixes)
        cleanedPath = os.path.join(inputDir, cleanedName)
        if cleanedName != videoName:
            print(f"Moving {videoName} to {cleanedName}")
            os.rename(videoPath, cleanedPath)


if __name__ == "__main__":
    main()

