# Lane and Car Detection
This repository aims to detect lane lines of the road using sobel filter and sliding window search algorithm and detect cars using yolov3.
# Setup
- Install [Anaconda Distribution](https://www.anaconda.com/products/distribution).
- In Anaconda shell install the following libraries:
```
pip install opencv-python
pip install opencv-contrib-python
pip install moviepy
```
- Install the weights and cfg files of YOLOv3-320 in the project directory through the follwing [link](https://pjreddie.com/darknet/yolo/)

# How To Run The Project
- Navigate to the project directory.
- Open terminal.
- Run the following script using the following command:

### Windows
```cmd
conda activate base
batchScript.bat <input_video_dir> <output_video_dir> --debug <0/1/2/3>
```
**_NOTE:_** 0 for normal mode, 1 for debugging mode, 2 for draw car mode, and 3 for debugging with draw car mode.

### Linux
```bash
conda activate base
./run.sh -<n/d/c/b>
<input_video_dir> <output_video_dir>
```
**_NOTE:_** n for normal mode, d for debugging mode, c for draw car mode, and b for debugging with draw car mode.

# Steps of The Project

- Use sobel edge detector to detect lane edges.
- Apply a perspective transform ("birds-eye view").
- Detect lane pixels and get the best curve fitting.
- Determine the curvature of the lane and vehicle position with respect to center.
- Apply inverse perspective transform to the detected lanes to add on the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
- Load weights and cfg files of yolov3.
- Apply net forward.
- Draw boxes on the detected cars.
- Detected cars can be visualized on the left of the video.
- Output can be visualized in debugging mode showing all the stages of detecting the lane and cars.
 
## Normal mode:

![normal](https://user-images.githubusercontent.com/55032660/169923971-43b734d8-140e-4789-8fe9-3f2d70ee8e61.png "normal mode")

## Debugging mode:

![debuggin mode](https://user-images.githubusercontent.com/55032660/169924065-42aca7af-866a-465f-9c95-0ccd198492a6.png "dubugging mode")

## Draw car mode:

![draw car mode](https://user-images.githubusercontent.com/55032660/169924129-4120593d-13c3-4eac-bf45-bbddb4d884a6.png "draw car mode")

## Debugging with draw car mode:

![debug with car mode](https://user-images.githubusercontent.com/55032660/169924172-42ded626-36a6-4f42-80ae-b186fe635822.png "debugging with draw car mode")
