# Lane Detection
This repository aims to detect lane lines of the road using sobel filter and sliding window search algorithm.
# Setup
- Install [Anaconda Distribution](https://www.anaconda.com/products/distribution).
- In Anaconda shell install the following libraries:
```
pip install opencv-python
pip install opencv-contrib-python
pip install moviepy
```
# How To Run The Project
- Navigate to the project directory.
- Open terminal.
- Run batch script using the following command:
```cmd
batchScript.bat <input_video_dir> <output_video_dir> --debug <0/1>
```
**_NOTE:_** 0 for normal mode while 1 for debugging mode.
# Steps of The Project
- Use sobel edge detector to detect lane edges.
- Apply a perspective transform ("birds-eye view").
- Detect lane pixels and get the best curve fitting.
- Determine the curvature of the lane and vehicle position with respect to center.
- Apply inverse perspective transform to the detected lanes to add on the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
- Output can be visualized in debugging mode showing all the stages of detecting the lane.
 
![debugg](https://user-images.githubusercontent.com/55032660/164126383-f8660bbe-61c6-4117-a8ca-06eabd17e841.png) ![not debug](https://user-images.githubusercontent.com/55032660/164126386-d36b7a2b-33b0-40ac-9cef-329e3f299f6d.png)
