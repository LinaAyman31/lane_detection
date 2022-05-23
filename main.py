from car_detection import * 
from lane_detection import * 
from moviepy.editor import VideoFileClip
import sys


def pipeline_v1(img):
    img = laneDetection_pipeline_v1(img)
    return carDetection_pipeline_v1(img)
def pipeline_v2(img):
    img = laneDetection_pipeline_v1(img)
    return carDetection_pipeline_v2(img)

def pipeline_v3(img):
    img = laneDetection_pipeline_v2(img)
    return carDetection_pipeline_v1(img)

def pipeline_v4(img):
    img = laneDetection_pipeline_v2(img)
    return carDetection_pipeline_v2(img)

def createVideo(input_clip, output_clip="./output.mp4", debug="0", draw_cars="0"):
    '''
    Function that create video either in normal or debug mode.
    
    Parameters:
        input_clip: Input video directory.
        output_clip: Output video directory.
        debug: flag of debug mode.
        
    Returns:
        output_clip: The output video.
    '''
       
    if(debug == "0" and draw_cars == "0"):
        #Create video file pipeline
        clip1 = VideoFileClip(input_clip)
        out_clip = clip1.fl_image(pipeline_v1) #NOTE: this function expects color images!!
        out_clip.write_videofile(output_clip, audio=False)

    elif(debug == "0" and draw_cars == "1"):
        #Create video file pipeline
        clip1 = VideoFileClip(input_clip)
        out_clip = clip1.fl_image(pipeline_v2) #NOTE: this function expects color images!!
        out_clip.write_videofile(output_clip, audio=False)

    elif(debug == "1" and draw_cars == "0"):
        #Create video file pipeline
        clip1 = VideoFileClip(input_clip)
        out_clip = clip1.fl_image(pipeline_v3) #NOTE: this function expects color images!!
        out_clip.write_videofile(output_clip, audio=False)

    else:
       #Create video file pipeline
        clip1 = VideoFileClip(input_clip)
        out_clip = clip1.fl_image(pipeline_v4) #NOTE: this function expects color images!!
        out_clip.write_videofile(output_clip, audio=False) 
        
    return output_clip  

def main():
    createVideo(sys.argv[1], sys.argv[2], sys.argv[3]) 

if __name__ == "__main__":
    main()