from car_detection import * 
from lane_detection import * 
from moviepy.editor import VideoFileClip
import sys


def pipeline_v1(img):
    '''
    Function that detect and draw the lanes on image and boxes on detected cars and calculate vechile posiion , left curvaure and right curvature.
    
    Parameters:
        img: Input image of the road.
        
    Returns:
        processed_img: The output image of road wih detected lanes on it and detected cars.
       
    '''
    img = laneDetection_pipeline_v1(img)
    return carDetection_pipeline_v1(img)
def pipeline_v2(img):
    '''
    Function that detect and draw the lanes on image and boxes on detected cars and drawn cars on the left and calculate vechile posiion , left curvaure and right curvature.
    
    Parameters:
        img: Input image of the road.
        
    Returns:
        processed_img: The output image of road wih detected lanes on it and detected cars and drawn cars on the left.
       
    '''
    img = laneDetection_pipeline_v1(img)
    return carDetection_pipeline_v2(img)

def pipeline_v3(img):
    '''
    Function that detect and draw the lanes on image and calculate vechile position , left curvaure and right curvature
    and concatenated all stages to the img and draw boxes on detected cars.
    
    Parameters:
        img: Input image of the road.
        
    Returns: 
        processed_img: The output image of road wih detected lanes on it and boxes on detected cars concatenated with thresholded, warped to birdseye, 
        sliding windows and lane lines images.
       
    '''
    img = laneDetection_pipeline_v2(img)
    return carDetection_pipeline_v1(img)

def pipeline_v4(img):
    '''
    Function that detect and draw the lanes on image and calculate vechile position , left curvaure and right curvature
    and concatenated all stages to the img and draw boxes on detected cars and drawn cars on the left.
    
    Parameters:
        img: Input image of the road.
        
    Returns: 
        processed_img: The output image of road wih detected lanes on it and boxes on detected cars and drawn cars on left concatenated with thresholded, warped to birdseye, 
        sliding windows and lane lines images.
       
    '''
    img = laneDetection_pipeline_v2(img)
    return carDetection_pipeline_v2(img)

def createVideo(input_clip, output_clip="./output.mp4", mode="0"):
    '''
    Function that create video either in normal or debug mode.
    
    Parameters:
        input_clip: Input video directory.
        output_clip: Output video directory.
        debug: flag of debug mode.
        
    Returns:
        output_clip: The output video.
    '''
       
    if(mode == "0"):
        #Create video file pipeline
        clip1 = VideoFileClip(input_clip)
        out_clip = clip1.fl_image(pipeline_v1) #NOTE: this function expects color images!!
        out_clip.write_videofile(output_clip, audio=False)

    elif(mode == "1"):
        #Create video file pipeline
        clip1 = VideoFileClip(input_clip)
        out_clip = clip1.fl_image(pipeline_v3) #NOTE: this function expects color images!!
        out_clip.write_videofile(output_clip, audio=False)

    elif(mode == "2"):
        #Create video file pipeline
        clip1 = VideoFileClip(input_clip)
        out_clip = clip1.fl_image(pipeline_v2) #NOTE: this function expects color images!!
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