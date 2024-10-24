###################################################################
# Import required libraries                                       #
###################################################################
import cv2
import ffmpeg


###################################################################
# Code to convert video to 640x640 pixels for AI model input      #
###################################################################

vid_path = r"InputVideos/kickoff.mp4"
output_path = r"InputVideos/640_kickoff.mp4"
input_vid = ffmpeg.input(vid_path)
vid = (
    input_vid
    .filter('scale', 640,640)
    .output(output_path)
    .overwrite_output()
    .run()
)




