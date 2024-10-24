###################################################################
# Import required libraries                                       #
###################################################################
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import os
from homography import pose_homography_detections
import torch
import gradio as gr
from PIL import Image, ImageColor
import imageio.v2 as imageio
import gc
import skimage
import pandas as pd


###################################################################
# Initialize variables for loading AI models and maps for display #
###################################################################
PLAYER_DETECTION_MODEL = YOLO('Models/MyPlayerDetector3.pt')
KEYPOINT_MODEL = YOLO("Models/noflip_keypoint_pose.pt") #('Models/HomeImagesModel.pt')
PLAYER_DETECTION_MODEL.model.to('cuda:0')
KEYPOINT_MODEL.model.to('cuda:0')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
orig_map = cv2.imread('Models/southfield.jpg', 1) 

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#FFFFFF', '#000000']), #(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)


###################################################################
# Function to determine team colors                               #
###################################################################
def create_colors_info(team1_name, team1_p_color, team1_gk_color, team2_name, team2_p_color, team2_gk_color):
    team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB")
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB")
    team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB")
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB")

    colors_dic = {
        team1_name:[team1_p_color_rgb, team1_gk_color_rgb],
        team2_name:[team2_p_color_rgb, team2_gk_color_rgb]
    }
    colors_list = colors_dic[team1_name]+colors_dic[team2_name] # Define color list to be used for detected player team prediction
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] # Converting color_list to L*a*b* space
    return colors_dic, color_list_lab

colors_dic, color_list_lab = create_colors_info('CB south', 'white', 'pink', 'Holicong', 'black', 'green')
nbr_team_colors = len(list(colors_dic.values())[0])

###################################################################
# Team assignment logic                                           #
###################################################################
def simple_team_assigner(frame, detections):
  obj_palette_list = []        # Initialize players color palette list
  palette_interval = (0,4)     # adjust this later
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
  for i in range(len(detections.xyxy)):
      bbox = detections.xyxy[i,:]                         # Get bbox info (x,y,x,y)
      obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]       # Crop bbox out of the frame
      obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
      center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1])
      center_filter_x2 = (obj_img_w//2)+(obj_img_w//5)
      center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1])
      center_filter_y2 = (obj_img_h//3)+(obj_img_h//5)
      center_filter = obj_img[center_filter_y1:center_filter_y2, 
                              center_filter_x1:center_filter_x2]
      obj_pil_img = Image.fromarray(np.uint8(center_filter))   
      reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)                   # Convert to web palette (216 colors)
      reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)                   # Convert to web palette (216 colors)
      palette = reduced.getpalette()                                                  # Get palette as [r,g,b,r,g,b,...]
      palette = [palette[3*n:3*n+3] for n in range(256)]                              # Group 3 by 3 = [[r,g,b],[r,g,b],...]
      color_count = [(n, palette[m]) for n,m in reduced.getcolors()]                  # Create list of palette colors with their frequency
      RGB_df = pd.DataFrame(color_count, columns = ['cnt', 'RGB']).sort_values(       # Create dataframe based on defined palette interval
                          by = 'cnt', ascending = False).iloc[
                              palette_interval[0]:palette_interval[1],:]
      palette = list(RGB_df.RGB)                                                      # Convert palette to list (for faster processing)
      obj_palette_list.append(palette) # size equals number of detected players, for each player histogram of color palatte
      #player_list.append(obj_img)
      #crop_list.append(obj_pil_img)

## Calculate distances between each color from every detected player color palette and the predefined teams colors
  players_distance_features = []
  # Loop over detected players extracted color palettes
  for palette in obj_palette_list:
      palette_distance = []
      palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]  # Convert colors to L*a*b* space
      # Loop over colors in palette
      for color in palette_lab:
          distance_list = []
          # Loop over predefined list of teams colors
          for c in color_list_lab:
              #distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
              distance = skimage.color.deltaE_cie76(color, c)                             # Calculate Euclidean distance in Lab color space
              distance_list.append(distance)                                              # Update distance list for current color
          palette_distance.append(distance_list)                                          # Update distance list for current palette
      players_distance_features.append(palette_distance)                                  # Update distance features list

  ## Predict detected players teams based on distance features
  detections.class_id = []
  # Loop over players distance features
  for i,distance_feats in enumerate(players_distance_features):
      vote_list=[]
      # Loop over distances for each color 
      for dist_list in distance_feats:
          team_idx = dist_list.index(min(dist_list))//nbr_team_colors                     # Assign team index for current color based on min distance
          vote_list.append(team_idx)                                                      # Update vote voting list with current color team prediction
      detections.class_id.append(max(vote_list, key=vote_list.count,default=0))                      # Predict current player team by vote counting


###################################################################
# Function to determine homographic projection                    #
###################################################################
def run_projection(infile, progress=gr.Progress()):
  torch.cuda.empty_cache()
  gc.collect()

  SOURCE_VIDEO_PATH = infile
  outvideo_path = "OutputVideos/output_video.mp4"
  prev_map = orig_map.copy()

  # find number of video frames to display progress
  cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cap.release()
  f = 0
  with imageio.get_writer(outvideo_path, fps=24,format='FFMPEG') as writer:
    map = 0
    results = PLAYER_DETECTION_MODEL.predict(infile, device="cuda:0", stream=True, stream_buffer=True, save=True,conf=0.3, imgsz=640)
    keypoint_results = KEYPOINT_MODEL.predict(infile, device = 'cuda:0', stream=True, save=True, stream_buffer=True, conf=0.5,imgsz=640)

    for result,keypoint in zip(results,keypoint_results): #frame_generator:
      frame = result.orig_img
      detections = sv.Detections.from_ultralytics(result)
      all_detections = detections
      players_detections = all_detections

      simple_team_assigner(frame, players_detections)
      all_detections = players_detections
      annotated_frame = frame.copy()
      annotated_frame = ellipse_annotator.annotate(
          scene=annotated_frame,
          detections=all_detections)

      map = orig_map.copy()
        ###### Process the radar with keypoint model
      if(pose_homography_detections(keypoint, map, players_detections)):
        prev_map = map
        map = Image.fromarray(cv2.cvtColor(map, cv2.COLOR_BGR2RGB))
      else:
        map = Image.fromarray(cv2.cvtColor(prev_map, cv2.COLOR_BGR2RGB))

      annotated_frame = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

      #merge images
      img2 = map.rotate(-90, Image.NEAREST, expand = 1)
      img1 = annotated_frame.resize((img2.width, annotated_frame.height*2), Image.Resampling.LANCZOS)
      total_width = max(img1.width,img2.width)
      total_height = img1.height+ img2.height
      
      # Create a new blank image with the total width and max height
      new_image = Image.new('RGB', (total_width, total_height))
      
      # Paste the images into the new image
      new_image.paste(img1, (0, 0))  # Paste the first image at the left side
      new_image.paste(img2, (0, img1.height))  # Paste the second image to the right of the first

      writer.append_data(np.array(new_image))
      
      try:
        progress(f/total_frames, 'Generating Output Video')
        f += 1
      except NameError:
        yield new_image, str(f)
        f += 1

  yield  gr.Video(value=outvideo_path,visible=True)


