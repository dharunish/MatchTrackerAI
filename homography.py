###################################################################
# Import required libraries                                       #
###################################################################
import cv2
import numpy as np
from ultralytics import YOLO

###################################################################
# Initialize keypoint locations in the map                        #
###################################################################
all_dst_pts = [
[865,1618], #- 10L 0
[865,1457], #- 20L   1
[865,1298],#- 30L  2
[865,1139],#- 40L  3
[865,979],#- '50'  4
[865,819],#- 40R  5
[865,660],#- 30R  6
[865,501],#- 20R   7
[866,342], #- 10R   8
[635,23],#- RGoalN 9
[994,23],#- FlagR  10
[994,1936],#- FlagL  11
[635,1933],#- LGoalN 12
[25,1925],#- UFlagL 13
[267,1613],#- U10L 14
[268,1453],#- U20L 15
[269,1295],#- U30L 16
[267,1134],#- U40L 17
[268,974],#- U50 18
[268,815],#- U40R 19
[269,654],#- U30R 20
[269,497],#- U20R 21
[269,336],#- U10R 22
[25,16],#- UFlagR 23
[409,21], # - RGoalF 24
[407,1936],#- LGoalF 25
]
all_dst_pts = np.array(all_dst_pts)


###################################################################
# Function to convert a point from video location to map location #
###################################################################
def convert_point(xy,H):
  point = np.array([xy[0],xy[1],1])
  point = np.dot(H,point)
  point /= point[2]
  return [int(point[0]),int(point[1])]


###################################################################
# Function to determine homography matrix and perform projection  #
###################################################################
def pose_homography_detections(result, map, detections):
  homog = []

  keypoints = result.keypoints.cpu().numpy()[0]
  mask = keypoints.xy[0,:,0]>0
      
  src_pts = keypoints.xy[0,mask]
  dst_pts = all_dst_pts[mask]
  
  if len(src_pts) > 3:
    homog, _ = cv2.findHomography(src_pts,dst_pts) 

  if len(homog):
    for i in range(len(detections)):
      x1, y1, x2, y2 = detections.xyxy[i]
      centerY = int((y1 + y2) // 2)
      centerX = int((x1 + x2) // 2)

      if detections.class_id[i] == 1:
        player_color = (0,0,0)
      else:
        player_color = (255,255,255)

      pxy = [centerX,  y2]
      dest_point = convert_point(pxy,homog)
      if dest_point[0] > 10 and dest_point[0] < 1130:
        cv2.circle(map, dest_point, 20, player_color, thickness=-1)
    return True
  else:
    return False