# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:40:57 2016

@author: SoonYau
"""

import cv2
import numpy as np

video=cv2.VideoCapture('./challenge.mp4')
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = video.get(cv2.CAP_PROP_FPS)

print(length,width,height,fps)
frame_idx=0
while(video.isOpened()):
   ret,frame=video.read()
   frame_idx+=1
   if (frame_idx==130):
       cv2.imwrite('test.jpg',frame)
   cv2.imshow("frame",frame)
   if (cv2.waitKey(int(1000/fps)) & 0xFF == ord('q')):
       break
   
video.release()
