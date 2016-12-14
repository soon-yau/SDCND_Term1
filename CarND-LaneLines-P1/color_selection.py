# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:13:55 2016

@author: SoonYau
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def whiten_yellow_lines(img):
    """
    Change yellow lines to white and return the processed image
    """
    color_select = np.copy(img)
    red_threshold = 120
    green_threshold = 120
    blue_threshold = 110

    thresholds = (img[:,:,0] < blue_threshold) \
               & (img[:,:,1] > green_threshold) \
               & (img[:,:,2] > red_threshold)  

    color_select[thresholds,:] = [255,255,255]
    return color_select

    
img=cv2.imread('test.jpg')
cv2.imshow('image',whiten_yellow_lines(img))
cv2.waitKey(0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()