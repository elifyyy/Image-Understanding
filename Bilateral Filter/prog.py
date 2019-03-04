#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
import time


# In[3]:


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian_func(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))

def my_bilateral_filter(src_img,filtered_img,x,y,diameter,sigma_i,sigma_s):
    window_limit = diameter // 2 
    wp = 0
    filtered_value = 0
    for i in range(diameter):
        for j in range(diameter):
            neighbour_x_index = x - (window_limit - j)
            neighbour_y_index = y - (window_limit - i)
            if neighbour_x_index >= len(src_img):
                neighbour_x_index = neighbour_x_index - len(src_img)
            if neighbour_y_index >= len(src_img[0]):
                neighbour_y_index = neighbour_y_index - len(src_img[0])
                
            gi = gaussian_func(src_img[neighbour_x_index][neighbour_y_index]-src_img[x][y],sigma_i)
            gs = gaussian_func(distance(x,y,neighbour_x_index,neighbour_y_index),sigma_s)
            filtered_value = filtered_value + src_img[neighbour_x_index][neighbour_y_index]*gi*gs
            wp_temp = gi*gs
            wp = wp + wp_temp
    filtered_value = filtered_value//wp
    filtered_image[x][y] = filtered_value

def applying_filter(src_img,filtered_img,sigma_i,sigma_s,diameter):
    for y in range(len(src_img[0])):
        for x in range (len(src_img)):
            my_bilateral_filter(src_img,filtered_img,x,y,diameter,sigma_i,sigma_s)
    return filtered_img        
            
    


# In[4]:


src = cv2.imread("in_img.jpg",0)
filtered_image = np.zeros(src.shape)
filtered_image_own = applying_filter(src,filtered_image,10.0,14.0,7)
cv2.imwrite("filtered_image_own.png", filtered_image_own)

filtered_image_OpenCV = cv2.bilateralFilter(src, 7, 10.0, 14.0)
cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)


# In[ ]:


"""
I tried different sigma values to see how they effect the image.

Small values did not changed the image much. I gave both intensity sigma and spatial sigma 3 and there was 
nearly no effect on the image. At least i couldn't see any difference.

When i increase sigma values the differences between orjinal image and filtered image increase to.

I gave intensity sigma 10 and spatial sigma 2 ; then i gave intensity sigma 20 and spatial sigma 2 to see the
effect of intensity sigma. I observed that when intensity sigma increase, color intensity became more uniform , 
sharpness of  lines decreased and image became more blur.

To observed the effect of spatial sigma first i gave it 20 then 2 while i keep intensity sigma the same. 
But i can not see the difference between two images.I could not observe visible effect.

I choose these sigma values because they are not close the end and begin of the [2,20] interval, if they were
too small i couldn't observe differences from orjinal image and if they were too big, filtered image would be 
too blur.




"""

