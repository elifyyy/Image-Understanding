#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from numpy import zeros
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv




def affine_deformation_matrix(lamda,alfa,tetha):#alfa tetha are radius
    rotation = [[lamda*math.cos(alfa),lamda*math.sin(alfa)],
               [-lamda*math.sin(alfa),lamda*math.cos(alfa)]]
    tilt = [[lamda*1/math.cos(tetha),0],
           [0,lamda*1]]
    affine_deformation = np.array( np.matmul(rotation,tilt) )
    
    return affine_deformation



def find_warp_size(affine_matrix):
    top_left_corner = np.matmul(affine_matrix,[[0],[0]])
    top_right_corner = np.matmul(affine_matrix,[[800],[0]])
    bottom_left_corner = np.matmul(affine_matrix,[[0],[640]])
    bottom_right_corner = np.matmul(affine_matrix,[[800],[640]])  
    
    x=[]
    y=[]
    x.append(top_left_corner[0][0])
    x.append(top_right_corner[0][0])
    x.append(bottom_left_corner[0][0])
    x.append(bottom_right_corner[0][0])
    x_max=max(x)
    x_min=min(x)
    
    y.append(top_left_corner[1][0])
    y.append(top_right_corner[1][0])
    y.append(bottom_left_corner[1][0])
    y.append(bottom_right_corner[1][0])
    y_max=max(y)
    y_min=min(y)
    
   
    
    wrap_width =x_max-x_min 
    wrap_height =y_max-y_min 
    
    return round(wrap_height),round(wrap_width)

def find_homography(affine_matrix,Rw,Rh):
    Wh,Ww = find_wrap_size(affine_matrix)
    Homography = np.matmul([[ affine_matrix[0][0], affine_matrix[0][1],0],
                                     [affine_matrix[1][0], affine_matrix[1][1],0],
                                      [0,0,1]],[[1,0,-Rw//2],
                                                [0,1,-Rh//2],
                                                [0,0,1]])
    Homography = np.matmul( [[1,0,Ww//2],
                          [0,1,Wh//2],
                          [0,0,1]],Homography)
    
    return Homography



def bilinear_interpolation(src_img,warp_img,homography_inv):
    for i in range (len(warp_img[0])): 
        for j in range (len(warp_img)): 
            temp = np.array (np.matmul(homography_inv,[[i],[j],[1]])) #invH ile wrap image kordinatları çarpıp,referans image de hangi noktolara denk geldiğini buluyoruz
            srcTemp = np.array([ [temp[0][0]/temp[2][0]],[temp[1][0]/temp[2][0]]] ) #src imgede hangi noktaya düştüğü 2D kordinat-normalization
            
            alfa,dontUse = math.modf(i)
            beta ,dontUse = math.modf(j)
            
            if (int(srcTemp[1][0]) >= len(src_img)-1 ):
                warp_img[j][i]= 0
            elif ( int(srcTemp[0][0]) >= len(src_img[0])-1):
                warp_img[j][i] = 0
            elif ( int(srcTemp[0][0]) <= 0):
                warp_img[j][i]= 0
            elif ( int(srcTemp[1][0]) <= 0):
                warp_img[j][i]= 0    
            else :
                warp_img[j][i] = (1-alfa)*(1-beta)* src_img[int(srcTemp[1][0])][int(srcTemp[0][0])]
                + alfa*(1-beta)*src_img[int(srcTemp[1][0])][int(srcTemp[0][0])+1]
                + (1-alfa)*beta* src_img[int(srcTemp[1][0])+1][int(srcTemp[0][0])]
                +alfa*beta*src_img[int(srcTemp[1][0])+1][int(srcTemp[0][0])+1]
            
            

    
    return warp_img




img = cv2.imread("img1.png",0)
affine_matrix=affine_deformation_matrix(0.5,math.radians(30),math.radians(50))
h,w= find_warp_size(affine_matrix)#0.5235987756 radius is 30 degree-- 0.872664626is 50 degree
wrap_image = zeros([int(h),int(w),3], dtype=np.uint8) #ilk verdiğin y ikincisi x
homography = find_homography(affine_matrix,len(img[0]),len(img))
homography_inverse = inv(homography)
output_image = bilinear_interpolation(img,wrap_image,homography_inverse)

cv2.imshow("OUTPUT",output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output.png", output_image)


# In[ ]:


"""
Comment-1

Multiplying left side matrix with affine matrix gives us resized warped image.
With multiplying right side matrix with affine matrix, the center of warp image is moved 
from its first location to its right locatin(also center of reference image and output image.)

Comment-2

We transform warp image to the reference image with inverse homography because we want 
to change intensity values of the image while we apply rotation,tilt and scale to 
the image. We need exact pixel coordinates of reference image (it could be float values,not integers.) 
that correspond pixel coordinates of warp image. We use these coordinates of referance image to calculate 
intenstiy of each pixel of warp image by applying bilinear interpolation.

Comment-3

While we transform the image with a matrix,coordinates values that we found are not
usually an integer like (2,3) or (8,6). They are usually floats like (3.7,6.2) etc.
So we look four other pixels that are closest to pixel coordinates that we found and 
we calculate new intensty values based on the weighted distances to this point
That's why we use bilinear interpolation to find intensity values.
Nearest-neighbor interpolation uses nearby reference pixel for the warped pixel value.
If we have categorical data like a map we can use nearest-neighbor method. But in our case bilinear
interpolation makes more sense because we have continious data and bilinear interpolation
use a weighted average of the four nearest pixel intesty value while nearest-neighbor interpolation
only use one nearst pixel intensty value.


"""


# In[ ]:





# In[ ]:




