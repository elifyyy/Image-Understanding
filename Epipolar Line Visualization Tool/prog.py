#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import math
from numpy.linalg import inv
from random import randint


# In[ ]:


img1 = cv2.imread("horse_0.JPG",1)
img2 = cv2.imread("horse_20.JPG",1)

#detect keypoints and compute their descriptors in both images.

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object with distance measurement cv2.NORM_HAMMING 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Initialize lists
list_kp1 = []
list_kp2 = []

# For each match
for mat in matches:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns 
    # y - rows
    # Get the coordinates
    #(x,y) tuples are in the col,row format,but i use row,col format that's why later i append them to list first 1.index next 0.index 
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt

    # Append to each list
    list_kp1.append((x1, y1))
    list_kp2.append((x2, y2))


# In[ ]:


ratio_arr = [] #to keep inlier ratios and find maximum
F_arr = [] # to keep calculated Fundamental matrices accordingly their inlier ratio
inliers = [] #to keep inliers found with related F matrix

N = 10000 # initial value
c=0 #count variable
while (c < N ):

    random_src = []
    random_dst = []

    for i in range (0,8):
        random_index = randint(0,len(list_kp1)-1)
        random_src.append([float(list_kp1[random_index][1]),float(list_kp1[random_index][0])])
        random_dst.append([float(list_kp2[random_index][1]),float(list_kp2[random_index][0])])

    #Normalizing points:

    def average_points(src_p,dst_p):
        total_row_src , total_col_src ,total_row_dst ,total_col_dst= 0,0,0,0
        for i in range (len(src_p)):
            total_row_src = total_row_src + float(src_p[i][1])
            total_col_src = total_col_src + float(src_p[i][0])
        for j in range (len(dst_p)):
            total_row_dst = total_row_dst + float(dst_p[j][1])
            total_col_dst = total_col_dst + float(dst_p[j][0])

        av_row_src = total_row_src / len(src_p)
        av_col_src = total_col_src / len(src_p)

        av_row_dst = total_row_dst / len(dst_p)
        av_col_dst = total_col_dst / len(dst_p)

        return (av_row_src,av_col_src),(av_row_dst,av_col_dst)


    src_av,dst_av = average_points(list_kp1,list_kp2)


    src_moved = []
    dst_moved = []

    for i in range (len(random_src)):
        temp_s =[]
        src_row_moved = float(random_src[i][0])-float(src_av[0])
        src_col_moved = float(random_src[i][1])-float (src_av[1])
        temp_s.append(src_row_moved)
        temp_s.append(src_col_moved)
        src_moved.append(temp_s)

    for i in range (len(random_dst)):
        temp_p =[]
        dst_row_moved = float(random_dst[i][0])-float(dst_av[0])
        dst_col_moved = float(random_dst[i][1])-float (dst_av[1])
        temp_p.append(dst_row_moved)
        temp_p.append(dst_col_moved)
        dst_moved.append(temp_p)   


    distance_src = 0;    
    for i in range (len(src_moved)):
        distance_src = distance_src + math.sqrt(src_moved[i][0]**2 + src_moved[i][1]**2)

    scale_src = (math.sqrt(2)) / (distance_src/len(src_moved))

    distance_dst = 0;    
    for i in range (len(dst_moved)):
        distance_dst = distance_dst + math.sqrt(dst_moved[i][0]**2 + dst_moved[i][1]**2)

    scale_dst = (math.sqrt(2)) / (distance_dst/len(dst_moved) )

    src_moved_scaled = []
    dst_moved_scaled = []

    for i in range (len(src_moved)):
        temp = []
        temp.append(scale_src*src_moved[i][0])
        temp.append(scale_src*src_moved[i][1])
        src_moved_scaled.append(temp)

    for i in range (len(dst_moved)):
        temp_arr = []
        temp_arr.append(scale_dst*dst_moved[i][0])
        temp_arr.append(scale_dst*dst_moved[i][1])
        dst_moved_scaled.append(temp_arr)


     ##Find matrix T
    T = np.array([[scale_src,0,-src_av[0]*scale_src],
                  [0,scale_src,-src_av[1]*scale_src],
                  [0,0,1]])


    ##Find matrix T_prime

    T_prime = np.array([[scale_dst,0,-dst_av[0]*scale_dst],
                        [0,scale_dst,-dst_av[1]*scale_dst],
                        [0,0,1]])

    ##Find matrix A


    A = []
    n = 8 #Normalized 8-point algorithm
    for i in range (0,n): 

        A.append( [dst_moved_scaled[i][0]*src_moved_scaled [i][0],dst_moved_scaled[i][0]*src_moved_scaled[i][1],dst_moved_scaled[i][0],dst_moved_scaled[i][1]*src_moved_scaled [i][0],dst_moved_scaled [i][1]*src_moved_scaled [i][1],dst_moved_scaled [i][1],src_moved_scaled [i][0],src_moved_scaled [i][1],1] )
    A = np.array(A)


    ##Find F_tilda


    u,s,v = np.linalg.svd(A, full_matrices=True)



    f1 =  v[8][0]
    f2 =  v[8][1]
    f3 =  v[8][2]
    f4 =  v[8][3]
    f5 =  v[8][4]
    f6 =  v[8][5]
    f7 =  v[8][6]
    f8 =  v[8][7]
    f9 =  v[8][8] 

    F_tilda = np.array([[f1,f2,f3],[f4,f5,f6],[f7,f8,f9]])


    ##Find F / unnormalize F_tilda

    F_temp = np.dot(inv(T_prime),F_tilda)
    F = np.array(np.dot(F_temp,T))
    
    #Forcing Rank-2 Constraint
    
    u,d,v = np.linalg.svd(F, full_matrices=True)
    vt = v.transpose()
    d = np.array([ [d[0],0,0], [0,d[1],0],[0,0,0]])
    F =np.dot(u,d)
    F =np.dot(F,vt)
    
    F_arr.append(F)
    
    
   #find epipolar lines and count inliers
    epipolar_lines = []
    for i in range (len(list_kp1)):
        line = np.array(np.dot(F,np.array([  [ float(list_kp1[i][1]) ] ,[float (list_kp1[i][0])],[1] ])))
        epipolar_lines.append (line)
     
    number_of_inliers = 0;
    inliers_temp = []
    for i in range (len(list_kp2)):
        distance =( abs(epipolar_lines[i][0]* float(list_kp2[i][1])+epipolar_lines[i][1]*float(list_kp2[i][0])+epipolar_lines[i][2])) / math.sqrt(epipolar_lines[i][0]**2 + epipolar_lines[i][1]**2)
        if(distance <= 3):
            number_of_inliers = number_of_inliers + 1 
            inliers_temp.append([list_kp1[i],list_kp2[i]])
    
    inlier_ratio = (number_of_inliers/len(epipolar_lines))*100
    ratio_arr.append(inlier_ratio)         
    inliers.append(inliers_temp)
    
    
    w = (number_of_inliers/len(list_kp1))**8
    s = 8
    N_updated = N
    if((math.log10(1-(w**s)))!=0):
        N_updated = int(  (math.log10(0.01)) / (math.log10(1-(w**s)))) 
    if(N_updated < N): #update N if new N is smaller than previous one, since we want to maximum number of inliers
        N = N_updated
    c = c+1
    
sorted_ratio_arr = sorted(ratio_arr)
max_ratio = sorted_ratio_arr[len(sorted_ratio_arr)-1] 
index = 0 #index of the maximum number in the ratio array is also index of F best in F_arr
          # and it is also index of inliers computed with F best in inliers list,because adding to lists was in the same order
for i in range(len(ratio_arr)):
    if(ratio_arr[i] == max_ratio):
        index = i

F_best = F_arr[index] 
inliers_computed_with_F_best = inliers[index]


# In[ ]:


#compute F again with inliers by using 8 point algorithm,update the inliers,
#do it until num of inliers converges

is_converge = 0 # if the difference between last 10 number_of_inliers 
                # is smaller than 5 , then it converges and break the loop
while(True):
    src_points = [] 
    dst_points = [] 
    number_of_inliers = len(inliers_computed_with_F_best) #loop until converges
    
    for i in range(len(inliers_computed_with_F_best)):
        src_points.append([float(inliers_computed_with_F_best[i][0][1]),float(inliers_computed_with_F_best[i][0][0])])
        dst_points.append([float(inliers_computed_with_F_best[i][1][1]),float(inliers_computed_with_F_best[i][1][0])])    


    def average_points(src_p,dst_p):
            total_row_src , total_col_src ,total_row_dst ,total_col_dst= 0,0,0,0
            for i in range (len(src_p)):
                total_row_src = total_row_src + float(src_p[i][0])
                total_col_src = total_col_src + float(src_p[i][1])
            for j in range (len(dst_p)):
                total_row_dst = total_row_dst + float(dst_p[j][0])
                total_col_dst = total_col_dst + float(dst_p[j][1])

            av_row_src = total_row_src / len(src_p)
            av_col_src = total_col_src / len(src_p)

            av_row_dst = total_row_dst / len(dst_p)
            av_col_dst = total_col_dst / len(dst_p)

            return (av_row_src,av_col_src),(av_row_dst,av_col_dst)


    src_av,dst_av = average_points(src_points,dst_points)


    src_moved = []
    dst_moved = []

    for i in range (len(src_points)):
        temp_s =[]
        src_row_moved = float(src_points[i][0])-float(src_av[0])
        src_col_moved = float(src_points[i][1])-float (src_av[1])
        temp_s.append(src_row_moved)
        temp_s.append(src_col_moved)
        src_moved.append(temp_s)

    for i in range (len(dst_points)):
        temp_p =[]
        dst_row_moved = float(dst_points[i][0])-float(dst_av[0])
        dst_col_moved = float(dst_points[i][1])-float (dst_av[1])
        temp_p.append(dst_row_moved)
        temp_p.append(dst_col_moved)
        dst_moved.append(temp_p)   


    distance_src = 0;    
    for i in range (len(src_moved)):
        distance_src = distance_src + math.sqrt(src_moved[i][0]**2 + src_moved[i][1]**2)

        scale_src = (math.sqrt(2)) / (distance_src/len(src_moved))

    distance_dst = 0;    
    for i in range (len(dst_moved)):
        distance_dst = distance_dst + math.sqrt(dst_moved[i][0]**2 + dst_moved[i][1]**2)

        scale_dst = (math.sqrt(2)) / (distance_dst/len(dst_moved) )

        src_moved_scaled = []
        dst_moved_scaled = []

    for i in range (len(src_moved)):
        temp = []
        temp.append(scale_src*src_moved[i][0])
        temp.append(scale_src*src_moved[i][1])
        src_moved_scaled.append(temp)

    for i in range (len(dst_moved)):
        temp_arr = []
        temp_arr.append(scale_dst*dst_moved[i][0])
        temp_arr.append(scale_dst*dst_moved[i][1])
        dst_moved_scaled.append(temp_arr)


         ##Find matrix T
    T = np.array([[scale_src,0,-src_av[0]*scale_src],
                      [0,scale_src,-src_av[1]*scale_src],
                      [0,0,1]])


        ##Find matrix T_prime

    T_prime = np.array([[scale_dst,0,-dst_av[0]*scale_dst],
                            [0,scale_dst,-dst_av[1]*scale_dst],
                            [0,0,1]])

        ##Find matrix A

    A = []
    n = len(inliers_computed_with_F_best) #Normalized 8-point algorithm at least 8 point
    for i in range (0,n): 
        A.append( [dst_moved_scaled[i][0]*src_moved_scaled [i][0],dst_moved_scaled[i][0]*src_moved_scaled[i][1],dst_moved_scaled[i][0],dst_moved_scaled[i][1]*src_moved_scaled [i][0],dst_moved_scaled [i][1]*src_moved_scaled [i][1],dst_moved_scaled [i][1],src_moved_scaled [i][0],src_moved_scaled [i][1],1] )
    A = np.array(A)

    ##Find F_tilda

    u,s,v = np.linalg.svd(A, full_matrices=True)

    f1 =  v[8][0]
    f2 =  v[8][1]
    f3 =  v[8][2]
    f4 =  v[8][3]
    f5 =  v[8][4]
    f6 =  v[8][5]
    f7 =  v[8][6]
    f8 =  v[8][7]
    f9 =  v[8][8] 

    F_tilda = np.array([[f1,f2,f3],[f4,f5,f6],[f7,f8,f9]])


    ##Find F_final / unnormalize F_tilda

    F_temp = np.dot(inv(T_prime),F_tilda)
    F_final = np.array(np.dot(F_temp,T))
    
    #Forcing Rank-2 Constraint 
    u,d,v = np.linalg.svd(F_final, full_matrices=True)
    vt = v.transpose()
    d = np.array([ [d[0],0,0], [0,d[1],0],[0,0,0]])
    F_final =np.dot(u,d)
    F_final =np.dot(F_final,vt)



    to_compare = number_of_inliers # to compare new num of inliers with previous one
    
    epipolar_lines = []
    for i in range (len(src_points)):
        line = np.array(np.dot(F_final,np.array([[float(src_points[i][0])] ,[float (src_points[i][1])],[1] ])))
        epipolar_lines.append (line)

    inliers_count = 0
    new_inliers = []
    for i in range (len(dst_points)):
        distance =( abs(epipolar_lines[i][0]* float(dst_points[i][0]) + epipolar_lines[i][1]*float(dst_points[i][1])+epipolar_lines[i][2])) / math.sqrt(epipolar_lines[i][0]**2 + epipolar_lines[i][1]**2)
        if(distance <= 3):
            inliers_count = inliers_count + 1 
            new_inliers.append([src_points[i],dst_points[i]]) #update the inliers
    
    if(len(new_inliers) >= len(inliers_computed_with_F_best)):#update inliers only when num of new inliers are greater or equal to previous one,since we want maximum num of inliers
        inliers_computed_with_F_best = new_inliers
        number_of_inliers = len(new_inliers)
    
    if(abs(to_compare - number_of_inliers <= 5)): 
        is_converge = is_converge + 1
    else:
        is_converge = 0 # while is_converge is being increased if a distinctly 
                        #different number appears,is_converges set to 0 and 
                        #counting start from beging untill it reaches 10.
    
    if(is_converge == 10): #if the difference between last 10 number_of_inliers 
                           # is smaller than 5 , then it converges and break the loop
        break


# In[ ]:


h,w,dont_use = img1.shape
def mouse_event(event,x,y,flags,param):
    if (event== cv2.EVENT_LBUTTONDBLCLK and 0<x<w and 0<y<h):
        cv2.circle(vis,(x,y),5,(0,0,255),-1)
        line = np.array(np.dot(F_final,np.array([[x],[y],[1] ]))) # line equation that we want to draw on second image
        #find end points of the line to draw it.
        #x1= 0 find y1
        x1 = 0
        y1 = -(line[2][0]/line[1][0]) #(x1,y1) point is the point where line cross y axis
        #y2=0 find x2
        y2 = 0
        x2 = -(line[2][0]/line[0][0]) #(x2,y2)  is the point where line cross x axis
        cv2.line(vis, (int(x1)+w,int(y1)), (int(x2)+w,int(y2)), (0, 0, 255), 5)
        #i add weight of the first image to x coordinates because i concatenate img1 and img2
        #since vis = img1+img2 , x coordinates of the img2 start from width of img1 
        #but y coordinates is same for both images. it is like orijin of the image2 is (width of img1,0)

vis = np.concatenate((img1, img2), axis=1)
cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse_event)

while(1):
    cv2.imshow('image',vis)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()


# In[ ]:


"""
COMMENT: Can 4 correspondences are enough to estimate Funda-
mental matrix like homograph estimation? Why? OR Why not?

    4 correspondences are not enough to estimate Fundamental matrix.Because
    
    [ x'x   x'y   x'  y'x   y'y   y'   x   y   1] [f1 
                                                   f2    
                                                   f3    ====> Af
                                                   .
                                                   .
                                                   .
                                                   f9]
                                                   
   This is a scalar equation and it's degree of freedom is one. Since we can only know
   the fundamental matrix up to scale, we need eight of these constrain to find the
   fundamental matrix.
"""


# In[ ]:




