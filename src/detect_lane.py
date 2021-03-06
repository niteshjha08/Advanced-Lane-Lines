import cv2
import numpy as np
import pickle
from binary_tuner import mag_sobel,abs_sobel_mag, dir_sobel, get_sobel_mag, hls_thresh,color_thresh
from perspective_transformations import perspective_transform, inv_perspective_transform, undistort_img, get_perspective_mtx,\
                        get_inv_perspective_mtx, get_distortion_measure
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Loading parameters stored in binary files
M = get_perspective_mtx()
Minv = get_inv_perspective_mtx()
ret, mtx, dist, rvecs, tvecs = get_distortion_measure()
# Initialising fit coefficients
left_coeff=np.zeros((3,1))
right_coeff=np.zeros((3,1))
# Flag to check if frame is the first in a video
first_run=True


# Generate binary images using Sobel magnitude and HLS thresholding
def get_binary(img):
    mag_sob=mag_sobel(img,(64,255))
    color_bin=color_thresh(img)
    bin_img=np.zeros_like(mag_sob)
    bin_img[(mag_sob == 255)| ((color_bin == 255))] = 255
    return bin_img


# Obtaining histogram of lower half of binary image for lane base location
def get_histogram(img):
    half_img=img[int(img.shape[0]/2):,:]
    histogram=np.sum(half_img,axis=0)
    return histogram


# Function to run whole pipeline
def process_video():
    cap=cv2.VideoCapture('./../project_video.mp4')
    while(cv2.waitKey(10)!=ord('q')):
        ret,img=cap.read()
        process_img(img)


# Generate video file as annotated output of raw video
def get_video():
    white_output='project_video_final_2ndtry.mp4'
    clip1 = VideoFileClip("./../project_video.mp4")
    white_clip = clip1.fl_image(process_img)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


# Visualisation for lane finding using histograms of images as successively they get cropped vertically
# This approach was tested but not used in the current pipeline.
def draw_boxes(img):
    margin=100
    nwindows=9
    window_size=80
    left_indices=[]
    right_indices=[]
    shape=img.shape
    copy=img.copy()
    for window in range(nwindows):
        copy=copy[:shape[0]-window*window_size,:]
        hist=get_histogram(copy)
        hist_left=hist[:int(hist.shape[0]/2)]
        hist_right=hist[int(hist.shape[0]/2):]
        left_max=np.argmax(hist_left)
        right_max=np.argmax(hist_right)
        left_indices.append(left_max)
        right_indices.append(right_max)

    for window in range(nwindows):
        cv2.rectangle(img,(left_indices[window]-int(window_size/2),shape[0]-window*window_size),\
                      (left_indices[window]+int(window_size/2),shape[0]-(window+1)*window_size),255,3)
        cv2.rectangle(img, (int(hist.shape[0]/2)+right_indices[window] - int(window_size / 2), shape[0]-window * window_size), \
                     (int(hist.shape[0]/2)+right_indices[window] + int(window_size / 2),shape[0]-(window+1)*window_size), 255, 3)
    cv2.imshow('img',img)
    cv2.waitKey()


# Generate search locations using windows which slide horizontally as it moves up the image
def sliding_window(img):
    global left_coeff,right_coeff
    left_indices=[]   # Save white pixels on left half of image
    right_indices=[]  # Save white pixels on right half of image
    # +\- 100 pixels are checked for non zero pixels
    margin=100
    # Image is divided into 9 parts along the vertical
    nwindows=9

    shape = img.shape
    window_height=shape[0]//nwindows
    minpix=50
    hist=get_histogram(img)
    midpoint = hist.shape[0] // 2
    # Find location of left lane base
    leftx_base=np.argmax(hist[:midpoint])
    # Find location of the right lane base
    rightx_base=midpoint + np.argmax(hist[midpoint:])

    imgcolor=np.dstack((img,img,img))
    # These variables will contain locations of all non-zero pixels
    nonzero=img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    leftx_current=leftx_base
    rightx_current=rightx_base

    for window in range(nwindows):
        # Defining window points within which pixels will be counted
        left_x_low = leftx_current - margin
        left_x_high = leftx_current + margin
        y_high=shape[0] - window * window_height
        y_low = shape[0] - (window + 1) * window_height
        right_x_low = rightx_current - margin
        right_x_high = rightx_current + margin
        # Visualising search windows
        cv2.rectangle(imgcolor,(left_x_low,y_low),(left_x_high,y_high),(255,255,255),2)
        cv2.rectangle(imgcolor, (right_x_low, y_low), (right_x_high, y_high), (255, 0, 0), 2)

        # Storing all nonzeros indices that lie within the window defined.
        # It stores the index of array which contains index of nonzero pixels
        left_nonzeros_in_window=((nonzerox>=left_x_low) & (nonzerox < left_x_high) & (nonzeroy>=y_low)
                                            & (nonzeroy < y_high)).nonzero()[0]

        right_nonzeros_in_window = ((nonzerox >= right_x_low) & (nonzerox <= right_x_high) & (nonzeroy >= y_low)
                                           & (nonzeroy < y_high)).nonzero()[0]
        # Check if sufficient pixels are found to shift window location, else it stays the same
        if(len(left_nonzeros_in_window)>minpix):
            leftx_current= np.int(np.mean(nonzerox[left_nonzeros_in_window]))
        if(len(right_nonzeros_in_window)>minpix):
            rightx_current= np.int(np.mean(nonzerox[right_nonzeros_in_window]))

        left_indices.append(left_nonzeros_in_window)
        right_indices.append(right_nonzeros_in_window)

    left_indices=np.concatenate(left_indices)
    right_indices=np.concatenate(right_indices)
    # return values of pixel location which are non zero and also lie within the search window margin.
    leftx = nonzerox[left_indices]
    lefty = nonzeroy[left_indices]
    rightx = nonzerox[right_indices]
    righty = nonzeroy[right_indices]
    return leftx,lefty,rightx,righty,imgcolor

# Takes img and searches around the polynomial which was fit in the previous fit, thus eliminating histogram based sliding window search
def search_around_poly(img):

    global left_coeff,right_coeff
    nonzero=img.nonzero()
    nonzerox=nonzero[1]
    nonzeroy=nonzero[0]
    # Defines margin width within which search will happen around the previous fit
    margin = 100
    left_lane_inds=((nonzerox > nonzeroy**2*left_coeff[0] + nonzeroy*left_coeff[1]+left_coeff[2]-margin)&\
                   (nonzerox < nonzeroy**2*left_coeff[0] + nonzeroy*left_coeff[1]+left_coeff[2]+margin)).nonzero()[0]
    right_lane_inds=((nonzerox > nonzeroy**2*right_coeff[0] + nonzeroy*right_coeff[1]+right_coeff[2]-margin)&\
                   (nonzerox < nonzeroy**2*right_coeff[0] + nonzeroy*right_coeff[1]+right_coeff[2]+margin)).nonzero()[0]
    # return values in this margin which had non zero pixels
    leftx=nonzerox[left_lane_inds]
    lefty=nonzeroy[left_lane_inds]
    rightx=nonzerox[right_lane_inds]
    righty=nonzeroy[right_lane_inds]
    return leftx,lefty,rightx,righty,img


# Visualise the search margins defined above, (in search_around_poly())
def draw_current_lanes(img):
    global left_coeff,right_coeff
    ys=np.linspace(0,img.shape[0]-1,img.shape[0])
    xleft=ys**2*left_coeff[0]+ys*left_coeff[1]+left_coeff[2]
    xright=ys**2*right_coeff[0]+ys*right_coeff[1]+right_coeff[2]
    left_pairs = [(xleft[i], ys[i]) for i in range(len(xleft))]
    right_pairs = [(xright[i], ys[i]) for i in range(len(xright))]
    left_pairs = np.array(left_pairs).astype(np.int32)
    left_pairs = left_pairs.reshape(-1, 1, 2)
    right_pairs = np.array(right_pairs).astype(np.int32)
    right_pairs = right_pairs.reshape(-1, 1, 2)

    blank = np.zeros_like(img)
    blank_color = np.dstack((blank, blank, blank))
    left_lmargin=np.copy(left_pairs).T
    left_rmargin = np.copy(left_pairs).T
    left_lmargin[0]=left_lmargin[0]-100
    left_lmargin = left_lmargin.T
    left_rmargin[0]=left_rmargin[0]+100
    left_rmargin=left_rmargin.T

    right_lmargin = np.copy(right_pairs).T
    right_rmargin = np.copy(right_pairs).T
    right_lmargin[0] = right_lmargin[0] - 100
    right_lmargin = right_lmargin.T
    right_rmargin[0] = right_rmargin[0] + 100
    right_rmargin = right_rmargin.T
    left_margin=fill_lane(left_lmargin,left_rmargin,blank_color,(0,0,255))
    both_margins=fill_lane(right_lmargin,right_rmargin,left_margin,(0,0,255))
    return both_margins


# Fits 2 degree polynomial using (x,y) pairs of lane pixels
def fit_line(leftx,lefty,rightx,righty,img):
    global left_coeff, right_coeff
    left_coeff=np.polyfit(lefty,leftx,2)
    right_coeff=np.polyfit(righty,rightx,2)

    lefty_arr=np.linspace(0,img.shape[1]-1,img.shape[1])
    leftx_arr=left_coeff[0]* (lefty_arr**2) + left_coeff[1]*lefty_arr + left_coeff[2]
    left_pairs=[(leftx_arr[i],lefty_arr[i]) for i in range(len(leftx_arr))]

    righty_arr = np.linspace(0, img.shape[1]-1, img.shape[1])
    rightx_arr = right_coeff[0] * (righty_arr ** 2) + right_coeff[1] * righty_arr + right_coeff[2]
    right_pairs = [(rightx_arr[i], righty_arr[i]) for i in range(len(rightx_arr))]
    # Finding lane center and vehicle offset
    lane_center=((left_coeff[0]* (700**2) + left_coeff[1]*700 + left_coeff[2])+(right_coeff[0] * (700 ** 2) + right_coeff[1] * 700 + right_coeff[2]))/2
    img_center=img.shape[1]/2
    lane_offset=img_center-lane_center

    left_pairs=np.array(left_pairs).astype(np.int32)
    left_pairs=left_pairs.reshape(-1,1,2)
    right_pairs=np.array(right_pairs).astype(np.int32)
    right_pairs=right_pairs.reshape(-1,1,2)
     # Drawing fit polynomial on the lanes
    cv2.polylines(img,[left_pairs],False,(0,255,0),5)
    cv2.polylines(img,[right_pairs],False,(0,255,0),5)
    R_left, R_right=radius_curvature(leftx,lefty,rightx,righty)
    return img,R_left,R_right,left_pairs,right_pairs,lane_offset


# Calculate radius of curvature of lane
def radius_curvature(leftx,lefty,rightx,righty):
    # Scale factors:
    # 3.7m width = 700 px
    # 30m length = 720 px
    scalex= 3.7/700
    scaley=30/720
    # closest 'y' to the car, lowermost point of the image
    y=719

    left_fit_coeff=np.polyfit(lefty*scaley,leftx*scalex,2)
    right_fit_coeff=np.polyfit(righty*scaley,rightx*scalex,2)
    R_left=(1+(2*left_fit_coeff[0]*y*scaley + left_fit_coeff[1])**2)**(3/2)/(2*left_fit_coeff[0])
    R_right=(1+(2*right_fit_coeff[0]*y*scaley + right_fit_coeff[1])**2)**(3/2)/(2*right_fit_coeff[0])
    return R_left,R_right


# Fill the lane found using fit_line() in the given color
def fill_lane(left_pairs,right_pairs,img,color=(0,255,0)):
    pts = np.array([left_pairs, np.flipud(right_pairs)])
    pts = np.concatenate(pts)
    cv2.fillPoly(img, [pts], color)
    return img


# Pipeline for a frame of a video
def process_img(img):
    global first_run
    #img=cv2.imread('./../test_images/test1.jpg)

    #As moviepy Videoclip function uses RGB scheme, and image is processed in BGR, conversion takes place.
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    # Undistortion
    undist=undistort_img(img,mtx,dist)
    # Binary image generation
    bin_img=get_binary(undist)

    blank_img = np.zeros_like(bin_img)
    blank_color=np.dstack((blank_img,blank_img,blank_img))
    # Perspective transformation of binary image
    warped=perspective_transform(bin_img,M)

    box = undist.copy()
    # For only the first frame of a vide, histogram based sliding search takes place, subsequently, previous fits are used to search
    # for new lane lines
    if(first_run):
        leftx,lefty,rightx,righty,imgbox = sliding_window(warped)

    else:
        #Visualising the search margin from previous fit
        search_margin = draw_current_lanes(warped)
        # Searching in margin for new fit
        leftx, lefty, rightx, righty, imgbox = search_around_poly(warped)

    #Fitting 2 degree polynomial
    imgcolor,R_left,R_right,left_pairs,right_pairs, lane_offset=fit_line(leftx,lefty,rightx,righty,imgbox)
    # Averaging left and right lane radius of curvature
    Rad_curve=(R_left+R_right)/2
    # Filling the detected lane in green(default)
    filled=fill_lane(left_pairs,right_pairs,blank_color)
    # Projecting the image back onto the original perspective
    filled_rev_warp=inv_perspective_transform(filled,Minv)
    # Weighted sum of this with undistorted image for transparency
    final = cv2.addWeighted(box, 1, filled_rev_warp, 0.3, 0)

    # Visualising search_around_poly(). Can comment whole block if not needed (line 297-line 306)
    if(first_run==False):
        warped_color=np.dstack((warped,warped,warped))
        bin_color=np.dstack((bin_img,bin_img,bin_img))
        print(search_margin.shape)
        print(warped.shape)
        search_margin_persp = inv_perspective_transform(search_margin, Minv)
        cv2.imshow('search_margin',search_margin_persp)
        marginimg=cv2.addWeighted(warped_color,1,search_margin,0.4,0)
        maginpersp=cv2.addWeighted(bin_color,1,search_margin_persp,0.3,0)
        cv2.imshow('marginimg',marginimg)

    # Converting pixel to m
    lane_offset= lane_offset * 3.7/685
    # For annotated output
    if(lane_offset<0):
        offset_direction="left"
    else:
        offset_direction="right"
    if(Rad_curve<0):
        sign="(-)"
    else:
        sign=""
    Rad_curve=np.abs(Rad_curve)
    lane_offset=np.abs(lane_offset)
    cv2.putText(final, "Radius of curvature={0}{1:.2f}m".format(sign,Rad_curve), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(final, "Vehicle is {0:.2f}m {1} of center".format(lane_offset,offset_direction), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    final=cv2.cvtColor(final,cv2.COLOR_BGR2RGB)
    cv2.imshow('final annotated image', final)
    # To make subsequent frames use search_around_poly() instead of sliding_windows()
    first_run=False
    return final


if __name__=='__main__':
    # To only display video frame by frame. Also use cv2.waitKey()  in process_img() to pause frames
     process_video()
    # Use this to generate video file. Edit source and output names first
     #get_video()
