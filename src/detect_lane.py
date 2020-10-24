import cv2
import numpy as np
import pickle
from binary_tuner import mag_sobel,abs_sobel_mag, dir_sobel, get_sobel_mag, hls_thresh,color_thresh
from perspective_transformations import perspective_transform, inv_perspective_transform, undistort_img, get_perspective_mtx,\
                        get_inv_perspective_mtx, get_distortion_measure
import time
from moviepy.editor import VideoFileClip


M = get_perspective_mtx()
Minv = get_inv_perspective_mtx()
ret, mtx, dist, rvecs, tvecs = get_distortion_measure()
left_coeff=np.zeros((3,1))
right_coeff=np.zeros((3,1))
first_run=True

def get_binary(img):
    #sobelx,actualsobelx=abs_sobel_mag(img,'x',(40,153))
    #cv2.imshow('sobelx',cv2.resize(sobelx,(sobelx.shape[1]//3,sobelx.shape[0]//3)))
    #sobely,actualsobely=abs_sobel_mag(img,'y',(49,211))
    #cv2.imshow('sobely', cv2.resize(sobely,(sobelx.shape[1]//3,sobelx.shape[0]//3)))
    #mag_sob=mag_sobel(img,(40,206))
    mag_sob=mag_sobel(img,(64,255))
    #cv2.imshow('mag_sob', cv2.resize(mag_sob,(sobelx.shape[1]//3,sobelx.shape[0]//3)))
    #hue_bin=hls_thresh(img,channel='h',(15,90))
    #cv2.imshow('hue', cv2.resize(hue_bin,(sobelx.shape[1]//3,sobelx.shape[0]//3)))
    #dir_sob=dir_sobel(img,(47,144))
    color_bin=color_thresh(img)

    #cv2.imshow('col_thr', cv2.resize(color_bin, (sobelx.shape[1] // 3, sobelx.shape[0] // 3)))
    bin_img=np.zeros_like(mag_sob)
    #bin_img[((sobelx==255) & (sobely==255))& ((dir_sob==255) | (mag_sob==255))]=255
    #bin_img[((sobelx == 255) & (sobely == 255)& (mag_sob == 255) ) | ((color_bin==255))   ] = 255
    bin_img[(mag_sob == 255)| ((color_bin == 255))] = 255
    return bin_img


def get_histogram(img):
    half_img=img[int(img.shape[0]/2):,:]
    histogram=np.sum(half_img,axis=0)
    return histogram


def process_video():
    cap=cv2.VideoCapture('./../project_video.mp4')
    ret,mtx,dist,rvecs,tvecs=get_distortion_measure()
    while(cv2.waitKey(10)!=ord('q')):
        ret,img=cap.read()
        process_img(img)



def get_video():
    videoloc='./../project_video.mp4'
    white_output='previousfit_video.mp4'

    clip1 = VideoFileClip("./../project_video.mp4")
    white_clip = clip1.fl_image(process_img)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


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
        cv2.rectangle(img, (int(hist.shape[0]/2)+right_indices[window] - int(window_size / 2), shape[0]-window*window_size), \
                     (int(hist.shape[0]/2)+right_indices[window] + int(window_size / 2),shape[0]-(window+1)*window_size), (255), 3)
    cv2.imshow('img',img)
    cv2.waitKey()

def sliding_window(img):
    global left_coeff,right_coeff
    print("Searching for the first time")
    left_indices=[]   # Save white pixels on left half of image
    right_indices=[]  # Save white pixels on right half of image
    margin=100
    nwindows=9
    shape = img.shape
    window_height=shape[0]//nwindows

    minpix=10

    hist=get_histogram(img)
    midpoint = hist.shape[0] // 2
    leftx_base=np.argmax(hist[:midpoint])
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

        cv2.rectangle(imgcolor,(left_x_low,y_low),(left_x_high,y_high),(255,255,255),2)
        cv2.rectangle(imgcolor, (right_x_low, y_low), (right_x_high, y_high), (255, 0, 0), 2)


        # Storing all nonzeros indices that lie within the window defined.
        left_nonzeros_in_window=((nonzerox>=left_x_low) & (nonzerox < left_x_high) & (nonzeroy>=y_low)
                                            & (nonzeroy < y_high)).nonzero()[0]

        right_nonzeros_in_window = ((nonzerox >= right_x_low) & (nonzerox <= right_x_high) & (nonzeroy >= y_low)
                                           & (nonzeroy < y_high)).nonzero()[0]

        if(len(left_nonzeros_in_window)>minpix):
            leftx_current= np.int(np.mean(nonzerox[left_nonzeros_in_window]))
        if(len(right_nonzeros_in_window)>minpix):
            rightx_current= np.int(np.mean(nonzerox[right_nonzeros_in_window]))

        left_indices.append(left_nonzeros_in_window)
        right_indices.append(right_nonzeros_in_window)

    left_indices=np.concatenate(left_indices)
    right_indices=np.concatenate(right_indices)

    leftx = nonzerox[left_indices]
    lefty = nonzeroy[left_indices]
    rightx = nonzerox[right_indices]
    righty = nonzeroy[right_indices]

    #cv2.imshow('test',imgcolor)
    #cv2.waitKey()
    return leftx,lefty,rightx,righty,imgcolor

# This function takes input from one image which has been fit with curve(sliding_window-->fit_line-->search_around_poly-->fit_line-->search_around_poly...)
def search_around_poly(img):
    print("Searching around previous fit")
    global left_coeff,right_coeff
    print("left coeff:",left_coeff)
    print("right_coeff:",right_coeff)
    nonzero=img.nonzero()
    nonzerox=nonzero[1]
    nonzeroy=nonzero[0]
    margin = 100

    left_lane_inds=((nonzerox > nonzeroy**2*left_coeff[0] + nonzeroy*left_coeff[1]+left_coeff[2]-margin)&\
                   (nonzerox < nonzeroy**2*left_coeff[0] + nonzeroy*left_coeff[1]+left_coeff[2]+margin)).nonzero()[0]
    right_lane_inds=((nonzerox > nonzeroy**2*right_coeff[0] + nonzeroy*right_coeff[1]+right_coeff[2]-margin)&\
                   (nonzerox < nonzeroy**2*right_coeff[0] + nonzeroy*right_coeff[1]+right_coeff[2]+margin)).nonzero()[0]

    leftx=nonzerox[left_lane_inds]
    lefty=nonzeroy[left_lane_inds]
    rightx=nonzerox[right_lane_inds]
    righty=nonzeroy[right_lane_inds]
    # Now send it to fit_line which will give new coefficients of fit. Then use it again for searching lanes in the next frame.
    return leftx,lefty,rightx,righty,img


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
    cv2.imshow('left lanes leftright',both_margins)
    cv2.waitKey()

    return both_margins


def fit_line(leftx,lefty,rightx,righty,img):
    #imgcolor=np.dstack((img,img,img))
    global left_coeff, right_coeff
    print(lefty)
    print(rightx)
    left_coeff=np.polyfit(lefty,leftx,2)
    right_coeff=np.polyfit(righty,rightx,2)


    lefty_arr=np.linspace(0,img.shape[1]-1,img.shape[1])
    leftx_arr=left_coeff[0]* (lefty_arr**2) + left_coeff[1]*lefty_arr + left_coeff[2]
    left_pairs=[(leftx_arr[i],lefty_arr[i]) for i in range(len(leftx_arr))]

    righty_arr = np.linspace(0, img.shape[1]-1, img.shape[1])
    rightx_arr = right_coeff[0] * (righty_arr ** 2) + right_coeff[1] * righty_arr + right_coeff[2]
    right_pairs = [(rightx_arr[i], righty_arr[i]) for i in range(len(rightx_arr))]

    left_pairs=np.array(left_pairs).astype(np.int32)
    left_pairs=left_pairs.reshape(-1,1,2)
    right_pairs=np.array(right_pairs).astype(np.int32)
    right_pairs=right_pairs.reshape(-1,1,2)

    cv2.polylines(img,[left_pairs],False,(0,255,0),5)
    cv2.polylines(img,[right_pairs],False,(0,255,0),5)
    R_left=get_rad_curvature(left_coeff[0],left_coeff[1],img.shape[0])
    R_right=get_rad_curvature(right_coeff[0],right_coeff[1],img.shape[0])

    return img,R_left,R_right,left_pairs,right_pairs


def get_rad_curvature(A,B,y):
    R=(1+(2*A*y + B)**2)**(3/2)/(2*A)
    return R




def fill_lane(left_pairs,right_pairs,img,color=(0,255,0)):
    pts = np.array([left_pairs, np.flipud(right_pairs)])
    pts = np.concatenate(pts)
    cv2.fillPoly(img, [pts], color)
    #cv2.imshow('filllane',img)
    return img

def process_img(img):
    global first_run
    cv2.imshow('actual input',img)
    #img=cv2.imread('./../test_images/test2.jpg')
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    undist=undistort_img(img,mtx,dist)
    cv2.imshow('undistorted',undist)
    bin_img=get_binary(undist)
    blank_img = np.zeros_like(bin_img)
    blank_color=np.dstack((blank_img,blank_img,blank_img))
    warped=perspective_transform(bin_img,M)
    box = undist.copy()
    if(first_run):
        leftx,lefty,rightx,righty,imgbox = sliding_window(warped)
    else:
        search_margin=draw_current_lanes(warped)
        cv2.imshow('search',search_margin)
        leftx,lefty,rightx,righty,imgbox = search_around_poly(warped)

    imgcolor,R_left,R_right,left_pairs,right_pairs=fit_line(leftx,lefty,rightx,righty,imgbox)
    Rad_curve=(R_left+R_right)/2
    filled=fill_lane(left_pairs,right_pairs,blank_color)
    cv2.imshow("this is filled result",filled)
    filled_rev_warp=inv_perspective_transform(filled,Minv)
    final = cv2.addWeighted(box, 1, filled_rev_warp, 0.3, 0)
    if(first_run==False):
        search_margin_persp = inv_perspective_transform(search_margin, Minv)
        cv2.imshow('search_margin',search_margin_persp)
        final=cv2.addWeighted(final,1,search_margin_persp,0.4,0)
    cv2.putText(final, "Radius of curvature:{:.2f}".format(Rad_curve), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    final=cv2.cvtColor(final,cv2.COLOR_BGR2RGB)
    cv2.imshow('filled223', final)
    first_run=False
    cv2.waitKey()
    return final


if __name__=='__main__':
     process_video()
     # frame=cv2.imread('./new_tests/test_img25.jpg')
     # final=process_img(frame)
     #get_video()