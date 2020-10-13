import cv2
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
# cap=cv2.VideoCapture('project_video.mp4')
with open('calib_param.pickle','rb') as f:
    [ret,mtx,dist,rvecs,tvecs]=pickle.load(f)

def nothing(x):
    a=2

def abs_sobel_mag(img,orient='x',threshold=(0,255),ksize=3):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if(orient=='x'):
        x=1
        y=0
    else:
        x=0
        y=1
    sobel=cv2.Sobel(gray,cv2.CV_64F,x,y,ksize=ksize)
    abs_sobel=np.absolute(sobel)
    norm_sobel=np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_op=np.zeros_like(gray)
    binary_op[((sobel>threshold[0]) & (sobel<threshold[1]))]=255
    return binary_op

def mag_sobel(img,thresh=(0,255),ksize=3):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobelx=cv2.Sobel(gray,cv2.CV_64F, 1,0)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1)
    abs_sobel_xy=np.sqrt(sobelx**2+sobely**2)
    norm_sobel=(255*abs_sobel_xy/np.max(abs_sobel_xy)).astype(np.uint8)
    binary_op=np.zeros_like(gray)
    binary_op[(norm_sobel>thresh[0]) & (norm_sobel<thresh[1])]=255
    return binary_op

def dir_sobel(img,thresh=(0,255),ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_sobel= np.arctan2(abs_sobely,abs_sobelx)
    norm_sobel = (255 * dir_sobel / np.max(dir_sobel)).astype(np.uint8)
    binary_op = np.zeros_like(gray)
    binary_op[(norm_sobel > thresh[0]) & (norm_sobel < thresh[1])] = 255
    return binary_op

cv2.namedWindow('tuner')
cv2.createTrackbar('lowerx','tuner',0,255,nothing)
cv2.createTrackbar('upperx','tuner',0,255,nothing)
cv2.createTrackbar('lowery','tuner',0,255,nothing)
cv2.createTrackbar('uppery','tuner',0,255,nothing)
cv2.createTrackbar('mag_lower','tuner',0,255,nothing)
cv2.createTrackbar('mag_upper','tuner',0,255,nothing)
cv2.createTrackbar('dir_low','tuner',0,314,nothing)
cv2.createTrackbar('dir_upper','tuner',0,314,nothing)


frame=cv2.imread('./../test_images/test2.jpg')
undist=cv2.undistort(frame,mtx,dist,None,None)

while(cv2.waitKey(10)!=ord('q')):
    lowerx=cv2.getTrackbarPos('lowerx','tuner')
    upperx=cv2.getTrackbarPos('upperx','tuner')
    lowery=cv2.getTrackbarPos('lowery','tuner')
    uppery=cv2.getTrackbarPos('uppery','tuner')
    mag_lower=cv2.getTrackbarPos('mag_lower','tuner')
    mag_upper=cv2.getTrackbarPos('mag_upper','tuner')
    dir_low=cv2.getTrackbarPos('dir_low','tuner')/100
    dir_upper=cv2.getTrackbarPos('dir_upper','tuner')/100

    sobelx=abs_sobel_mag(frame,'x',(lowerx,upperx))
    sobely = abs_sobel_mag(frame, 'y', (lowery, uppery))
    mag_sob=mag_sobel(frame,(mag_lower,mag_upper))
    dir_sob=dir_sobel(frame,(dir_low,dir_upper))
    binary_img=np.zeros_like(sobelx)
    binary_img[((sobelx==255) | (sobely==255))& ((dir_sob==255) | (mag_sob==255))]=255
    # cv2.imshow('sobelx',sobelx)
    # cv2.imshow('sobely', sobely)
    # #cv2.imshow('combined', binary_img)
    # cv2.imshow('mag',mag_sob)
    # cv2.imshow('dir',dir_sob)
    # cv2.imshow('bin', binary_img)

    fig=plt.figure()
    fig.set_figheight(100)
    fig.set_figwidth(100)
    ax1=fig.add_subplot(3,2,1)
    ax1.imshow(sobelx,cmap="gray")
    ax1.title.set_text('sobelx')
    ax2=fig.add_subplot(3,2,2)
    ax2.title.set_text('sobely')
    ax2.imshow(sobely,cmap="gray")
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.title.set_text('mag')
    ax3.imshow(mag_sob,cmap="gray")
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.title.set_text('dir')
    ax4.imshow(dir_sob,cmap="gray")
    ax5=fig.add_subplot(3,2,5)
    ax5.imshow(binary_img,cmap="gray")
    ax5.title.set_text('bin')
    fig.canvas.flush_events()

    plt.show()
    #cv2.imshow('original',frame)
    #cv2.imshow('undistorted',undist)

