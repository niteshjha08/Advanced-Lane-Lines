import cv2
import numpy as np
import pickle


cap=cv2.VideoCapture('project_video.mp4')
with open('calib_param.pickle','rb') as f:
    [ret,mtx,dist,rvecs,tvecs]=pickle.load(f)

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

while(cv2.waitKey(0)!=ord('q')):
    ret,frame=cap.read()
    undist=cv2.undistort(frame,mtx,dist,None,None)
    sobelx=abs_sobel_mag(frame,'x',(70,150))
    sobely = abs_sobel_mag(frame, 'y', (70, 150))
    mag_sobel=mag_sobel(frame,(50,150))
    binary_img=np.zeros_like(sobelx)
    binary_img[(sobelx==255) | (sobely==255)]=255
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely', sobely)
    cv2.imshow('combined', binary_img)
    cv2.imshow('mag',mag_sobel)
    #cv2.imshow('original',frame)
    #cv2.imshow('undistorted',undist)
