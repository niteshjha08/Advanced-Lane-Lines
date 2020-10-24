import cv2
import numpy as np
import pickle
import math
from perspective_transformations import perspective_transform
# cap=cv2.VideoCapture('project_video.mp4')


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
    binary_op[((norm_sobel>threshold[0]) & (norm_sobel<threshold[1]))]=255
    return binary_op,sobel

def get_sobel_mag(img,sobelx,sobely,thresh=(0,255),ksize=3):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    abs_sobel_xy = np.sqrt(sobelx ** 2 + sobely ** 2)
    norm_sobel = (255 * abs_sobel_xy / np.max(abs_sobel_xy)).astype(np.uint8)
    binary_op = np.zeros_like(gray)
    binary_op[(norm_sobel > thresh[0]) & (norm_sobel < thresh[1])] = 255

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

def dir_sobel(img,thresh=(0,np.pi/2),ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_sobel= np.arctan2(abs_sobely,abs_sobelx)
    binary_op = np.zeros_like(gray)
    binary_op[(dir_sobel > thresh[0]) & (dir_sobel < thresh[1])] = 255
    return binary_op

def tuner_function(image):
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('tune')
    cv2.namedWindow('tune1')
    cv2.createTrackbar('lowerx', 'tune', 0, 255, nothing)
    cv2.createTrackbar('upperx', 'tune', 0, 255, nothing)
    cv2.createTrackbar('lowery', 'tune', 0, 255, nothing)
    cv2.createTrackbar('uppery', 'tune', 0, 255, nothing)
    cv2.createTrackbar('mag_lower', 'tune1', 0, 255, nothing)
    cv2.createTrackbar('mag_upper', 'tune1', 0, 255, nothing)
    cv2.createTrackbar('dir_low', 'tune1', 0, 255, nothing)
    cv2.createTrackbar('dir_upper', 'tune1', 0, 255, nothing)
    while(cv2.waitKey(10)!=ord('q')):
        lowerx=cv2.getTrackbarPos('lowerx','tune') # (28,153)
        upperx=cv2.getTrackbarPos('upperx','tune')
        lowery=cv2.getTrackbarPos('lowery','tune') # (49,211)
        uppery=cv2.getTrackbarPos('uppery','tune')
        mag_lower=cv2.getTrackbarPos('mag_lower','tune1') # (40,206)
        mag_upper=cv2.getTrackbarPos('mag_upper','tune1')
        dir_low=cv2.getTrackbarPos('dir_low','tune1')/100   # (47,144)
        dir_upper=cv2.getTrackbarPos('dir_upper','tune1')/100
        sobelx,_=abs_sobel_mag(image,'x',(lowerx,upperx))
        sobely,_= abs_sobel_mag(image, 'y', (lowery, uppery))
        mag_sob=mag_sobel(image,(mag_lower,mag_upper))
        dir_sob=dir_sobel(image,(dir_low,dir_upper))
        binary_img=np.zeros_like(sobelx)
        binary_img[((sobelx==255) & (sobely==255))& ((dir_sob==255) | (mag_sob==255))]=255


        cv2.imshow('sobelx',cv2.resize(sobelx,(int(sobelx.shape[1]/2),int(sobelx.shape[0]/2))))
        cv2.imshow('sobely', cv2.resize(sobely, (int(sobely.shape[1] / 2), int(sobely.shape[0] / 2))))
        cv2.imshow('mag', cv2.resize(mag_sob, (int(sobely.shape[1] / 2), int(sobely.shape[0] / 2))))
        cv2.imshow('dir', cv2.resize(dir_sob, (int(sobelx.shape[1] / 2), int(sobelx.shape[0] / 2))))
        cv2.imshow('bin', cv2.resize(binary_img, (int(sobelx.shape[1] / 2), int(sobelx.shape[0] / 2))))



def main():
    cv2.namedWindow('tune')
    cv2.namedWindow('tune1')
    cv2.createTrackbar('lowerx','tune',0,255,nothing)
    cv2.createTrackbar('upperx','tune',0,255,nothing)
    cv2.createTrackbar('lowery','tune',0,255,nothing)
    cv2.createTrackbar('uppery','tune',0,255,nothing)
    cv2.createTrackbar('mag_lower','tune1',0,255,nothing)
    cv2.createTrackbar('mag_upper','tune1',0,255,nothing)
    cv2.createTrackbar('dir_low','tune1',0,255,nothing)
    cv2.createTrackbar('dir_upper','tune1',0,255,nothing)

    frame=cv2.imread('./../test_images/straight_lines1.jpg')
    undist=undistort_img(frame)
    pipeline(undist)


def hls_thresh(img,channel,thresh=(0,255)):
    hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    if(channel=='h'):
        c=0
    if(channel=='l'):
        c=1
    if(channel=='s'):
        c=2
    hls_channel=hls[:,:,c]
    binary_op=np.zeros_like(hls_channel)
    binary_op[(hls_channel>thresh[0]) & (hls_channel<thresh[1])]=255
    return binary_op

def hls_tuner(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    cv2.namedWindow('tuner')
    cv2.createTrackbar('huelow','tuner',0,255,nothing)
    cv2.createTrackbar('satlow', 'tuner', 0, 255, nothing)
    cv2.createTrackbar('liglow', 'tuner', 0, 255, nothing)
    cv2.createTrackbar('huehigh', 'tuner', 0, 255, nothing)
    cv2.createTrackbar('sathigh', 'tuner', 0, 255, nothing)
    cv2.createTrackbar('lighigh', 'tuner', 0, 255, nothing)
    while(cv2.waitKey(100)!=ord('q')):
        hlow_val=cv2.getTrackbarPos('huelow','tuner')
        slow_val=cv2.getTrackbarPos('satlow','tuner')
        llow_val=cv2.getTrackbarPos('liglow','tuner')
        hhigh_val = cv2.getTrackbarPos('huehigh', 'tuner')
        shigh_val = cv2.getTrackbarPos('sathigh', 'tuner')
        lhigh_val = cv2.getTrackbarPos('lighigh', 'tuner')
        hue_bin=np.zeros_like(h_channel)
        sat_bin=hue_bin.copy()
        lig_bin=hue_bin.copy()
        hue_bin[(h_channel>hlow_val)&(h_channel<hhigh_val)]=255
        sat_bin[(s_channel > slow_val) & (s_channel < shigh_val)] = 255
        lig_bin[(l_channel > llow_val) & (l_channel < lhigh_val)] = 255
        cv2.imshow('hue', cv2.resize(hue_bin,(hue_bin.shape[1]//3,hue_bin.shape[0]//3)))
        cv2.imshow('sat', cv2.resize(sat_bin,(hue_bin.shape[1]//3,hue_bin.shape[0]//3)))
        cv2.imshow('light',cv2.resize(lig_bin,(hue_bin.shape[1]//3,hue_bin.shape[0]//3)))
        bin_img=np.zeros_like(hue_bin)
        bin_img[((hue_bin==255)&(sat_bin==255))]=255
        cv2.imshow('finally',cv2.resize(bin_img,(hue_bin.shape[1]//3,hue_bin.shape[0]//3)))


def color_thresh(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:, :, 0]
    s_channel = hls[:, :, 2]
    hue_bin = np.zeros_like(h_channel)
    sat_bin = hue_bin.copy()

    hue_bin[(h_channel > 15) & (h_channel < 90)] = 255
    sat_bin[(s_channel > 80) & (s_channel < 255)] = 255
    bin1=np.zeros_like(h_channel)
    bin1[(hue_bin==255)&(sat_bin==255)]=255
    # cv2.imshow('binHA',bin1)
    # cv2.imshow('actual was this', img)
    # cv2.imshow('hue was this', hue_bin)
    # cv2.imshow('sat was this', sat_bin)
    # cv2.waitKey()
    return bin1



if __name__=='__main__':
    #main()
    print("running main")
    frame = cv2.imread('./new_tests/test_img25.jpg')
    #hls_tuner(frame)
    color_thresh(frame)
    #pipeline(frame)


