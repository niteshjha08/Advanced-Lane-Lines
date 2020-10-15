import cv2
import numpy as np
import pickle
from binary_tuner import mag_sobel,abs_sobel_mag, dir_sobel, perspective_transform, undistort_img
import matplotlib.pyplot as plt

with open('calib_param.pickle','rb') as f:
    [ret,mtx,dist,rvecs,tvecs]=pickle.load(f)


def get_binary(img):
    sobelx=abs_sobel_mag(img,'x',(28,153))
    sobely=abs_sobel_mag(img,'y',(49,211))
    mag_sob=mag_sobel(img,(40,206))
    dir_sob=dir_sobel(img,(47,144))
    bin_img=np.zeros_like(sobelx)
    bin_img[((sobelx==255) & (sobely==255))& ((dir_sob==255) | (mag_sob==255))]=255
    return bin_img

def get_histogram(img):
    half_img=img[int(img.shape[0]/2):,:]
    histogram=np.sum(half_img,axis=0)
    return histogram

def main():
    img=cv2.imread('./../test_images/test2.jpg')
    undist=undistort_img(img)
    bin_img=get_binary(undist)
    warped=perspective_transform(bin_img)
    draw_boxes(warped)
    # hist=get_histogram(warped)
    # cv2.imshow('bin_img',bin_img)
    # plt.plot(hist)
    # plt.show()
    # cv2.waitKey()

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

if __name__=='__main__':
    main()