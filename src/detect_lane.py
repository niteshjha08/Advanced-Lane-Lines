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
    img=cv2.imread('./../test_images/straight_lines1.jpg')
    undist=undistort_img(img)
    bin_img=get_binary(undist)
    warped=perspective_transform(bin_img)
    hist=get_histogram(warped)
    cv2.imshow('bin_img',bin_img)
    plt.plot(hist)
    plt.show()
    cv2.waitKey()

if __name__=='__main__':
    main()