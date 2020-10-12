import cv2
import pickle
img=cv2.imread('test_images/straight_lines1.jpg')
with open('calib_param.pickle','rb') as f:
    vals=pickle.load(f)

#print(vals)
[ret,mtx,dist,rvecs,tvecs]=vals

undist=cv2.undistort(img,mtx,dist,None,None)
cv2.imshow('undist',undist)
cv2.imshow('original',img)
cv2.waitKey()