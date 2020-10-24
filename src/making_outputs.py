from perspective_transformations import get_distortion_measure, get_perspective_mtx,get_inv_perspective_mtx, perspective_transform
import cv2
import numpy as np
ret, mtx, dist, rvecs, tvecs = get_distortion_measure()
M = get_perspective_mtx()
Minv = get_inv_perspective_mtx()
img=cv2.imread('./../test_images/straight_lines1.jpg')
undist=cv2.undistort(img,mtx,dist,None,mtx)
dst = np.array([[[300, 720]], [[980, 720]], [[980, 0]], [[300, 0]]])
points = np.array([[[200, 720]], [[1100, 720]], [[685, 450]], [[595, 450]]])
undist=cv2.polylines(undist,[points],True,(0,0,255),2)
warp=perspective_transform(undist,M)

undist=cv2.polylines(undist,[points],True,(0,0,255),2)
cv2.imshow('undist',undist)
cv2.imshow('warp',warp)
cv2.waitKey()