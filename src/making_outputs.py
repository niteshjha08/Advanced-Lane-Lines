from perspective_transformations import get_distortion_measure
import cv2

ret, mtx, dist, rvecs, tvecs = get_distortion_measure()
img=cv2.imread('./../camera_cal/calibration1.jpg')
undist=cv2.undistort(img,mtx,dist,None,mtx)

cv2.imshow('undist',undist)
cv2.imshow('img',img)
cv2.imwrite('./../result_images/calibration1_undist.jpg',undist)
cv2.waitKey()