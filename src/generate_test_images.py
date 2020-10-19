import cv2
import numpy as np

cap=cv2.VideoCapture('./../project_video.mp4')

while(True):
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    if(cv2.waitKey()==ord('q')):
        break
    if(cv2.waitKey()==ord('s')):
        cv2.imwrite('test_img.jpg',frame)
