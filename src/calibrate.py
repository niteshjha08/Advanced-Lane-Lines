import cv2
import numpy as np
import os
import pickle
calib_dir='camera_cal'
img_paths=[]
nx=9
ny=6
imagepoints=[]
objectpoints=[]
for dir,subdir,filename in os.walk(calib_dir):
    for file in filename:
        if(file.endswith('.jpg')):
            img_paths.append(os.path.join(dir,file))
objp=np.zeros((54,3),dtype=np.float32)
objp[:,:2]=np.mgrid[:9,:6].T.reshape(-1,2)

for img in img_paths:
    image=cv2.imread(img)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,(nx,ny),None)
    if(ret==True):
        imagepoints.append(corners)
        objectpoints.append(objp)

ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(objectpoints,imagepoints,gray.shape[::-1],None,None)
calib_param=[ret,mtx,dist,rvecs,tvecs]
with open('calib_param.pickle','wb') as f:
    pickle.dump(calib_param,f)
# img=cv2.imread('test_images/test1.jpg')
# undist=cv2.undistort(img,mtx,dist,None,mtx)
# cv2.imshow('undist',undist)
# cv2.imshow('img',img)
# cv2.waitKey()




