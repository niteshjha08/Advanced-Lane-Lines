import cv2
import numpy as np
import pickle
import math

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

def write_perspective_transform(image):
    print('writing mtx')
    dst = np.array([[[300, 720]], [[980, 720]], [[980, 0]], [[300, 0]]])
    points = np.array([[[200, 720]], [[1100, 720]], [[685, 450]], [[595, 450]]])
    M = cv2.getPerspectiveTransform(points.astype(np.float32), dst.astype(np.float32))
    warpimg = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)
    with open('perspective_mtx.pickle','wb') as f:
      pickle.dump(M,f)
    return warpimg

def write_inv_perspective_transform(image):
    print('writing mtx inv')
    dst = np.array([[[300, 720]], [[980, 720]], [[980, 0]], [[300, 0]]])
    points = np.array([[[200, 720]], [[1100, 720]], [[685, 450]], [[595, 450]]])
    M = cv2.getPerspectiveTransform(dst.astype(np.float32), points.astype(np.float32))
    warpimg = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)
    with open('inverse_perspective_mtx.pickle','wb') as f:
      pickle.dump(M,f)
    return warpimg


def get_perspective_mtx():
    print("reading mtx")
    with open('perspective_mtx.pickle', 'rb') as f:
        M = pickle.load(f)
    return M


def get_inv_perspective_mtx():
    print("Reading inv mtx")
    with open('inverse_perspective_mtx.pickle','rb') as f:
        Minv = pickle.load(f)
    return Minv


def perspective_transform(img,M):
    warpimg = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
    return warpimg


def inv_perspective_transform(img,Minv):
    warpimg = cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
    return warpimg


def pipeline(image):
    while(cv2.waitKey(10)!=ord('q')):
        lowerx=cv2.getTrackbarPos('lowerx','tune') # (28,153)
        upperx=cv2.getTrackbarPos('upperx','tune')
        lowery=cv2.getTrackbarPos('lowery','tune') # (49,211)
        uppery=cv2.getTrackbarPos('uppery','tune')
        mag_lower=cv2.getTrackbarPos('mag_lower','tune1') # (40,206)
        mag_upper=cv2.getTrackbarPos('mag_upper','tune1')
        dir_low=cv2.getTrackbarPos('dir_low','tune1')/100   # (47,144)
        dir_upper=cv2.getTrackbarPos('dir_upper','tune1')/100
        sobelx=abs_sobel_mag(image,'x',(lowerx,upperx))
        sobely = abs_sobel_mag(image, 'y', (lowery, uppery))
        mag_sob=mag_sobel(image,(mag_lower,mag_upper))
        dir_sob=dir_sobel(image,(dir_low,dir_upper))
        binary_img=np.zeros_like(sobelx)
        binary_img[((sobelx==255) & (sobely==255))& ((dir_sob==255) | (mag_sob==255))]=255
        warpimg,orig=perspective_transform(binary_img)

        cv2.imshow('sobelx',cv2.resize(sobelx,(int(sobelx.shape[1]/2),int(sobelx.shape[0]/2))))
        cv2.imshow('sobely', cv2.resize(sobely, (int(sobely.shape[1] / 2), int(sobely.shape[0] / 2))))
        cv2.imshow('mag', cv2.resize(mag_sob, (int(sobely.shape[1] / 2), int(sobely.shape[0] / 2))))
        cv2.imshow('dir', cv2.resize(dir_sob, (int(sobelx.shape[1] / 2), int(sobelx.shape[0] / 2))))
        cv2.imshow('bin', cv2.resize(binary_img, (int(sobelx.shape[1] / 2), int(sobelx.shape[0] / 2))))
        cv2.imshow('warp', cv2.resize(warpimg, (int(sobelx.shape[1] / 2), int(sobelx.shape[0] / 2))))
        cv2.imshow('orig', cv2.resize(orig, (int(sobelx.shape[1] / 2), int(sobelx.shape[0] / 2))))

def undistort_img(img,mtx,dist):
    undist = cv2.undistort(img, mtx, dist, None, None)
    return undist


def get_distortion_measure():
    print("getting dist measure")
    with open('calib_param.pickle', 'rb') as f:
        [ret, mtx, dist, rvecs, tvecs] = pickle.load(f)
    return ret,mtx,dist,rvecs,tvecs

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

def checkperspective():
    frame = cv2.imread('./../test_images/straight_lines1.jpg')
    warp = perspective_transform(frame)
    revwarp = inv_perspective_transform(warp)
    cv2.imshow('warp', warp)
    cv2.imshow('rev', revwarp)
    cv2.imshow('frame', frame)
    merge = cv2.addWeighted(frame, 0.5, revwarp, 0.5, 0)
    cv2.imshow('merge', merge)
    cv2.waitKey()

def checknewfunctionmag():
    image=cv2.imread('./../test_images/straight_lines1.jpg')
    sobelx,actualsobelx = abs_sobel_mag(image, 'x', (28,153))
    sobely,actualsobely = abs_sobel_mag(image, 'y', (49,211))
    mag_sob = get_sobel_mag(image, actualsobelx,actualsobely,(40, 206))
    mag2=mag_sobel(image,(40,206))
    cv2.imshow('mag2',mag2)
    cv2.imshow('mag1',mag_sob)
    cv2.waitKey()

if __name__=='__main__':
    #main()
    print("running main")
    checknewfunctionmag()


