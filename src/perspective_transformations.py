
import cv2
import numpy as np
import pickle

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


def undistort_img(img,mtx,dist):
    undist = cv2.undistort(img, mtx, dist, None, None)
    return undist


def get_distortion_measure():
    print("getting dist measure")
    with open('calib_param.pickle', 'rb') as f:
        [ret, mtx, dist, rvecs, tvecs] = pickle.load(f)
    return ret,mtx,dist,rvecs,tvecs


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
