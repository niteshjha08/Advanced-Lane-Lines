# **Advanced lane lines detection**

(insert one final annotated image here)

Detect lane lines in a little more flexible conditions (curvature, lighting variations, camera distortions) using computer vision techniques. This is an extension of the [*Basic lane detection*](https://github.com/niteshjha08/Basic_Lane_detection) which was constrained to straight lines and region-specific search.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration
The code for calibration procedure is in the file  [calibrate.py]()
* Images from [*camera_cal*]() was used for calibration. 9x6 chessboard images were used and OpenCV functions to calibrate the camera.  
* Imagepoints, i.e. image corners were found using `findChessboardCorners()` function and objectpoints were manually coded to lie in a plane (z=0). Then `calibrateCamera()` was used to obtain camera matrix, distortion coefficients, rvecs and tvecs. These parameters were saved in the file `calib_param.pickle` to be used later.

Here is an example of this result:

(Original calibration1)             (calibration1_undist)
**1. Camera Calibration and Distortion correction**



**2. Thresholding on color and gradient values**
The file [binary_tuner.py]() contains all code using for testing out and figuring threshold values for this step.
A binary image is obtained using Sobel operator and HLS colorspace thresholding.
 The current implementation uses the functions `mag_sob()` and `color_thresh()` for sobel magnitude and HLS color thresholding respectively. The hue and saturation channels were used as they were most useful for the given conditions. 
The output for this step is shown below:

(Insert image................)

**3. Perspective transformation**
The file [perspective_transformations.py]() contains relevant functions for these.
As images are from the car's perspective, obtaining lane information such as curvature and offset is not possible directly. Hence the birds-eye view is generated using a perspective transformation. 
 

* Imagepoints are hard-coded for this, and the desired destination points are written down accordingly.

OpenCV function `getPerspectiveTransform()` was used to get the perspective matrix and was then stored in `perspective_mtx.pickle`. The inverse matrix was also found and stored in `inverse_perspective_mtx.pickle` for inverse projection in the later stages of the pipeline. `warpPespective()` was used to perform the transformation of image. The result of this perspective transformation is shown below.
(Insert...... color images)


This step is applied in the pipeline to binary-thresholded images. The output is shown below:
(Insert binary images...)


**All functions used after this stage is in file [detect_lane.py]()**
\
**4. Detecting lanes from this image**


This is done in the function `sliding_window()` 
This process begins using a histogram-approach.
* The lower half of the image is used to form a histogram of pixel intensities along the (half) height of the image. Two maximum peaks are chosen which will be the initial lane location. 

(Insert image)

* Then, after the lane bases are determined, a sliding window approach is used to find the lanes in the rest of the image. The window 'slides' horizontally if the number of nonzero pixels found within a 'margin' of the current position is greater than a threshold 'minpix'. Once these windows are found, lane pixels detected are appended to a list. The windows move up and cover the whole length of the lane.

(Insert Image)

* Then, `fit_line` function uses the array of pixels for left and right lanes and fits a second order polynomial to them. 

(insert image)

In addition, it calculates vehicle offset (using lane center and image center) and radius of curvature of both lanes using the function `radius_curvature()`. [Note: It is assumed that the camera is mounted on the center of the car]. 
For converting the pixel space to world space, the following conversion ratio is used:
\
30 metres = 720 pixels in the vertical direction of the perspective transformed image.
\
3.7 metres = 700 pixels in the horizontal direction of the perspective transformed image.


However, the histogram-->sliding_window-->fit_line is computationally ineffecient for every frame. Can we do better? Yes!
* As lanes don't shift all that much from frame to frame, the previous fit lanes can provide a rough estimate of the lanes in the next frame. To leverage this fact, `search_around_poly()` is used. It narrows the search with the margin parameter (+/- 100 here), only within which the search takes place for non-zero pixels. 

(insert image)
* These are then used to fit the actual curve for this frame (again, using `fit_line()` ). Global variables are used to retain values across the frames. * * Thus histogram and sliding window search has to run only once using this method. Also this gives added advantage of neglecting noisy pixels that might have gone through the binary thresholding step (although this is not used or relied on for that purpose)

**5. Projection of lane back onto the image**

The `fill_lane()` function fills the detected lane boundaries. 


(Insert Image)

It is then inverse-perspective-transformed back to the original image, and `addWeighted()` is used to form the final annotated image.

(Insert image)