# **Advanced lane lines detection**

![final_img](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/final_output.PNG)

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
The code for calibration procedure is in the file  [calibrate.py](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/src/calibrate.py)
* Images from [*camera_cal*]() was used for calibration. 9x6 chessboard images were used and OpenCV functions to calibrate the camera.  
* Imagepoints, i.e. image corners were found using `findChessboardCorners()` function and objectpoints were manually coded to lie in a plane (z=0). Then `calibrateCamera()` was used to obtain camera matrix, distortion coefficients, rvecs and tvecs. These parameters were saved in the file `calib_param.pickle` to be used later.

Here is an example of this result:
\
![chessboard distorted](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/calibration1_distorted.jpg)  
 
*Distorted chessboard image* 
\
![chess_undist](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/calibration1_undist.jpg)
*Undistorted (corrected image)*

## Pipeline
### 1. Camera Calibration and Distortion correction
* Calibration parameters are loaded from `calib_param.pickle` using `get_distortion_measure()` in calibrate.py.
* Camera matrix and distortion coeff. are used to undistor the video frame (img).

![Distorted](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/distorted_input.PNG)  
*Distorted frame*
![Undist](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/undistorted.PNG)
*Undistorted frame*


### 2. Thresholding on color and gradient values
The file [binary_tuner.py](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/src/binary_tuner.py) contains all code using for testing out and figuring threshold values for this step.
A binary image is obtained using Sobel operator and HLS colorspace thresholding.
 The current implementation uses the functions `mag_sob()` and `color_thresh()` for sobel magnitude and HLS color thresholding respectively. The hue and saturation channels were used as they were most useful for the given conditions. 
The output for this step is shown below:

![binary_img](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/binary.PNG)

### 3. Perspective transformation
The file [perspective_transformations.py](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/src/perspective_transformations.py) contains relevant functions for these.
As images are from the car's perspective, obtaining lane information such as curvature and offset is not possible directly. Hence the birds-eye view is generated using a perspective transformation. 
 

* Imagepoints are hard-coded for this, and the desired destination points are written down accordingly.

OpenCV function `getPerspectiveTransform()` was used to get the perspective matrix and was then stored in `perspective_mtx.pickle`. The inverse matrix was also found and stored in `inverse_perspective_mtx.pickle` for inverse projection in the later stages of the pipeline. `warpPespective()` was used to perform the transformation of image. The result of this perspective transformation is shown below.
![colorimg,trapezium](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/perspectivepoints.PNG) ![imagewarp](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/perspectivepoints_result.PNG)


This step is applied in the pipeline to binary-thresholded images. The output is shown below:
![warped_binary](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/warped.PNG)


**All functions used after this stage is in file [detect_lane.py](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/src/detect_lane.py)**

### 4. Detecting lanes from this image


This is done in the function `sliding_window()` 
This process begins using a histogram-approach.
* The lower half of the image is used to form a histogram of pixel intensities along the (half) height of the image. Two maximum peaks are chosen which will be the initial lane location. 

![histogram](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/histogram.PNG)

Then, after the lane bases are determined, a sliding window approach is used to find the lanes in the rest of the image:
* The image is divided into 9 parts along the vertical.
* The base lane location is fixed at the two peaks of histogram.
* For the part above this base, a margin of +\- 100 is checked for non-zero pixels.
* The window 'slides' horizontally if the number of nonzero pixels found within a 'margin' of the current position is greater than a threshold 'minpix'. Once these windows are found, lane pixels detected are appended to a list. The windows move up and cover the whole length of the lane.

![sliding_window](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/sliding_window.PNG)

* Then, `fit_line` function uses the array of pixels for left and right lanes and fits a second order polynomial to them. 

![fit_line](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/fit_line.PNG)

In addition, it calculates vehicle offset (using lane center and image center) and radius of curvature of both lanes using the function `radius_curvature()`. [Note: It is assumed that the camera is mounted on the center of the car]. 
For converting the pixel space to world space, the following conversion ratio is used:
\
30 metres = 720 pixels in the vertical direction of the perspective transformed image.
\
3.7 metres = 700 pixels in the horizontal direction of the perspective transformed image.


However, the histogram-->sliding_window-->fit_line is computationally ineffecient for every frame. Can we do better? Yes!
* As lanes don't shift all that much from frame to frame, the previous fit lanes can provide a rough estimate of the lanes in the next frame. To leverage this fact, `search_around_poly()` is used. It narrows the search with the margin parameter (+/- 100 here), only within which the search takes place for non-zero pixels. 

![marginal search](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/search_around_poly.PNG)
* These are then used to fit the actual curve for this frame (again, using `fit_line()` ). Global variables are used to retain values across the frames. * * Thus histogram and sliding window search has to run only once using this method. Also this gives added advantage of neglecting noisy pixels that might have gone through the binary thresholding step (although this is not used or relied on for that purpose)

### 5. Projection of lane back onto the image

The `fill_lane()` function fills the detected lane boundaries. 


![filledimg](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/fill_lane.PNG)

* To form the final output image, the original image is taken
* The 'lane filled' image above is perspective transformed to project it back onto the image.
* A weighted average of these two images is taken.
* Calculated radius of curvature of left and right lanes are averaged to form one radius of curvature. It is then added to this image.
* Lane offset calculated is also added to the image.


![finalimg](https://github.com/niteshjha08/Advanced-Lane-Lines/blob/master/writeup_images/final_output.PNG)