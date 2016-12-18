## Advanced Lane Finding

The goals / steps of this project are the following:  

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  The video called `project_video.mp4` is the video your pipeline should work well on.  `challenge_video.mp4` is an extra (and optional) challenge for you if you want to test your pipeline.

If you're feeling ambitious (totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## File Structures
The code were implemented in two jupyter notebook:
1. calibration.ipynb to perform camera calibration. The distortion and camera matrix is stored in pickle file named camera_cal.p
2. advanced_lane_finding.ipynb contain codes to perform lane detection, curvature and car position estimation

# Advanced Lane Finding Pipeline
- load calibration matrices
- load image/video frame as RGB image
- apply calibration matrices to undistort image
- apply color threshold
   - convert image into HLS color space
   - use threshold [100,255] on saturation (S) channel to extract white and yellow lines. However, shadow on roads tend to fall within the same range
   - use threshold [0, 30] on hue (H) channel to filter out dark color e.g. shadow
   - do logical AND of binary images from S and H channels after thresholding to return a binary image
- separately, transform the undistorted image to grayscale and perform sober edge detection that uses:
   - gradient in x direction, threshold=(10, 255)
   - magnitude threshold = (50, 255)
   - direction threshold = (13.5 degree, 76.5 degree)
   - these values are used mainly to try to detect the dashed lanes and of course there are endless combination that can be tried to improve the detection
- color thresholding is good at extracting solid yellow left lane while sober edge detection is good at extracting dashed line. Therefore, they are combined (logical OR) to give a better representation

- With the use of code from project 1 (not shown here), straight lines were found on advanced lane finding test images where trapezoid were drawn to find 4 corners that represent points on parallel lanes. These values are then used as hard coded value as 'src_pts' for this project, assuming the same camera and car were used to capture the video footage.
- Region masking with trapezoid is used to mask out uninterested region 
- The binary image after color and gradient thresholding are warped using src_pts and dst_pts where the latter specifies the image positions that you want the src_pts to warp into. The values of dst_pts need to be chosen carefully so that the warped image capture the curve of lanes farther away.
- We know road lanes are parallel to each other, so in warped image, we can assume they are parallel lines for portion that are very close to the camera. We can make use of this information:
   - apply another region masking in the bottom center region of warped image to remove textures/scratches in the middle of lane that would confuse the road finding
   - use bottom left and right corner of destination points for warping as starting x position for lane finding
- Use windowing, and mean of histogram to find the next x center position towards the top of the image. Do this separately to find left and right lanes then combine them into a single binary image. I have tried to search the lane using path from previously fitted polynomial curve but they did not work more reliable and was not included in the final code.
- Then we do polynomial fitting on the left and right curve found
- Correlation is performed on first two polynomial coefficient of left and right fit as these two coefficient determine the shape of the curve and the third is y-offset of left and right lane which should be different and hence not used for correlation. Then the correlation is used to roughly determine if the lanes are curving to the same direction, if not, then perhaps lines were not found correctly and the previously found values are used instead.
- Curvatures are then calculated
- xm_per_pix is calculated by assuming lane width is 3.7, and divided by the distance of lane position (in pixel) in the warped image. From this, the car position offset with relative to lane is calculated.
- Finally, polygon is filled between the fitted polynomial lines in warped image. This is then unwarped and added to the undistorted color image. 


   
