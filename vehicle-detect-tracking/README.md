## Vehicle Tracking

The objective of this project is to detect, localize and track cars in the video.

The approach is to use sliding windows and linear Support Vector Machine (SVM) trained on Histogram of Gradient (HoG) and color histogram. The code was written in Python and also use modules from sklearn and OpenCV.
I will describe the main design in following texts while the reasoning of software design e.g. definition of Class and software flow are left as comment in Jupyter Notebook.

# Features
The main feature to use in HoG, supported by HoC (Histogram of Color). The main idea of using HoG is that cars have similar edges which is different from many of the non-car images. 
Color information is used to improve the prediction accuracy. The histograms were first normalised to between 0.0 and 1.0 as returned by the library, which is then normalized to between -1.0 and 1.0 to help to speed up training of SVM.
The parameters of features such as color space, the number of bins were decided by trying different parameters and pick the one that would give best result.

# Decision Tree
I have also train a classifier based on decision tree and have plotted the tree for illustration. However, this is not used in the final code for vehicle tracking.

# Training
The test accuracy on original GTI and KITTI training dataset gave about 0.96 accuracy. However, when testing in on the video, there were quite a number of false positive detected. 
Therefore, some negative samples i.e. non-car images have been added into the training set and the final prediction accuracy is 0.98.

# Sliding Windows
Multi-scale windows are used, the following code snippet define the ((window size), (sliding distance), (range in vertical direction)) in (x,y) direction
    self.multiscale_windows=[[(80,80),(25,25),(400,560)],
                             [(100,100),(37,37),(400,550)],
                             [(180,180),(70,70),(400,660)]]

# Merging overlapping boxes
Due to the nature of overlapping slidding windows, there may be several overlapping boxes around a car. Therefore, algorithm is run to merge all the overlapping boxes into a bigger one.

# Tracking
A new car detected is not confirmed immediately. Instead, a counter is initialized and  will increment in every frame if it overlap with the new boxes found in the next frame. Only once the counter reaches a given number e.g. 3, only we have confidence that this is not a false positive and to accept this is a car. 

# Future improvement
The number of features that are used can be reduced by using decision tree to prune insignificant feature in order to speed up the detection. 
There are still some false positive, one way to improve it would be to use CNN based approach. 
The tracking algorithm is also a simple one and the car will lost track when it is occluded, Kalman or particile filter can be implemented to offer more robust tracking.
The likely scenario when detection may fail is when there is car accident and the cars are coming at different viewing angle e.g. the car is rolling toward you and the classifier may not pick that up as a car and perform collision avoidance.