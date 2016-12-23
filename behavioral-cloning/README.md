## Behavioral Cloning Project
This project is teach a car to drive itself in with end-to-end deep learning by using pictures captured from front cameras of the car and the steering angles.

# Data and Augmentation
The core training data is provided by Udacity. However, the dataset is considered as small with only 8k samples. Attempts have been made to create additional data for recovery but it didn't work well as it was suspected that the incompatible driving styles made the neural network training more challenging and result in higher loss. Therefore, it was decided to stick with the given dataset and used other techniques to improve the training. Given small sample size, various data augmentation techniques were employed to increase the number of samples.

## Camera images and steering angle
Therefore, images from all three cameras:left, center, and right are used randomly to train the CNN (convolutional neural network). For images from left camera, a angular shift of +0.25 (equivalannt to 6.25 degrees to the right) is added to the steering angle from center. Similarly, 0.25 is subtracted for right image. 

## Image Translation
The image is translated horizontally and vertically with range of +-0.2*image columns and +-0.1*image rows respectively.

## Flipping Image
Image is also being flipped horizontally randomly in which the sign of steering angle is to be reversed.

## Brightness
The image is converted from BGR to HSV and the brightness is reduced by multiplying a random factor of 0.2 to 1.0. 

## Image Size
The top 32 rows of image is cropped out to remove portion that are unrelated to road and driving conditions, while the bottom 20 rows are cropped to remove the image of car dashboard. Then the image is resize to 64x64 to reduce computational and memory cost.

Many of these effects are illustrated in driving_lesson.ipynb.

# Convolutional Neural Network
The convolution neural network architecture in Nvida's "End to End Learning for Self-Driving Cars" paper https://arxiv.org/abs/1604.07316 was taken as reference. However, since the image size and numbers are far fewer in this project, the last convolutional layer and first fully connected layer were removed. 

The finally architecture is consist of:
1. Conv layer - 24 filters, 5x5 kernel size, stride 2x2, ReLU activation
2. Conv layer - 36 filters, 5x5 kernel size, stride 2x2, ReLU activation
3. Conv layer - 48 filters, 5x5 kernel size, stride 2x2, ReLU activation
4. Conv layer - 64 filters, 3x3 kernel size, stride 1x1, ReLU activation
5. Dropout layer - 0.5
6. Fully connected layer - 500 neurons, ReLU activation
7. Fully connected layer - 50 neurons, ReLU activation
8. Fully connected layer - 10 neurons, ReLU activation
9. Dropout layer - 0.5
10. Fully connected layer - 1 neurons

# Training
Adam optimizer with learning rate of 5e-3 and decay rate of 0.75 was used to train the neural network and the minimum mean square error (MSE) was choosen as metrics.

A image generator was written to generate image for random camera choice, image translation, image flipping and image brightness.

Since the dataset is unbalanced with many more images on straight road i.e. steering angle of 0, it is therefore crucial to make sure that the ratio of 0 degree is reduced in training (together with data augmentation) so that neural network could learn to steer at curves with larger steering angle. A probabiliy thresholding method (credit to Vivek Yadav https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.xv4b8t1zl) was used to allow more large angles to be train initially, and gradually include more 0 degree angle in subsequent epochs. 

MSE was used mainly to see that the loss function decreases in every epoch to make sure the network is being trainned correctly. However, lower MSE does not guarantee that car could drive properly, it may actually be overfitting the higher steering angle and cause the car to drive in zig-zag fashion. Therefore, if I see zig-zagging in simulation, I would reduce the number of epochs and re-run the simulation. I used 5 epochs in final model that resulted in MSE of slightly over 0.04. 
