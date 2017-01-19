
# coding: utf-8

# # Traffic Sign Classification with Keras
# 
# Keras exists to make coding deep neural networks simpler. To demonstrate just how easy it is, you’re going to use Keras to build a convolutional neural network in a few dozen lines of code.
# 
# You’ll be connecting the concepts from the previous lessons to the methods that Keras provides.

# ## Dataset
# 
# The network you'll build with Keras is similar to the example that you can find in Keras’s GitHub repository that builds out a [convolutional neural network for MNIST](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py). 
# 
# However, instead of using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, you're going to use the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset that you've used previously.
# 
# You can download pickle files with sanitized traffic sign data here.

# ## Overview
# 
# Here are the steps you'll take to build the network:
# 
# 1. First load the data.
# 2. Build a feedforward neural network to classify traffic signs.
# 3. Build a convolutional neural network to classify traffic signs.
# 
# Keep an eye on the network’s accuracy over time. Once the accuracy reaches the 98% range, you can be confident that you’ve built and trained an effective model.

# ## Load the Data
# 
# Start by importing the data from the pickle file.

# In[16]:

import os
import numpy as np
import tensorflow as tf
import pickle 
import cv2
import h5py
import json

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


def process_image(img):
    # convert to YCrCrb
    yuv_img=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
    
    # equalise Y channel 
    yuv_img[:,:,0]=cv2.equalizeHist(yuv_img[:,:,0])
    
    return yuv_img


data_path=os.getcwd()+"/traffic-signs-data"
training_file = data_path+"/train.p"
testing_file = data_path+"/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

nb_classes=43

# Pre-process Image
for i in range(len(X_train)):
    X_train[i]=process_image(X_train[i])

for i in range(len(X_test)):
    X_test[i]=process_image(X_test[i])

# Look at distibution of classes
hist,bin_edges=np.histogram(y_train, nb_classes)

# add extra samples to make the balance the distribution
extra_samples=(max(hist)-hist)
total_extra_samples=np.sum(extra_samples)
X_train_extra=np.empty((total_extra_samples,32,32, 3),dtype=np.uint8)
y_train_extra=np.empty((total_extra_samples))
i=0
for cls in range(nb_classes):
    class_samples= X_train[y_train==cls]
    n=len(class_samples)
    for _ in range(extra_samples[cls]):
        original=class_samples[int(np.random.uniform()*n)]
        #copy=transform_image(original)
        copy=original
        X_train_extra[i]=np.reshape(copy,(1,32,32,3))
        y_train_extra[i]=np.array([cls])
        i+=1

X_train=np.vstack((X_train, X_train_extra))
y_train=np.hstack((y_train, np.array(y_train_extra)))

#########

X_train=(X_train.astype(np.float32)-127.5)/255
X_test=(X_test.astype(np.float32)-127.5)/255

X_train, y_train = shuffle(X_train, y_train)
#X_train, X_val, y_train, y_val=train_test_split(
#X_train, y_train, test_size=0.01, random_state=0)

Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# In[107]:


# TODO: Re-construct the network and add dropout after the pooling layer.
# TODO: Compile and train the model.
model=Sequential()

# Conv 1
nb_filters=64
kernel_size=(3,3)
#pool_size=(2,2)
model.add(Convolution2D(nb_filters,kernel_size[0], kernel_size[1],
                       border_mode='valid', subsample=(1,1), input_shape=(32,32,3)))
#model.add(BatchNormalization(epsilon=1e-9))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=pool_size))

# Conv 2
nb_filters=64
kernel_size=(3,3)
#pool_size=(1,1)
model.add(Convolution2D(nb_filters,kernel_size[0], kernel_size[1], subsample=(2,2),
                       border_mode='valid'))
model.add(BatchNormalization(epsilon=1e-9))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=pool_size))

# Conv 3

nb_filters=64
kernel_size=(3,3)
#pool_size=(2,2)
model.add(Convolution2D(nb_filters,kernel_size[0], kernel_size[1], subsample=(1,1),
                       border_mode='valid'))
#model.add(BatchNormalization(epsilon=1e-9))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=pool_size))

# Conv 4

nb_filters=64
kernel_size=(3,3)
#pool_size=(1,1)
model.add(Convolution2D(nb_filters,kernel_size[0], kernel_size[1], subsample=(2,2),
                       border_mode='valid'))
model.add(BatchNormalization(epsilon=1e-9))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=pool_size))

# Dense 1
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

# Output
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

adam=Adam(lr=5e-3, decay=0.85)

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Save model
model_json=model.to_json()
with open("traffic_sign.json","w") as json_file:
    json.dump(model_json, json_file)

# load pre-trained weights
model.load_weights('traffic_sign.h5')    

# Create checkpoints
checkpoint=ModelCheckpoint('traffic_sign.h5',monitor='val_acc',save_weights_only=True,
save_best_only=True,verbose=1)
callbacks_list=[checkpoint]
    
train_datagen=ImageDataGenerator( 
              width_shift_range=0.15, 
              height_shift_range=0.15,
              zoom_range=0.15,
              rotation_range=15,
              horizontal_flip=False)

train_generator=train_datagen.flow(X_train, Y_train, batch_size=2500)
model.fit_generator(train_generator, samples_per_epoch=Y_train.shape[0], nb_epoch=30,validation_data=(X_test, Y_test),callbacks=callbacks_list, verbose=1)

# In[108]:

#history=model.fit(X_train, Y_train, nb_epoch=100, batch_size=500, verbose=1,
#                 validation_data=(X_val, Y_val))


# **Validation Accuracy**: (fill in here)

# In[109]:

score = model.evaluate(X_test, Y_test, verbose=1, batch_size=500)

# STOP: Do not change the tests below. Your implementation should pass these tests.
print('Test accuracy:%.4f'%score[1])


# ## Optimization
# Congratulations! You've built a neural network with convolutions, pooling, dropout, and fully-connected layers, all in just a few lines of code.
# 
# Have fun with the model and see how well you can do! Add more layers, or regularization, or different padding, or batches, or more training epochs.
# 
# What is the best validation accuracy you can achieve?

# In[ ]:




