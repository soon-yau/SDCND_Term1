
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import h5py
import json

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution2D
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


# # Pre-processing

# In[2]:

data_path='./data/udacity/'

dataset=pd.read_csv(data_path+'driving_log.csv', skiprows=1,skipinitialspace=True,
              names=['center','left','right','steering','throttle','brake','speed'])



# In[3]:

def process_img(line_data):
    # select randomly center, left or right camera

    camera=np.random.randint(3)
    if camera==0:
        fpath=line_data['left'].strip()
        angle_shift=0.25
    if camera==1:
        fpath=line_data['center'].strip()
        angle_shift=0.0
    if camera==2:
        fpath=line_data['right'].strip()
        angle_shift=-0.25
    
    # read image    
    img=cv2.imread(data_path+fpath)

    # add steering angle shift
    angle = line_data['steering']+angle_shift
    
    # Translation
    x_trans_range= int(0.2*img.shape[1])
    x_trans = np.random.randint(low=-x_trans_range, high=x_trans_range)
    angle_trans = x_trans/x_trans_range*0.2
    angle=angle+angle_trans
    
    y_trans_range= int(0.1*img.shape[0])
    y_trans = np.random.randint(low=-y_trans_range, high=y_trans_range)
    trans_matrix = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
    img=cv2.warpAffine(img, trans_matrix, (img.shape[1], img.shape[0]))

    # Convert Color space and augment brightness
    img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    brightness_factor=np.random.uniform(low=0.2, high=1.0)
    img[:,:,2]=img[:,:,2]*brightness_factor
    
    # Resize image
    img=img[32:-20,:,:]
    img=cv2.resize(img,(64,64))
    
    # flip image
    if np.random.randint(2)==0:
        img=cv2.flip(img,1)
        angle=-1*angle

    # normalise
    img=np.float32(img/255-0.5)
    
    return img, angle


# In[4]:
# CNN Architecture

model=Sequential()

# Convolutional 1
model.add(Convolution2D(nb_filter=24,nb_row=5, nb_col=5, subsample=(2,2),
                        border_mode='valid', activation='relu', input_shape=(64,64,3)))
model.add(Convolution2D(nb_filter=36,nb_row=5, nb_col=5, subsample=(2,2),
                        border_mode='valid' ,activation='relu'))
model.add(Convolution2D(nb_filter=48,nb_row=5, nb_col=5, subsample=(2,2),
                        border_mode='valid' ,activation='relu'))
model.add(Convolution2D(nb_filter=64,nb_row=3, nb_col=3, subsample=(1,1),
                        border_mode='valid' ,activation='relu'))

model.add(Dropout(0.5))
# Dense 
model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))

# Output
model.add(Dense(1))
    
# load pre-trained weights
#model.load_weights('model.h5')

# compile
optimizer=Adam(lr=5e-3, decay=0.75)

model.compile(loss='mean_squared_error', 
              optimizer=optimizer, 
              metrics=['mean_squared_error'])

# Save model
model_json=model.to_json()
with open("model.json","w") as json_file:
    json.dump(model_json, json_file)
   


# # Training

# In[5]:

def image_generator_(bias_threshold, batch_size=200):

    gen_img=np.empty((batch_size, 64,64,3),np.float32)
    gen_steering=np.empty((batch_size),np.float32)
    while 1:
        for i in range(batch_size):
            idx=np.random.randint(len(dataset))
            exceed_threshold=False
            while exceed_threshold==False:
                x,y=process_img(dataset.iloc[idx])
            
                if ((abs(y)<0.1) and (np.random.uniform()>bias_threshold)) or (abs(y)>=0.1):
                    exceed_threshold=True
                    
            gen_img[i]=x
            gen_steering[i]=y
        yield gen_img, gen_steering


# In[6]:

total_nb_epoch=5

for epoch in range(total_nb_epoch):
    bias_threshold=1/(epoch+1)
    image_generator=image_generator_(bias_threshold, batch_size=1000)
    model.fit_generator(image_generator, samples_per_epoch=20000, nb_epoch=1, verbose=1)
    
# Save weight
model.save_weights('model.h5')


# In[ ]:



