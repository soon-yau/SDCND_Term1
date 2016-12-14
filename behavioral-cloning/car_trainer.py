import os
import numpy as np
import tensorflow as tf
import pickle 
import cv2
import h5py
import json
import argparse
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#from keras import backend as K

#def mean_diff_error(y_true, y_pred):
#    return K.mean(-y_pred+y_true,axis=-1)

def driving_lesson(train, nb_epoch):
        # Load dataset
	X_train, Y_train = train['ctr_frames'], train['steerings']

        # Split into training and validation sets
        X_train, X_val, Y_train, Y_val=train_test_split(X_train, Y_train, test_size=0.2)

	#Compile and train the model.
	model=Sequential()

	# Convolutional 1
	model.add(Convolution2D(nb_filter=24,nb_row=5, nb_col=5, subsample=(2,2),
						   border_mode='valid', activation='relu', input_shape=(60,160,3)))
	model.add(Convolution2D(nb_filter=36,nb_row=5, nb_col=5, subsample=(2,2),
						   border_mode='valid' ,activation='relu'))
	model.add(Convolution2D(nb_filter=48,nb_row=5, nb_col=5, subsample=(2,2),
						   border_mode='valid' ,activation='relu'))
						   
	model.add(Convolution2D(nb_filter=64,nb_row=3, nb_col=3, subsample=(1,1),
						   border_mode='valid' ,activation='relu'))
	#model.add(Convolution2D(nb_filter=64,nb_row=3, nb_col=3, subsample=(1,1),
	#                       border_mode='valid' ,activation='relu'))					   

	# Dense 
	model.add(Flatten())

	#model.add(Dense(1164, activation='relu'))
	#model.add(Dense(100, activation='relu'))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dropout(0.5))	
	# Output
	model.add(Dense(1))
        
        # load pre-trained weights
	model.load_weights('model.h5')

        # compile
	optimizer=Adam(lr=1e-4, decay=0.5)
	model.compile(loss='mean_squared_error', 
				  optimizer=optimizer, 
				  metrics=['mean_squared_error'])

        # Create checkpoints
        checkpoint=ModelCheckpoint('model.h5',monitor='train_loss',save_weights_only=True,verbose=1)
        callbacks_list=[checkpoint]
        # jitter images
#        train_datagen=ImageDataGenerator(width_shift_range=0.2, rotation_range=20)
#	train_generator=train_datagen.flow(X_train, Y_train, batch_size=500)

#        model.fit_generator(train_generator, samples_per_epoch=Y_train.shape[0], validation_data=(X_val, #Y_val),nb_epoch=nb_epoch,callbacks=callbacks_list, verbose=1)

	history=model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=700, validation_data=(X_val, Y_val),callbacks=callbacks_list, verbose=1)

	# **Validation Accuracy**: (fill in here)
	#score = model.evaluate(X_test, Y_test, verbose=1, batch_size=500)

	# Save model and weights 
	model_json=model.to_json()
	with open("model.json","w") as json_file:
		json.dump(model_json, json_file)

	model.save_weights('model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driving Lesson')
    parser.add_argument('pickle_path', type=str,
    help='Path to training pickle file')
    parser.add_argument('epochs',type=int)
    args = parser.parse_args()
    with open(args.pickle_path, 'rb') as f:
        train = pickle.load(f)
    driving_lesson(train, args.epochs)	
