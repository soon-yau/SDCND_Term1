import tensorflow as tf
import math
import cv2
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# TODO: fill this in based on where you saved the training and testing data
data_path=os.getcwd()+"/traffic-signs-data"
#training_file = os.getcwd()+"/train3.p"
training_file = data_path+"/train2.p"
testing_file = data_path+"/test2.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# TODO: number of training examples
n_train = X_train.shape[0]

# TODO: number of testing examples
n_test = X_test.shape[0]

# TODO: what's the shape of an image?
image_shape =X_train.shape[1:3]

# TODO: how many classes are in the dataset
n_classes = max(y_train)+1

n_channel = X_train.shape[3]

# Distribution
hist,bin_edges=np.histogram(y_train, n_classes)

extra_samples=(max(hist)-hist)
total_extra_samples=np.sum(extra_samples)
X_train_extra=np.empty((total_extra_samples,32,32,3))
y_train_extra=np.empty((total_extra_samples))
i=0
for cls in range(n_classes):
    class_samples= X_train[y_train==cls]
    #jitter_per_sample=int(math.ceil(extra_samples[cls]/float(len(class_samples))))
    #print(cls,extra_samples[cls],len(class_samples),jitter_per_sample)
    n=len(class_samples)
    for _ in range(extra_samples[cls]):	
	original=class_samples[int(np.random.uniform()*n-1)]
        copy=np.reshape(original,(1,32,32,3))
        X_train_extra[i]=copy
        y_train_extra[i]=np.array([cls])
        i+=1


X_train_extra=np.vstack((X_train, X_train_extra))
#print(y_train.shape, y_train_extra.shape)
y_train_extra=np.hstack((y_train, np.array(y_train_extra)))
print("original shape",y_train.shape, X_train.shape)
print("extra shape",y_train_extra.shape, X_train_extra.shape)

plt.subplot(2,1,1)
plt.hist(y_train,n_classes)
plt.title("Distribution of training classes")
plt.subplot(2,1,2)
plt.hist(y_train_extra,n_classes)
plt.title("Distribution of test classes")
plt.show()

filehandler = open("train4_repeat.p","wb")
new_train={}
new_train['features']=X_train_extra
new_train['labels']=y_train_extra
pickle.dump(new_train,filehandler)
filehandler.close()
new_train=[]





