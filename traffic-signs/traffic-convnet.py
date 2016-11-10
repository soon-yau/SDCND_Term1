
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Import dataset
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
n_train=60000
n_test=10000

# In[11]:

def weight_variables(shape, stddev=1e-1):
    return(tf.Variable(tf.truncated_normal(shape ,mean=0.0,stddev=stddev)))

def bias_variables(shape):
    return(tf.Variable(tf.zeros(shape)))

def conv2d(x,W,stride=1):
    return(tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME'))

def max_pool(x, ksize=2):
    return(tf.nn.max_pool(x,
                         ksize=[1,ksize,ksize,1],
                         strides=[1,ksize,ksize,1],
                         padding='SAME'))


# In[9]:

# Create single layer fully connected layer
num_classes=10
image_pixel=28
image_size=image_pixel*image_pixel
x=tf.placeholder(tf.float32,[None,image_size])
y_=tf.placeholder(tf.float32,[None,num_classes])

downsample=1
# Convolutional layer 1
ksize=5
n_features1=32
W_conv1=weight_variables([ksize,ksize,1,n_features1])
b_conv1=bias_variables([n_features1])

x_image=tf.reshape(x,[-1,image_pixel,image_pixel,1])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool(h_conv1)
downsample*=2
# Convolutional layer 2
ksize=5
n_features2=64
W_conv2=weight_variables([ksize,ksize,n_features1,n_features2])
b_conv2=bias_variables([n_features2])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool(h_conv2)
downsample*=2

# Fully connected layer
n_fc_neurons=1024
fc_input_dim=image_pixel/downsample
W_fc1=weight_variables([fc_input_dim*fc_input_dim*n_features2,n_fc_neurons])
b_fc1=bias_variables([n_fc_neurons])

h_pool2_flat=tf.reshape(h_pool2,[-1,fc_input_dim*fc_input_dim*n_features2])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# Output layer
W_fc2=weight_variables([n_fc_neurons,num_classes])
b_fc2=weight_variables([num_classes])
y=tf.matmul(h_fc1,W_fc2)+b_fc2

# define loss
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
# define training
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   

# run training
import time

n_epoch=30
batch_size=200
n_inter=n_train/batch_size
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(mnist.test.labels.shape)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(n_epoch):
        epoch_start_t=time.clock()
        for _ in range(n_inter):
            batch_x, batch_y=mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_x, y_:batch_y})
        acc=sess.run(accuracy,feed_dict={x:batch_x, y_:batch_y})
        epoch_stop_t=time.clock()
        print("Epoch %d, elapsed %.3f training accuracy=%f"%(epoch,epoch_stop_t-epoch_start_t,acc))
    # Test trained model
    n_batch=1000
    n_inter=n_test/n_batch
    
    acc=0
    for i in range(n_inter):
    	acc+=sess.run(accuracy,feed_dict={x:mnist.test.images[i*n_batch:(i+1)*n_batch],
                                      y_:mnist.test.labels[i*n_batch:(i+1)*n_batch]})
        
    print("Test accuracy=%f"%(acc/n_inter))




