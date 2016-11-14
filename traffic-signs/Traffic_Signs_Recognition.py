
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# 
# ## Step 1: Dataset Exploration
# 
# Visualize the German Traffic Signs Dataset. This is open ended, some suggestions include: plotting traffic signs images, plotting the count of each sign, etc. Be creative!
# 
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - features -> the images pixel values, (width, height, channels)
# - labels -> the label of the traffic sign
# - sizes -> the original width and height of the image, (width, height)
# - coords -> coordinates of a bounding box around the sign in the image, (x1, y1, x2, y2)

# In[37]:

# Load pickled data
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


# In[38]:

### To start off let's do a basic data summary.

# TODO: number of training examples
n_train = X_train.shape[0]

# TODO: number of testing examples
n_test = X_test.shape[0]

# TODO: what's the shape of an image?
image_shape =X_train.shape[1:3]

# TODO: how many classes are in the dataset
n_classes = max(y_train)+1

n_channel = X_train.shape[3]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of channels =", n_channel)
print("Number of classes =", n_classes)


# In[8]:

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
"""
get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (64, 64) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

examples_per_class=10
n_classes_display=10
classes_start=33  # choose the range of image classes to display
for cls_idx, cls in enumerate(range(classes_start,classes_start+n_classes_display)):
    idxs = np.where((y_train == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerateran(idxs):
        plt.subplot(n_classes_display, examples_per_class, cls_idx * examples_per_class + i +1)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')

            
"""

# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Your model can be derived from a deep feedforward net or a deep convolutional network.
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# In[39]:

### Preprocess the data here.
### Feel free to use as many code cells as needed.


# ### Question 1 
# 
# _Describe the techniques used to preprocess the data._

# **Answer:**
# Normalisation by taking average of all training samples and extract it from all training, validation and test samples. 

# In[40]:

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img

### Generate data additional (if you want to!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.
hist,bin_edges=np.histogram(y_train, n_classes)

# add extra samples to make the balance the distribution
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
        #copy=transform_image(original,10,5,5)
        copy=original
        X_train_extra[i]=np.reshape(copy,(1,32,32,3))
        y_train_extra[i]=np.array([cls])
        i+=1

X_train_extra=np.vstack((X_train, X_train_extra))
y_train_extra=np.hstack((y_train, np.array(y_train_extra)))

"""
filehandler = open("train3.p","wb")
new_train={}
new_train['features']=X_train_extra
new_train['labels']=y_train_extra
pickle.dump(new_train,filehandler)
filehandler.close()
new_train=[]
"""
plt.subplot(2,1,1)
plt.hist(y_train,n_classes)
plt.title("Distribution of training classes")
plt.subplot(2,1,2)
plt.hist(y_train_extra,n_classes)
plt.title("Distribution of test classes")
plt.show()

X_train_sub, X_valid_sub, y_train_sub, y_valid_sub=train_test_split(
X_train_extra, y_train_extra, test_size=0.02, random_state=0)
X_train_extra=[]
y_train_extra=[]
X_mean=np.mean(X_train_sub, axis=0, dtype=np.float32)
X_train_sub=X_train_sub.astype(np.float32)-X_mean
X_test=X_test.astype(np.float32)-X_mean
"""
X_train_sub, X_valid_sub, y_train_sub, y_valid_sub=train_test_split(
X_train, y_train, test_size=0.02, random_state=0)

X_mean=np.mean(X_train_sub, axis=0, dtype=np.float32)
X_train_sub=X_train_sub.astype(np.float32)-X_mean
X_test=X_test.astype(np.float32)-X_mean
"""
#print("n_train_sub",n_train_sub)
#print("n_train",n_train)
#print(range(n_train_sub,n_train))

# ### Question 2
# 
# _Describe how you set up the training, validation and testing data for your model. If you generated additional data, why?_

# **Answer:**

# In[58]:

### Define your architecture here.
### Feel free to use as many code cells as needed.
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

image_size=X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
x=tf.placeholder(tf.float32,[None,image_shape[0],image_shape[1],n_channel], name="x")
y_=tf.placeholder(tf.int64,[None],name="y_")
keep_prob=tf.placeholder(tf.float32)
downsample=1
# Convolutional layer 1
ksize=3
n_features1=32
x_image=tf.reshape(x,[-1,image_shape[0],image_shape[1],n_channel])
W_conv1=weight_variables([ksize,ksize,n_channel,n_features1])
b_conv1=bias_variables([n_features1])
h_conv1=conv2d(x_image,W_conv1)+b_conv1
# Add batch normalisation
scale1 = tf.Variable(tf.ones([n_features1]))
beta1 = tf.Variable(tf.zeros([n_features1]))
batch_mean1, batch_var1=tf.nn.moments(h_conv1,[0])
h_conv1_bn= tf.nn.batch_normalization(h_conv1,batch_mean1,batch_var1,beta1,scale1,1e-9)

h_conv1_relu=tf.nn.relu(h_conv1_bn)
h_conv1_drop=tf.nn.dropout(h_conv1_relu,keep_prob)
h_pool1=max_pool(h_conv1_drop)

downsample*=2
# Convolutional layer 2
ksize=3
n_features2=64
W_conv2=weight_variables([ksize,ksize,n_features1,n_features2])
b_conv2=bias_variables([n_features2])
h_conv2=conv2d(h_pool1,W_conv2)+b_conv2

# Add batch normalisation
scale2 = tf.Variable(tf.ones([n_features2]))
beta2 = tf.Variable(tf.zeros([n_features2]))
batch_mean2, batch_var2=tf.nn.moments(h_conv2,[0])
h_conv2_bn= tf.nn.batch_normalization(h_conv2,batch_mean2,batch_var2,beta2,scale2,1e-9)

h_conv2_relu=tf.nn.relu(h_conv2_bn)
h_conv2_drop=tf.nn.dropout(h_conv2_relu,keep_prob)
h_pool2=max_pool(h_conv2_drop)
downsample*=2

# Fully connected layer
n_fc_neurons=1024
fc_input_dim=image_shape[0]/downsample
W_fc1=weight_variables([fc_input_dim*fc_input_dim*n_features2,n_fc_neurons])
b_fc1=bias_variables([n_fc_neurons])
h_pool2_flat=tf.reshape(h_pool2,[-1,fc_input_dim*fc_input_dim*n_features2])
h_pool_fc1=tf.matmul(h_pool2_flat,W_fc1)+b_fc1
# Add batch normalisation
scale_fc1 = tf.Variable(tf.ones([n_fc_neurons]))
beta_fc1 = tf.Variable(tf.zeros([n_fc_neurons]))
batch_mean_fc1, batch_var_fc1=tf.nn.moments(h_pool_fc1,[0])
h_fc1_bn= tf.nn.batch_normalization(h_pool_fc1,batch_mean_fc1,batch_var_fc1,beta_fc1,scale_fc1,1e-9)
h_fc1=tf.nn.relu(h_fc1_bn)
# Add dropout
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

# Output layer
W_fc2=weight_variables([n_fc_neurons,n_classes])
b_fc2=weight_variables([n_classes])
y=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

# define loss
cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_))
# define training
learning_rate=tf.placeholder(tf.float32)
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)   

### Calculate accuracy
### 
correct_prediction=tf.equal(tf.argmax(y,1),y_)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
"""
def calc_accuracy(n_batch=100, X, y):
    n_inter=y.shape[0]/n_batch    
    acc=0
    for i in range(n_inter):
	batch_start=i*batch_size
        batch_stop=(i+1)*batch_size
	acc+=sess.run(correct_prediction,feed_dict={x:X[batch_start:batch_stop],
y_:y[batch_start:batch_stop], keep_prob:1.0})		
	t_acc/=n_inter
	train_acc+=[t_acc]
"""


# ### Question 3
# 
# _What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
# ](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._
# 

# **Answer:**

# In[59]:

### Train your model here.
### Feel free to use as many code cells as needed.

n_epoch=20
batch_size=100
n_inter=X_train_sub.shape[0]/batch_size

train_acc, train_loss=[],[]
valid_acc = []
learn_rate=5e-3
decay_rate=0.8
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(n_epoch):
        epoch_loss=0
        epoch_start_t=time.clock()
        for i in range(n_inter):
	    batch_start=i*batch_size
            batch_stop=(i+1)*batch_size
            batch_x=X_train_sub[batch_start:batch_stop]
            batch_y=y_train_sub[batch_start:batch_stop]
            _,batch_loss=sess.run([train_step,cross_entropy], feed_dict={x:batch_x, y_:batch_y,keep_prob:0.7,learning_rate:learn_rate})
            epoch_loss+=batch_loss
        train_loss+=[epoch_loss/batch_size]
	learn_rate*=decay_rate
	# Validation accuracy
	n_batch=500
	n_inter=y_train_sub.shape[0]/n_batch    

        v_acc=sess.run(accuracy,feed_dict={x:X_valid_sub, y_:y_valid_sub, keep_prob:1.0})
	valid_acc+=[v_acc]

	# Test accuracy, divide into batch to fit into GPU memory
	n_batch=500
	n_inter=y_train_sub.shape[0]/n_batch    
	t_acc=0
	for i in range(n_inter):
	    batch_start=i*batch_size
            batch_stop=(i+1)*batch_size
	    t_acc+=sess.run(accuracy,feed_dict={x:X_train_sub[batch_start:batch_stop],
y_:y_train_sub[batch_start:batch_stop], keep_prob:1.0})		
	t_acc/=n_inter
	train_acc+=[t_acc]

        epoch_stop_t=time.clock()
        print("Epoch %d, train accuracy=%.5f, valid accuracy=%.5f loss=%.2f elapsed time=%.2f s"%(epoch,t_acc,v_acc,epoch_loss/batch_size,epoch_stop_t-epoch_start_t))    

    # Test trained model
    n_batch=500
    n_inter=int(n_test/n_batch)
    acc=0
    for i in range(n_inter):
        batch_start=i*batch_size
        batch_stop=(i+1)*batch_size
    	acc+=sess.run(accuracy,feed_dict={x:X_test[batch_start:batch_stop],y_:y_test[batch_start:batch_stop], keep_prob:1.0})        
        #print("id=%d acc=%.5f"%(i,acc))
    print("\nTest accuracy=%f"%(acc/n_inter))

   
# Plot
plt.figure()
epoch=np.arange(len(train_loss))
plt.subplot(2,1,1)
plt.plot(epoch, train_loss)
plt.xlabel('epoch')
plt.ylabel('training loss')

plt.subplot(2,1,2)
plt.plot(epoch, train_acc, epoch, valid_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Training','Validation'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

# ### Question 4
# 
# _How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_
# 

# **Answer:**

# ### Question 5
# 
# 
# _What approach did you take in coming up with a solution to this problem?_

# **Answer:**

# ---
# 
# ## Step 3: Test a Model on New Images
# 
# Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# In[3]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Question 6
# 
# _Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It would be helpful to plot the images in the notebook._
# 
# 

# **Answer:**

# In[4]:

### Run the predictions here.
### Feel free to use as many code cells as needed.


# ### Question 7
# 
# _Is your model able to perform equally well on captured pictures or a live camera stream when compared to testing on the dataset?_
# 

# **Answer:**

# In[ ]:

### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.


# ### Question 8
# 
# *Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)*
# 

# **Answer:**

# ### Question 9
# _If necessary, provide documentation for how an interface was built for your model to load and classify newly-acquired images._
# 

# **Answer:**

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# In[ ]:



