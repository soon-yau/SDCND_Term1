import numpy as np
import os
import pickle
import cv2
import csv
import pandas as pd
import argparse

# List directories of video footages to include into CNN training sets
train_dirs=['data_all','track1_center1','track1_recovery1','data2_1']
#train_dirs=['track1_center1']

# resize image to 80x160 and crop out the top 20 rows
# convert colour space to YUV
# normalise to within plus minus 0.5
def process_image(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img=cv2.resize(img,(160,80),interpolation=cv2.INTER_AREA)  
    img=img[20:,:,:]
    img=(img/255.0)-0.5    
    return img
##############################################################

if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Driving Lesson')
    #parser.add_argument('datadir', type=str, help='path of img files')
    #args = parser.parse_args()
    #data_path=args.datadir
    pickle_path=os.getcwd()
    train={}

    for subdir in train_dirs:
        data_path=os.getcwd()+'/data/'+subdir+'/'
        img_path=data_path+"IMG/"
    
        nb_frames=(len(pd.read_csv(data_path+'driving_log.csv')))+1
        print(nb_frames)
        ctr_frames=np.empty((nb_frames,80-20,160,3),np.float32)
        steerings=np.empty((nb_frames),np.float32)
    
        with open(data_path+'driving_log.csv','rt') as csvfile:
            readf=csv.reader(csvfile)    
            #readf.next()
            for i, row in enumerate(readf):
                [ctr_fname, left_fname, right_fname, steering, throttle, brake, speed]=row
                # process center image
                temp=ctr_fname.split('\\')
                img=cv2.imread(img_path+temp[-1])
                ctr_frames[i]=process_image(img)
                steerings[i]=steering
        if not train:
            train['ctr_frames']=ctr_frames
            train['steerings']=steerings
        else:
            train['ctr_frames']=np.vstack((train['ctr_frames'],ctr_frames))
            train['steerings']=np.hstack((train['steerings'],steerings))

    print(train['ctr_frames'].shape)
    print(train['steerings'].shape)
    # Dump into pick files
    with open(pickle_path+'/train.p', 'wb') as f:
        pickle.dump(train, f, protocol=2)
	

