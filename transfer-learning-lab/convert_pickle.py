#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:37:59 2016
This script is to convert pickle format from Python 3.x to Python 2.x
by changing the pickle protocol to 3
@author: sc15770
"""
import os
import pickle

data_path=os.getcwd()
training_file = data_path+"/bottleneck_vgg/vgg_traffic_100_bottleneck_features_train.p"
testing_file = data_path+"/bottleneck_vgg/vgg_traffic_bottleneck_features_validation.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

training2_file = data_path+"/bottleneck_vgg/vgg_traffic_100_bottleneck_features_train.p"
testing2_file = data_path+"/bottleneck_vgg/vgg_traffic_bottleneck_features_validation.p"    

with open(training2_file, "wb") as f:
    pickle.dump(train, f, protocol=2)
    
with open(testing2_file, "wb") as f:
    pickle.dump(test, f, protocol=2)    
   
