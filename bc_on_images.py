#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:24:53 2019

@author: vinit
"""

# dataset is like
import glob
import cv2
from keras.preprocessing.image import img_to_array
from sklearn.utils import shuffle
import pandas as pd

def preprocess_data():
    image_data = []
    labels = []
    train_normal_paths = glob.glob("normal/*.bmp")
    train_abnormal_paths = glob.glob("abnormal/*bmp")
    data = train_normal_paths + train_abnormal_paths
    data = shuffle(data)
    for image in data:
        img = cv2.imread(image)
        img = cv2.resize(img,(224,224))
        img = img_to_array(img)
        image_data.append(img)
        
        label = image.split(os.path.sep)[-2]
        label = 0 if label == 'normal' else 1
        labels.append(label)
    return image_data,labels

data,labels = preprocess_data()

import numpy as np
data = np.array(data,dtype = 'float')/255.0
labels = np.array(labels)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,labels,random_state = 42,test_size = 0.25)

from keras.utils import to_categorical
y_train = to_categorical(y_train,num_classes=2)
y_test = to_categorical(y_test,num_classes=2)

from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(rotation_range= 30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense,Convolution2D
from keras.activations import relu,softmax
from keras.layers.core import Dropout,Flatten,Activation
from keras.layers.convolutional import MaxPooling2D,ZeroPadding2D
from keras.activations import softmax
def cnn_create():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape = (224,224,3)))
    model.add(Convolution2D(64,(3,3),activation=relu))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,(3,3),activation=relu))
    model.add(MaxPooling2D((2,2),strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,(3,3),activation=relu))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,(3,3),activation=relu))
    model.add(MaxPooling2D((2,2),strides=(2,2)))    

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,(3,3),activation=relu))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,(3,3),activation=relu))
    model.add(ZeroPadding2D((1,1),input_shape = (32,32,3)))
    model.add(Convolution2D(256,(3,3),activation=relu))
    model.add(MaxPooling2D((2,2),strides=(2,2)))    
       
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,(3,3),activation=relu))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,(3,3),activation=relu))
    model.add(ZeroPadding2D((1,1),input_shape = (32,32,3)))
    model.add(Convolution2D(512,(3,3),activation=relu))
    model.add(MaxPooling2D((2,2),strides=(2,2)))      
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,(3,3),activation=relu))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,(3,3),activation=relu))
    model.add(ZeroPadding2D((1,1),input_shape = (32,32,3)))
    model.add(Convolution2D(512,(3,3),activation=relu))
    model.add(MaxPooling2D((2,2),strides=(2,2)))
    
    model.add(Convolution2D( 4096 , (7 , 7) , activation=relu))
    model.add(Dropout(0.5))
    model.add(Convolution2D( 4096 , (1 , 1) , activation=relu))
    model.add(Dropout(0.5))
    model.add(Convolution2D( 2622 , (1 , 1) ))
    model.add(Flatten())
    model.add(Activation(softmax))
    return model

model = cnn_create()



    