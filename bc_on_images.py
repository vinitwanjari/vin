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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, time
import matplotlib.pyplot as plt
#from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
#from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import os
from keras import optimizers
import warnings
warnings.filterwarnings('ignore')



train_data = pd.read_csv('C:/Users/w638611/Desktop/Extraction_using_image_processing/Mnist_data/mnist_fashion/fashion-mnist_train.csv')
test_data = pd.read_csv('C:/Users/w638611/Desktop/Extraction_using_image_processing/Mnist_data/mnist_fashion/fashion-mnist_test.csv')

#train_data=train_data.sample(n=5000)
#test_data=test_data.sample(n=5000)

train_data.shape #(60,000*785)
test_data.shape #(10000,785)
train_X= np.array(train_data.iloc[:,1:])
test_X= np.array(test_data.iloc[:,1:])
train_Y= np.array (train_data.iloc[:,0]) # (60000,)
test_Y = np.array(test_data.iloc[:,0]) #(10000,)

classes = np.unique(train_Y)
num_classes = len(classes)

# Convert the images into 3 channels
train_X=np.dstack([train_X] * 3)
test_X=np.dstack([test_X]*3)
train_X.shape,test_X.shape

# Reshape images as per the tensor format required by tensorflow
train_X = train_X.reshape(-1, 28,28,3)
test_X= test_X.reshape (-1,28,28,3)

# Resize the images 48*48 as required by VGG16
from keras.preprocessing.image import img_to_array, array_to_img
train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_X])
test_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in test_X])
#train_x = preprocess_input(x)
print(train_X.shape, test_X.shape)


# Normalise the data and change data type
train_X = train_X / 255.
test_X = test_X / 255.
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')


# Converting Labels to one hot encoded format
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)



# Define the parameters for instanitaing VGG16 model. 
IMG_WIDTH = 48
IMG_HEIGHT = 48
IMG_DEPTH = 3
BATCH_SIZE = 8




#  Create base model of VGG16
model = VGG16(weights='imagenet',
                  include_top=False, 
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
                 )
model.summary()

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False
    
#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)
from keras.models import Sequential,Model


# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.summary()

history=model_final.fit(train_X,train_Y_one_hot,batch_size=BATCH_SIZE,epochs=10,validation_data=(test_X,test_Y_one_hot),verbose=2)


    
