#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:33:47 2019

@author: ghuiii
"""
################## import the libraries #################################
import pandas as pd
import numpy as np
import keras
#from keras.applications.vgg16 import VGG16
import os
import glob
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array 
from PIL import Image

##################### preparing the dataset directories #################
data_dir = "flowers-recognition/flowers"
daisy = glob.glob(os.path.join(data_dir,"daisy/*.jpg"))[:500]
dandelion = glob.glob(os.path.join(data_dir,"dandelion/*.jpg"))[:500]
rose = glob.glob(os.path.join(data_dir,"rose/*.jpg"))[:500]
sunflower = glob.glob(os.path.join(data_dir,"sunflower/*.jpg"))[:500]
tulip = glob.glob(os.path.join(data_dir,"tulip/*.jpg"))[:500]


######extract the feature or convert image into array the data ##########

def img_array(image):
    img = img_to_array(Image.open(image).resize((100,100))).flatten()
    return img

daisy_data = []
dandelion_data = []
rose_data = []
sunflower_data = []
tulip_data = []
data = {0:daisy,1:dandelion,2:rose,3:sunflower,4:tulip}
for i in data:
    for image in range(0,len(data[i])):
        if data[i] == daisy:
            daisy_data.append(list(img_array(daisy[image])))
        elif data[i] == dandelion:
            dandelion_data.append(list(img_array(dandelion[image])))
        elif data[i] == rose:
            rose_data.append(list(img_array(rose[image])))
        elif data[i] == sunflower:
            sunflower_data.append(list(img_array(sunflower[image])))
        else:
            tulip_data.append(list(img_array(tulip[image])))
#print(daisy_data[0])
#print(len(daisy_data))
#print(len(dandelion_data))
#print(len(rose_data))
#print(len(sunflower_data))
#print(len(tulip_data))

######################### prepairing dataset ###############################
def data_pre(ndata,i):
    new_data = pd.DataFrame(ndata)
    new_target = pd.DataFrame(np.repeat(i,len(ndata)))
    return new_data,new_target


daisy_data,daisy_target = data_pre(daisy_data,0)
dandelion_data,dandelion_target = data_pre(dandelion_data,1)
rose_data,rose_target = data_pre(rose_data,2)
sunflower_data,sunflower_target = data_pre(sunflower_data,3)
tulip_data,tulip_target = data_pre(tulip_data,4)
#print(tulip_data.head())
#print(tulip_target.head())

###################### concate the all class dataset #########################

final_data = pd.concat([daisy_data,dandelion_data,rose_data,sunflower_data,tulip_data],axis = 0)
final_target = pd.concat([daisy_target,dandelion_target,rose_target,sunflower_target,tulip_target],axis = 0)
#print(final_data.shape)
#print(final_target.shape)
final_target.columns = ["Label"]
df = pd.concat([final_data,final_target],axis=1)
#print(df.head())
#print(df.shape)


###################### Shuffle the dataset ####################################
from sklearn.utils import shuffle
df = shuffle(df)
#print(df.head())
y = df["Label"]
X = df.drop('Label',axis=1)
#print(y.head())
#print(X.head())
################################################################################
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 42)

def normalize(data):
	d = np.array(data)/255
	return d

X_train = normalize(X_train)
print(X_train.shape)
print(X_train)
X_test = normalize(X_test)

from keras.utils import np_utils
def np_cat(data):
	c = np_utils.to_categorical(data)
	return c

y_train = np_cat(y_train)
y_test = np_cat(y_test)
#print(y_train)
###################################################################################

X_train = X_train.reshape(X_train.shape[0],100,100,3)
X_test = X_test.reshape(X_test.shape[0],100,100,3)

####################################################################################
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
from keras.callbacks import TensorBoard
#def model_new():
model=Sequential()
model.add(Conv2D(64,kernel_size=5,padding='same',input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(128,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(150,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(600,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#return model

board = TensorBoard(log_dir = "log/",write_graph=True, write_images=True)
final_model=model.fit(X_train,y_train,batch_size=16,epochs=40,validation_data=(X_test,y_test),verbose=1,callbacks = [board])

'''
model_json = model.to_json()
with open('model.json','w') as json_file:
	json_file.write(model_json)
model.save_weights('model.h5')
print("saved model to disk")
#final_model.save("vinit_flower.h5")

json_file = open('model.h5','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print("loaded model from disk")

'''

import matplotlib.pyplot as plt
accuracy = final_model.history['acc']
valid_accuracy = final_model.history['val_acc']
epoch = range(len(accuracy))
plt.plot(epoch,accuracy,'bo',label = "training accuracy")
plt.plot(epoch,valid_accuracy,'bo',label = "validation accuracy")
plt.title("Training_Testing_Accuracy")
plt.legend()
plt.show()





