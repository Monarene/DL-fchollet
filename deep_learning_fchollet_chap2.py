# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:47:23 2018

@author: Michael
"""
#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.datasets import load_digits

#importing datasets and preprocessing part
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000,28*28)) #reshaping the data to how pythom wants to see it
test_images = test_images.reshape((10000,28*28))
train_images= train_images.astype('float32')/255
test_images= test_images.astype("float32")/255
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#writing the ANN basically 512 and 10
classifier = Sequential()
classifier.add(Dense(512,activation='relu', input_shape=(28*28,))) #I kinda noticed input_shape is basically the number of cols them the comma 
classifier.add(Dense(10,activation='softmax'))

#adding the compiler and fitting the data
classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit(train_images,train_labels,epochs=5,batch_size=128)

#the tessting phase
test_loss,test_accuracy=classifier.evaluate(test_images,test_labels)
print("This is the test accuracy", test_accuracy)

#understanding the chapoter
(rtrain_images,rtrain_labels),(rtest_images,rtest_labels)=mnist.load_data()
digit = rtrain_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
a_slice= rtrain_images[10:100] #a type of slicing
b_slice=rtrain_images[10:100,:,:] #basically gives the same result as above






















