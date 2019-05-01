# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 23:44:39 2018

@author: Michael
"""

#bringing in the neccessary imports
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.datasets import imdb
from keras import regularizers 

#K-Fold validation description
k=4
number_of_samples=len(data)//4
np.random.shuffle(data)
validation_scores=[]

for i in range(k):
    validation_data=data[number_of_samples*i:number_of_samples*(i+1)]
    training_data=data[:number_of_samples*i] + data[number_of_samples*(i+1):]
    #a description of the algorithm
    model=get_model()
    model.train_model(training_data)
    validation_score=model.evaluate(validation_data)
    validation_scores.append(validation_score)
    
validation_score=np.average(validation_scores)
#after much merrygo-rounding, and hyper-parameter training wwe hinted on some stuff
#then training the entire model on full training_set and optimizing on full test_set 
model.train(data)
test_score=model.evaluate(data)

#Iterated K-Fold iteration

#controlling overfitting using weight regulators
(train_x,train_y), (test_x,test_y)= imdb.load_data(num_words=5000)

#vectorizing the independent value data
def vectorizer(sequence, dimension=5000):
    free=np.zeros((len(sequence),dimension))
    for i,data in enumerate(sequence):
        free[i,data]=1.0
    return free
    
#setting the two variabels
x_train=vectorizer(train_x)
x_test=vectorizer(test_x)    

#setting up the y-variables
y_train=np.asarray(train_y).astype('float32')
y_test=np.asarray(test_y).astype('float32')

#split train_dataset
partial_x_train=x_train[:15000]
partial_y_train=y_train[:15000]
val_x=x_train[15000:]
val_y=y_train[15000:]

#we can begin building the model
model=Sequential()
model.add(Dense(16,activation='relu', input_shape=(5000,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
classifier=model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,
                     validation_data=[val_x,val_y])


#the model built using regularization techniques
model_2=Sequential()
model_2.add(Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu',
                input_shape=(5000,)))
model_2.add(Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model_2.add(Dense(1,activation='sigmoid'))

model_2.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
classifier_2=model_2.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,
                       validation_data=[val_x,val_y])

#comparing the two versions of deep learning code regarding 
history=classifier.history
history_2=classifier_2.history
loss=history['val_loss']
loss_2=history_2['val_loss']
epochs=range(1,21)

#building up some plots
plt.figure()
plt.title("Ordinary Vs L2 Regularized")
plt.plot(epochs,loss,'b+',label='Original Model')
plt.plot(epochs,loss_2,'bo',label='L2 Regularized Model')
plt.xlabel('Epochs')
plt.ylabel("Validation Loss")
plt.legend(loc='upper left')

#sealed
#trying to use the drop out method developed by Geoffrey Hinton
#And then we attempt to maek two comparisons 
#The drop-out regularized Vs the Ordinary
#The drop-out regularized Vs the L2 regularized

model_3=Sequential()
model_3.add(Dense(16,activation='relu',input_shape=(5000,)))
model_3.add(Dropout(0.5))
model_3.add(Dense(16,activation='relu'))
model_3.add(Dropout(0.5))
model_3.add(Dense(1,activation='sigmoid'))

model_3.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
classfier_3=model_3.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,
            validation_data=[val_x,val_y])

history_3=classfier_3.history
loss_3=history_3['val_loss']

#L2 Vs DropOut regularized
plt.figure()
plt.title("L2 regularized Vs DropOut Regularized")
plt.plot(epochs, loss_2,'b+',label='L2 regularized')
plt.plot(epochs, loss_3,'bo',label='Dropout Regularized')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(loc='upper left')

#Original Model Vs DropOut regularized
plt.figure()
plt.title("Original MOdel Vs DropOut Regularized")
plt.plot(epochs, loss,'b+',label='Original Model')
plt.plot(epochs, loss_3,'bo',label='Dropout Regularized')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(loc='upper left')

#In this testCase the dropout algorithm actually performed better





























































