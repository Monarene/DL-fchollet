bb#l -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 11:14:42 2018

@author: Michael
"""

#importing thr relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

#loading in the data 
(train_data,train_labels),(test_data,test_labels)= imdb.load_data(num_words=5000)

#vectorizing the data
def vectorizer(sequence,dimension=5000):
    free=np.zeros((len(sequence),dimension))
    for i,data in enumerate(sequence):
        free[i,data]=1.
    return free
x_train=vectorizer(train_data)
X_test=vectorizer(test_data) 

#vectorizing the labels data
y_train=np.asarray(train_labels).astype('float32')j dgb
y_test=np.asarray(test_labels).astype('float32')

#building a model
classifier=Sequential()
classifier.add(Dense(16,activation='relu',input_shape=(5000,)))
classifier.add(Dense(16,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

#inorder to test the model, we must split the data into partial
partial_x_train=x_train[10000:]
partial_y_train=y_train[10000:]
val_x=x_train[:10000]
val_y=y_train[:10000]

#compiling the model
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model=classifier.fit(partial_x_train,partial_y_train,epochs=10,batch_size=512,
               validation_data=[val_x,val_y])

#using the model's dictionary to get datq about the model
history=model.history
history.keys()    #this is to check out the keys in hisory dictionary

#plotting the graph against acc and val_accuracy
#loss
epochs=range(1,11)
loss=history['loss']
val_loss=history['val_loss']
plt.figure()
plt.title("Plotting of loss and val loss against epochs")
plt.plot(epochs,loss,'r^',label="Loss")
plt.plot(epochs,val_loss,'ro', label="Val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper left')

#accuracy
acc=history['acc']
val_acc=history['val_acc']
plt.figure()
plt.title("Plotting of acc and val_acc against epochs")
plt.plot(epochs,acc,'bo',label="Accuracy")
plt.plot(epochs,val_acc,'b',label="Val_Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

#finally evaluate and build the most correct model
#reconstructing the right model
the_model=Sequential()
the_model.add(Dense(16,activation='relu',input_shape=(5000,)))
the_model.add(Dense(16,activation="relu"))
the_model.add(Dense(1,activation="sigmoid"))

#cmpiling and fitting the new model
the_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
the_model=the_model.fit(partial_x_train,partial_y_train,epochs=10,batch_size=512,
              validation_data=[val_x,val_y])
history_new=the_model.history

#plotting the accuracy grpah
#accu
acc=history_new['acc']
val_acc=history_new['val_acc']
plt.figure()
plt.title("Plotting of acc and val_acc against epochs")
plt.plot(epochs,acc,'bo',label="Accuracy")
plt.plot(epochs,val_acc,'b',label="Val_Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

#final model
final_model=Sequential()
final_model.add(Dense(16,activation="relu",input_shape=(5000,)))
final_model.add(Dense(16,activation="relu"))
final_model.add(Dense(1,activation="sigmoid"))

#compiling and fitting model
final_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
final_model.fit(x_train,y_train,epochs=3,batch_size=512,validation_data=[x_test,y_test])
#Surprising Adam and rmsPROP were giving similiar results

#Reuter Wires Problem
from keras.datasets import reuters

#importing and cleaning the data
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=5000)

def vectorizer(sequences, dimension=5000): #vectorizer to vectorize the lists
    follow=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        follow[i, sequence]=1.
    return follow
x_train=vectorizer(train_data)
x_test=vectorizer(test_data)

def label_vectorizer(sequences,dimension=46):
    follower=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        follower[i,sequence]=1.
    return follower
y_train=label_vectorizer(train_labels)
y_test=label_vectorizer(test_labels)

#another way
y_trains=to_categorical(train_labels)
y_tests=to_categorical(test_labels)

#splitting the data to get validation and testing data
partial_x_train=x_train[1000:]
val_x=x_train[:1000]
partial_y_train=y_train[1000:]
val_y=y_train[:1000]

#begin building the model
mclassifier=Sequential()
mclassifier.add(Dense(64,activation='relu',input_shape=(5000,)))
mclassifier.add(Dense(64,activation='relu'))
mclassifier.add(Dense(46,activation='softmax'))

#compiling and fitting the model
mclassifier.compile(optimizer='rmsprop',loss='categorical_crossentropy'
                    ,metrics=['accuracy'])
mmodel=mclassifier.fit(partial_x_train,partial_y_train,epochs=20,batch_size=20,
                validation_data=[val_x,val_y])

#use the datanto plot graphs to test overfitting 
history=mmodel.history
acc=history['acc']
val_acc=history['val_acc']
loss=history['loss']
val_loss=history['val_loss']
epochs=range(1,len(acc)+1)

#for accuracy
plt.figure("Accuracy")
plt.plot(epochs,acc,'b-',label="Training Accuracy")
plt.plot(epochs,val_acc,'bo',label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuaracy")
plt.legend(loc='upper left')
plt.show()

#for loss
plt.figure("Loss")
plt.plot(epochs,loss,'b-',label="Training Loss")
plt.plot(epochs,val_loss,'bo',label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper left')
plt.show()

#building the final model on nine epochs
f_model=Sequential()
f_model.add(Dense(64,activation='relu',input_shape=(5000,)))
f_model.add(Dense(64,activation='relu'))
f_model.add(Dense(46,activation='softmax'))
#fiitting and compiling the model
f_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
f_model.fit(x_train,y_train,epochs=9,batch_size=512)
f_model.evaluate(x_test,y_test)
y_pred=f_model.predict(x_test)

#Deep learning regression problem
#Boston house problem
#importing relevancies
import numpy as np
import matplotlib as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing

#import and Normalizing the dataset by hand
(train_data,train_target),(test_data,test_target)=boston_housing.load_data()

#normalizing the data by hand
mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std

#building the model using a function
def build_model():
    model=Sequential()
    model.add(Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

#writing the k-fold cross validation by hand
k=4
val_samples=len(train_data) // 4
num_epochs=100
all_scores=[]
all_accuracy=[]

for i in range(k):
    print("Prrocessing Fold "+ str(i))
    val_train=train_data[i*val_samples:(i+1)*val_samples]
    val_test=train_target[i*val_samples:(i+1)*val_samples]
    
    partial_train=np.concatenate([train_data[:i*val_samples],
                                  train_data[(i+1)*val_samples:]],axis=0)
    partial_target=np.concatenate([train_target[:i*val_samples],
                                   train_target[(i+1)*val_samples:]],axis=0)
    
    model=build_model()
    model.fit(partial_train,partial_target,batch_size=1,epochs=num_epochs,
              verbose=0)
    val_mse,val_mae=model.evaluate(val_train,val_test,verbose=0)
    all_scores.append(val_mse)
    all_accuracy.append(val_mae)

#writing another k-fold validation algorithm with history ishh
num_epochs=500
k=4
val_samples=len(train_data)//4
all_history=[]
for i in range(k):
    print("Processing Fold ",i )
    val_train=train_data[i*val_samples:(i+1)*val_samples]
    val_target=train_target[i*val_samples:(i+1)*val_samples]
    
    partial_train=np.concatenate([train_data[:i*val_samples],
                                  train_data[(i+1)*val_samples:]],axis=0)
    partial_target=np.concatenate([train_target[:i*val_samples],
                                   train_target[(i+1)*val_samples:]], axis=0)
    
    model=build_model()
    modeler=model.fit(partial_train,partial_target,epochs=num_epochs,batch_size=2,
                      verbose=False,validation_data=[val_train,val_target])
    model_history=modeler.history
    mae=model_history['val_mean_absolute_error']
    all_history.append(mae)

#function to find the average of the means
average_history=[np.mean([x[i] for x in all_history]) for i in range(num_epochs)]

#plotting the base graph of the average of all the four validation_scores
plt.figure("The good plot")
plt.plot(range(1,len(average_history)+1),average_history)
plt.xlabel("Epochs")
plt.ylabel("Mean_Absolute_Accuracy")
plt.show()

#we would smoothen the curve by removing the first 10 elements and 
#replace each point with an exponential moving average
def smoothner(points,pf=0.9):
    smooth_average=[]
    for point in points:
        if smooth_average:
            previous=smooth_average[-1]
            add= previous*pf + point*(1-pf)
            smooth_average.append(add)
        else:
            smooth_average.append(point)
    return smooth_average

smooth_curve=smoothner(average_history[10:])

#plot new graph to determine where we stand
plt.figure("The Better Book")
plt.plot(range(1,len(smooth_curve)+1),smooth_curve)
plt.xlabel("Epochs")
plt.ylabel("Mean_absolute_error")
plt.show()


























