# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 19:16:16 2018

@author: Michael
"""
#importing relevancies
import matplotlib as plt
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

#data preprocessing
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape((60000,28,28,1))
train_images= train_images.astype('float32')/255
test_images=test_images.reshape((10000,28,28,1))
test_images=test_images.astype('float32')/255
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#building Covnets professionally
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))   #model.summary  helps to understand basic properties of the model 
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5,batch_size=64)

#evaluating the model on the training set
test_loss,test_acc=model.evaluate(test_images,test_labels)

#ended

#Solving the cat and dog challenge using the deep learning book
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout

#I didnt know i could the code for this but here goes 
import os,shutil

original_dir = 'Kaggle_original_data'
base_dir= 'cats_and_dogs_small'
os.mkdir(base_dir)

train_dir=os.path.join(base_dir,'train')
os.mkdir(train_dir)
train_cats_dir=os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)
train_dogs_dir=os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

validation_dir=os.path.join(base_dir,'validation')
os.mkdir(validation_dir)
validation_cats_dir=os.path.join(validation_dir,'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir=os.path.join(validation_dir,'dogs')
os.mkdir(validation_dogs_dir)

test_dir=os.path.join(base_dir,'test')
os.mkdir(test_dir)
test_cats_dir=os.path.join(test_dir,'cats')
os.mkdir(test_cats_dir)
test_dogs_dir=os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)

#to copy the files
fnames=['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(original_dir,fname)
    dst=os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)
    
fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src=os.path.join(original_dir,fname)
    dst=os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src=os.path.join(original_dir,fname)
    dst=os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames=['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(original_dir,fname)
    dst=os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)
    
fnames=['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src=os.path.join(original_dir,fname)
    dst=os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src,dst)

fnames=['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src=os.path.join(original_dir,fname)
    dst=os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)

#setting up the model
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=1e-4),metrics=['acc'])

#time to use an endless generator to generate the images
train_datagen=ImageDataGenerator(rescale=1./255, rotation_range=40,
                                 width_shift_range=0.2,height_shift_range=0.2,
                                 shear_range=0.2,zoom_range=0.2,horizontal_flip=True,
                                 fill_mode='nearest')

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory('cats_and_dogs_small/train',
                                                  target_size=(150,150),
                                                  classes='binary',
                                                  batch_size=20)

test_generator=test_datagen.flow_from_directory('cats_and_dogs_small/validation',
                                                target_size=(150,150),
                                                classes='binary',
                                                batch_size=20)

animal_classifier=model.fit_generator(train_generator,steps_per_epoch=100,
                            epochs=30,validation_data=test_generator,
                            validation_steps=50)


model.save('cat_dog_classifier.h5')

#using a pretrained keras model
#importing and setting the aesome pretrained models
from keras.applications import VGG16
conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))

#importing the neccessary 
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#setup the data-base and build the feature extractor
base_dir='cats_and_dogs_small'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')

batch_size=20
datagen=ImageDataGenerator(1./255)

def extract_features(directory,samples):
    features=np.zeros(shape=(samples,4,4,512))
    labels=np.zeros(shape=(samples))
    generator=datagen.flow_from_directory(directory,
                                          target_size=(150,150),
                                          batch_size=batch_size,
                                          class_mode='binary')
    i=0
    for inputs_data,label_data in generator:
        features_extracted=conv_base.predict(inputs_data)
        features[i*batch_size:(i+1)*batch_size]=features_extracted
        labels[i*batch_size:(i+1)*batch_size]=label_data
        i+=1
        if i*batch_size >= samples:
            break
    return features,labels

train_features,train_labels=extract_features(train_dir,2000)
test_features,test_labels=extract_features(test_dir,1000)
validation_features,validation_labels=extract_features(validation_dir,1000)

#reshaping the feature matrices
train_features=np.reshape(train_features,(2000,4*4*512))
test_features=np.reshape(test_features,(1000,4*4*512))
validation_features=np.reshape(validation_features,(1000,4*4*512))

#building the densely connected layer to plug in the extracted features
model=Sequential()
model.add(Dense(256,activation='relu',input_dim=4*4*512))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer= RMSprop(lr=2e-5),loss="binary_crossentropy",metrics=['acc'])
vgg_extracted=model.fit(train_features,train_labels,epochs=30,batch_size=20,
              validation_data=(validation_features,validation_labels))

#visualizing the training and testing
history=vgg_extracted.history
num_of_epochs=30
epochs=range(1,num_of_epochs + 1)
acc= history['acc']
val_acc=history['val_acc']
loss=history['loss']
val_loss=history['val_loss']

#plotting for accuracy
plt.figure()
plt.title("Accuracy VS Validation Accuracy")
plt.plot(epochs,acc,'ro',label='accuracy')
plt.plot(epochs,val_acc,'r-',label='validation Acccuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

#plotting for loss
plt.figure()
plt.title('training loss VS validation loss')
plt.plot(epochs,loss,'bo',label='loss')
plt.plot(epochs,val_loss,'b-',label='validation_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper left')


#Experimentallly nut not tested on my code, we could use plugging the conv_base as 
# a normal conv neural_network
model_2=Sequential()
model_2.add(conv_base)
model_2.add(Flatten())
model_2.add(Dense(256,activation='relu'))
model_2.add(Dense(1,activation='sigmoid'))

#complementary code
model_2.summary()
print('List of trainable parameters before freezing: ',len(model_2.trainable_weights))
conv_base.trainable=False
print('List of trainable parameters before freezing:',len(model_2.trainable_weights))

#dealing with the data fro the dirctory by using a datagen
train_datagen=ImageDataGenerator(1./255,rotation_range=40,width_shift_range=0.2,
                                 height_shift_range=0.2,zoom_range=0.2,shear_range=0.2,
                                 horizontal_flip=True,fill_mode='nearest')

test_datagen=ImageDataGenerator(1./255)
train_generator=train_datagen.flow_from_directory('cats_and_dogs_small/train',
                                                  batch_size=20,target_size=(150,150),
                                                  class_mode='binary')
validation_generator=test_datagen.flow_from_directory('cats_and_dogs_small/validation',
                                                      batch_size=30,target_size=(150,150),
                                                      class_mode='binary')
model_2.compile(optimizer=RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])
model_2.fit_generator(train_generator,steps_per_epoch=100,epochs=30,
                      validation_data=validation_generator,validation_steps=50)

#Fine tuning; By freezing and unfreezing models
conv_base.summary()

#seting up the layers tobe unfroozen
conv_base.trainable=True
conv_base.layers
set_trainable=False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False

#Then we plug in the new conv_base network like any layrs
model_3 =Sequential()
model_3.add(conv_base)
model_3.add(Flatten())
model_3.add(Dense(256,activation='relu'))
model_3.add(Dense(1,activation='sigmoid'))

#build the generators and the compilers
model_3.compile(optimizer=RMSprop(lr=1e-05),loss='bianry_crossentropy',metrics=['acc'])
the_trainable_vgg=model_3.fit_generator(train_generator,steps_per_epoch=100,epochs=100,
                                        validation_data=validation_generator,
                                        validation_steps=50)

#tryiny to evaluate the model on test_dataset
test_generator=test_datagen.flow_from_directory('cats_and_dogs_small/test',batch_size=20,
                                                target_size=(150,150),class_mode='binary')
test_acc,test_loss=model_3.evaluate_generator(test_generator,steps=50)

#This Chapter is getting really messy
#visualizing the cativation
from keras.preprocessing import image
import numpy as np

img_path='cats_and_dogs_small/test/cats/cat.1700.jpg'
img=image.load_img(img_path,target_size=(150,150))
img_array=image.img_to_array(img)
final_image_array=np.expand_dims(img_array,axis=0)
final_image_array=final_image_array/255
img_tensor=final_image_array

#shoiwng the image on a plot
plt.figure()
plt.imshow(final_image_array[0])
plt.show()

#using model to submit one input and multiple ooutputs
from keras.models import Model
layer_outputs=[layer.output for layer in model.layers[:8]]
activation_model=Model(inputs=model.input,outputs=layer_outputs)
activations=activation_model.predict(img_tensor)

print(activations[0].shape)
first_layer_activation=activations[0]
#visualizing the fourth channel int the first layer(note that there are 32 channels)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

#visualizing the seventh layernof the first channel
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
#unfinished

#Visualizing Convnets filters
#deeriving the loss of the filter
from keras.applications import VGG16
from keras import backend as K

model=VGG16(weights='imagenet',include_top=False)
filter_index=0
layer_name='block3_conv1'

layer_output=model.get_layer(layer_name).output
loss=K.mean(layer_output[:,:,:,filter_index])

#finding the gradient and then doing gradient normalization with L2
grads=K.gradients(loss,model.input)[0]
grads/=(K.sqrt(K.mean(K.square(grads))) + 1e-5)

#defining a function to deriive loss and gradient values and testing that function
loss_grad = K.function([model.input],[loss,grads])

#blah blah blah
input_img=np.random.random((1,150,150,3))*20 +128.

step=1.
for i in range(40):
    loss_values,grad_values=loss_grad([input_img])
    input_img+=grad_values*step
    
#make the values RGB values
def demystify(x):
    x-=x.mean()
    x/=(x.std() + 1e-5)
    x*=0.1
    
    x+=0.5
    x=np.clip(x,0,1)
    
    x*=255
    x=np.clip(x,0,255).astype('uint8')
    return x

demystify(input_img)
    
#one big function to encapsulate all we ahve done
def generateprocedure(layer_name,filter_index,size=150):
    layer_output=model.get_layer(layer_name).output
    loss=K.mean(layer_output[:,:,:,filter_index])
    grads=K.gradients(loss,model.input)[0]
    grads/=(K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate=K.function([model.input],[loss,grads])
    input_img=np.random.random((1,size,size,3)) *20 +128.
    step= 1.
    for i in range(40):
        loss_values,grad_values=iterate([input_img])
        input_img+=grad_values*step
    img=input_img[0]
    demystify(img)
    return img

#plotting the filter for the 
plt.imshow(generateprocedure('block3_conv1',0))

#remains code to generate a grid of all the filter responses in the layer
#importing the libraries necessary for this work
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np

#converting the image to an array
img=image.load_img('african_elephant.jpg',target_size=(224,224))
img=image.img_to_array(img)
img_tensor=np.expand_dims(img,axis=0)
img_tensor=preprocess_input(img_tensor)

#bing in VGG16 and predict and et all
model=VGG16(weights='imagenet')
preds=model.predict(img_tensor)
print('Predicted', decode_predictions(preds, top=3)[0])







