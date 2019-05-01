# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 01:12:04 2018

@author: Michael
"""

#libraies
import numpy as np
import string
import keras
from keras.preprocessing.text import Tokenizer

#a simple Toy example of Tokenizatinn
samples=['The cat sat on the mat.','The dog ate my lunch']
token_index={}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word]=len(token_index) + 1
            
max_length=10
results=np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))
for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index=token_index.get(word)
        results[i,j,index]=1.
        
#string level one hot encoding
characters=string.printable
token_index_str=dict(zip(range(1,len(characters)+1),characters))
max_length=50
results=np.zeros(shape=(len(samples),max_length,max(token_index_str.keys())+1))
for i,sample in enumerate(samples):
    for j,character in enumerate(sample):
        index=token_index_str.get(character)
        results[i,j,index]=1.
        
#using keras Tokenizer to implement word Tokenization
tokenize=Tokenizer(num_words=1000)
tokenize.fit_on_texts(samples)
sequences=tokenize.texts_to_sequences(samples)
token_index_keras=tokenize.texts_to_matrix(samples)
word_index=tokenize.word_index

#building this kind of netwworks for embedding text
import keras
from keras.preprocessing import sequence
from keras.layers import Flatten,Dense,Embedding
from keras.models import Sequential
from keras.datasets import imdb

#start the preprocessing and the building of the network
#embedding_layer=Embedding(1000,64)
max_features=10000
max_len=20
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
x_train=sequence.pad_sequences(x_train,max_len)
x_test=sequence.pad_sequences(x_test,max_len)

model=Sequential()
model.add(Embedding(10000,8,input_length=max_len))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
the_model=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2)


#building a better model creatingg and cleaning the data from scratch
#importing the neccesary libraries
import keras
import os
from keras.layers import Dense,Embedding,Flatten
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

#creating the file direcctoriews and including all that should be included
imbd_dir = 'imbd/text_data'
train = os.path.join(imbd_dir,'train')

labels =[]
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train,label_type)
    for f_name in os.listdir(dir_name):
        if f_name[-4:] == '.txt':
            f = open(os.path.join(dir_name,f_name), encoding = 'utf8')
            texts.append(f.read())   
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

#defining the variables and preprocessing the text
maxlen = 100
validation_samples = 10000
training_samples = 200
max_words = 10000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('the length of the tokens ', len(word_index))

data = pad_sequences(sequences, maxlen = maxlen)
labels = np.asarray(labels)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]

x_val = data[training_samples:training_samples + validation_samples ]
y_val = labels[training_samples:training_samples + validation_samples ]

#setting up the embedding file once 
glove_dir = 'glove_data'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype = 'float32')
    embeddings_index[word] = coefs
f.close()

print('the length of the word embeddings are', len(embeddings_index))

#build the embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((max_words , embedding_dim))
for word,i in word_index.items():
    if i < max_words:
        embedding_vector =embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
#building the network structure
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length = maxlen))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

#seetting the first layer in the model with the embedding matrix
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

#compile and fit the model
model.compile('rmsprop',loss = 'binary_crossentropy', metrics = ['acc'])
model_history = model.fit(x_train,y_train,epochs=10,batch_size=10,
          validation_data=(x_val,y_val))
model.save_weights('pretrained_glove_model.h5')



#plotting the visualizations
acc = model_history.history['acc'] 
loss = model_history.history['loss'] 
val_acc = model_history.history['val_acc'] 
val_loss = model_history.history['val_loss'] 

epochs = range(1 , len(loss) +1)
plt.figure()
plt.title('Training Accuracy vs Validation Accuracy')
plt.plot(epochs,acc,'bo',label= 'Accuracy')
plt.plot(epochs,val_acc,'b-',label= 'Validation Accuracy')
plt.legend(loc = 'upper left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.figure()
plt.title('Training Loss vs Validation Loss')
plt.plot(epochs,loss,'ro',label= 'Loss')
plt.plot(epochs,val_loss,'r-',label= 'Validation Loss')
plt.legend(loc = 'upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#training the model without using the oretrained embedding
model_1 = Sequential()
model_1.add(Embedding(max_words, embedding_dim, input_length = maxlen))
model_1.add(Flatten())
model_1.add(Dense(32, activation='relu'))
model_1.add(Dense(1, activation='sigmoid'))

model_1.summary()

model_1.compile('rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
model_1_history=model_1.fit(x_train,y_train, epochs=10, batch_size=32, 
            validation_data = (x_val, y_val))


#plotting the visualizations
acc = model_1_history.history['acc'] 
loss = model_1_history.history['loss'] 
val_acc = model_1_history.history['val_acc'] 
val_loss = model_1_history.history['val_loss'] 

epochs = range(1 , len(loss) +1)
plt.figure('Without using pretrained Word embeddings Accuracy')
plt.title('Training Accuracy vs Validation Accuracy')
plt.plot(epochs,acc,'bo',label= 'Accuracy')
plt.plot(epochs,val_acc,'b-',label= 'Validation Accuracy')
plt.legend(loc = 'upper left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.figure('Without using pretrained Word embeddings Loss')
plt.title('Training Loss vs Validation Loss')
plt.plot(epochs,loss,'ro',label= 'Loss')
plt.plot(epochs,val_loss,'r-',label= 'Validation Loss')
plt.legend(loc = 'upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#oredicting on the test dataset and getting it ready
test_dir = os.path.join(imbd_dir, 'test')
labels_2 = []
texts_2 = []

for label_type in ['neg','pos']:
    dir_name = os.path.join(test_dir, label_type)
    for f_name in sorted(os.listdir(dir_name)):
        if f_name[-4:] == '.txt':
            f = open(os.path.join(dir_name,f_name),encoding='utf8')
            texts_2.append(f.read())
            f.close()
            if label_type == 'neg':
                labels_2.append(0)
            else:
                labels_2.append(1)

sequences = tokenizer.texts_to_sequences(texts_2)
x_test = pad_sequences(sequences, maxlen = maxlen)
y_test = np.asarray(labels_2)

#test the results of the pretrained models on generally new data
model.load_weights('pretrained_glove_model.h5')
model.evaluate(x_test,y_test)

#building a numpy simple RNN layer
timesteps = 100
output_features = 64
input_features = 32

inputs = np.random.random((timesteps,input_features))
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
state_t = np.zeros((output_features,))

successive_outputs = []
for input_t in inputs:
    output = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output)
    state_t = output
final_output_seequence = np.concatenate(successive_outputs, axis = 0)

#running on simple RNNs
model = Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32))
model.summary()

#stacking successive RNN layers for the purpose of seeing the effect
model = Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32, return_sequences = True))
model.add(SimpleRNN(32, return_sequences = True))
model.add(SimpleRNN(32, return_sequences = True))
model.add(SimpleRNN(32))
model.summary()

#building RNN for works 
max_features = 10000
maxlen = 500
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words = max_features)

#pre[aring the data]
print(len(input_train), 'Length of the train sequence')
print(len(input_test), 'Length of the test sequence')
print(input_train.shape, 'the shape of the train input sequence')
print(input_test.shape, 'the shape of the test input sequence')

input_train = pad_sequences(input_train, maxlen = maxlen)
input_test = pad_sequences(input_test, maxlen = maxlen)

print(input_train.shape)

model = Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation = 'relu'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',metrics = ['accuracy'])
model_details = model.fit(input_train, y_train, batch_size = 128, epochs = 10, validation_split = 0.2)

#visualizing this model
acc = model_details.history['acc']
val_acc = model_details.history['val_acc']
loss = model_details.history['loss']
val_loss = model_details.history['val_loss']

epochs = range(1, len(acc)+1)
plt.figure()
plt.plot(epochs, acc, 'bo',label = 'Accuracy')
plt.plot(epochs, val_acc, 'b-', label = 'Validation Accuracy')
plt.legend(loc = 'upper left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(epochs, loss, 'bo',label = 'Loss')
plt.plot(epochs, val_loss, 'b-', label = 'Validation Loss')
plt.legend(loc = 'upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#let us use LSTMs and visualize onn those
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer = 'rmsprop',loss='binary_crossentropy', metrics = ['accuracy'])
model_details = model.fit(input_train, y_train,epochs=10, batch_size=128,
                          validation_split=0.2)

#visualizing this model
acc = model_details.history['acc']
val_acc = model_details.history['val_acc']
loss = model_details.history['loss']
val_loss = model_details.history['val_loss']

epochs = range(1, len(acc)+1)
plt.figure()
plt.plot(epochs, acc, 'bo',label = 'Accuracy')
plt.plot(epochs, val_acc, 'b-', label = 'Validation Accuracy')
plt.legend(loc = 'upper left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(epochs, loss, 'bo',label = 'Loss')
plt.plot(epochs, val_loss, 'b-', label = 'Validation Loss')
plt.legend(loc = 'upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss')















