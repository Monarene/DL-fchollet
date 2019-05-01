# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 07:14:43 2018

@author: Michael
"""

#importing neccesart libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import sys

import keras
from keras import layers
from keras.applications import inception_v3
from keras import backend as K


#building  a character level language model with LSTM
path = keras.utils.get_file(
'nietzsche.txt',
origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

#data preprocessing
maxlen = 60
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen,step):
    sentences.append(text[i:maxlen + i])
    next_chars.append(text[maxlen + i])
 
print('number of generated sequences', len(sentences))
chars = sorted(list(set(text)))
this_dict = dict((char,chars.index(char)) for char in chars)

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, this_dict[char]] = 1
    y[i , this_dict[next_chars[i]]] = 1

#build keras model
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape = (maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation = 'softmax'))

optimizer = keras.optimizers.RMSprop(lr = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
model.summary()

#build the temperature temperating factor
def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
    
#write loop to generate text
for epoch in range(60):
    print('Epoch number : ', epoch)
    model.fit(x,y,batch_size = 128,epochs=1)
    pick = np.random.randint(0, len(sentences) - maxlen - 1)
    generated_text = text[pick : pick + maxlen]
    print('Generated text ---- " ', generated_text, ' "')
    
    for temperature in [0.2,0.3]:
        print('-------Temperature: ', temperature)
        sys.stdout.write(generated_text)
        
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t,char in enumerate(generated_text):
                sampled[0, t , this_dict[char]] = 1.
            preds = model.predict(sampled, verbose = False)[0]
            next_index  = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
            
#implemnting google DeepDreams in keras
K.set_learning_phase(0)
model = inception_v3.InceptionV3(weights ='imagenet', include_top = False)
model.summary()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    













