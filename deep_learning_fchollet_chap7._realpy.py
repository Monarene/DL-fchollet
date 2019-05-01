# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:05:17 2018

@author: Michael
"""

#importing neccessary libraires
import keras
from keras import Input, layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model

#making the dataset
x_train = np.random.random((1000,64))
y_train = np.random.random((1000,32))

#building the model using keras functional API
input_tensor = Input(shape=(64,))
layer1 = layers.Dense(32, activation='relu')(input_tensor)
layer2 = layers.Dense(32, activation='relu')(layer1)
output_tensor = layers.Dense(32, activation='softmax')(layer2)
model = Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128)

#building an awesome multi-input model
text_size_vocab = 10000
question_size_vocab = 500
answer_size_vocab = 500

text_input = Input((None,), dtype='int32', name='text')
embedding_layer = layers.Embedding(64, text_size_vocab)(text_input)
encoded_text = layers.LSTM(32)(embedding_layer)















































































