# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:06:45 2018

@author: Michael
"""

#importing the f imports 
import tensorflow as tf

#attempting to draw a graph
sess=tf.Session()
my_graph=tf.Graph()
with my_graph.as_default():
    variable=tf.Variable(40,name='navin')
    initialize=tf.global_variables_initializer()

with tf.Session(graph=my_graph) as sees:
    sees.run(initialize)
    print(sees.run(variable))

#making constant in tensorflow
import tensorflow as tf
x=tf.constant(12,dtype='float32') #construction phase
sess=tf.Session()
print(sess.run(x))

#making vairables
y=tf.Variable(x+11)
model=tf.global_variables_initializer()
sess=tf.Session()
sess.run(model)
print(sess.run(y))


#more variables
x=tf.Constant([14,23,40,30])
y=tf.Variable(x*2+100)
model=tf.global_variables_initializer()
sess=tf.Session()
sess.run(model)
sess.run(y)

#creating a placehoder
x=tf.placeholder("float",None)
y=x*10+500
sess=tf.Session()
placeX = sess.run(y, feed_dict={x:[10,20,30,40]})
print(placeX)

#more placeholdes
x=tf.placeholder("float", None)
y=x*10+500
sess=tf.Session()
dataX = [[121,100,81,64],
         [4,5,8,10]]
placeX = sess.run(y,feed_dict={x:dataX })
print(placeX)

#we are baout to use tensorflow to decode an image
image=tf.image.decode_jpeg(tf.read_file("./resources/code_gear.jpg"),channels=3)
sess=tf.InteractiveSession()
shape=sess.run(tf.shape(image))
print(shape)

















