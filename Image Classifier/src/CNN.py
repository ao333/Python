#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:53:32 2018

@author: wuyuxiang
"""
from __future__ import print_function
import matplotlib.pyplot as plt
from src.utils.data_utils import get_FER2013_data
from plot import plot_acc_loss
import keras
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#load data from file
data = get_FER2013_data()
x_train,y_train,x_test,y_test,x_val,y_val = data['X_train'], data['y_train'],data['X_test'],data['y_test'],data['X_val'],data['y_val'] 

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)

y_train = y_train.reshape(y_train.shape[0], 1)

x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

y_test = y_test.reshape(y_test.shape[0], 1)

x_val = x_val.reshape(x_val.shape[0], 48, 48, 1)

y_val = y_val.reshape(y_val.shape[0], 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

input_shape = (48, 48, 1)
batch_size = 128
num_classes = 7
epochs = 150
VERBOSE = 1
VALIDATION_SPLIT = 0.2
#normalize the data
x_train /= 255
x_test /= 255
x_val /= 255



#convert a class vector (integers) to binary class matrix
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
#we can use add to manipulate the model later
model = Sequential()

#the 1st conovolutional layer
model.add(Conv2D(32, kernel_size=(5, 5),
                        border_mode='valid',
                        input_shape=input_shape))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
#the 2nd conovolutional layer
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
#the 3rd conovolutional layer
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
 
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

# Flatten layer, transfer the multi-dimensions to one dimension
model.add(Flatten())
#full connection layer, including 128 neurons
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#contains 10 output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#the sunmmary of the model structure
#model.summary()
#learning process and init the optimizer and model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

#data augumentarion
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

filepath="./models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#train the model
history = model.fit_generator(datagen.flow(x_train, y_train,
                    batch_size=batch_size),
                    samples_per_epoch=x_train.shape[0],
                    nb_epoch=epochs,
                    callbacks = [checkpoint],
                    validation_data=(x_val, y_val))



score = model.evaluate(x_test,y_test,verbose = VERBOSE)
#display the socre and accuracy
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_acc_loss(history, epochs)
plt.show()










