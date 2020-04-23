from __future__ import print_function
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import os
from keras.initializers import Nadam


from keras.optimizers import SGD,Adam
from keras.optimizers import RMSprop
from keras.callbacks.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau


# creating model

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(200,200,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),kernel_initializer='lecun_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='lecun_normal'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), kernel_initializer='lecun_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), kernel_initializer='lecun_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, kernel_initializer='lecun_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax', kernel_initializer='lecun_normal'))

# compiling model
model.compile(optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),loss="categorical_crossentropy",metrics=["accuracy"])

# summary of model
model.summary()






train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        r"Train_images",
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical',shuffle = True)


filepath = r'weight_file.h5'

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss', 
                             verbose=1,
                             save_best_only=True)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0, 
                          patience=3, 
                          verbose=1, 
                          restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=10, 
                              verbose=1, 
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]


model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=100)

# saving model history
model.save(filepath)

# sving model weights
model.save_weights(filepath)