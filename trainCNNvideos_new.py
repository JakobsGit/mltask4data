# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 19:14:26 2018

@author: alexa
"""

import os
import tensorflow as tf	
import numpy as np
#from tf_utils import input_fn_from_dataset,input_fn_frame_from_dataset,save_tf_record,prob_positive_class_from_prediction
#from get_data import get_videos_from_folder,get_target_from_csv
from utils import save_solution

### own imports ###

from tensorflow import keras
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_auc_score

### own imports ###

#dir_path = os.path.dirname(os.path.realpath(__file__))
#train_folder = os.path.join(dir_path,"../train/")
#test_folder = os.path.join(dir_path,"../test/")

#train_target = os.path.join(dir_path,'../train_target.csv')
#my_solution_file = os.path.join(dir_path,'../solution.csv')

#tf_record_dir = os.path.join(dir_path, '..','tf_records')
#os.makedirs(tf_record_dir, exist_ok=True)

#tf_record_train = os.path.join(tf_record_dir, 'train' + '.tfrecords')
#tf_record_test = os.path.join(tf_record_dir, 'test' + '.tfrecords')

#if not os.path.exists(tf_record_train):
#	x_train = get_videos_from_folder(train_folder)
#	y_train = get_target_from_csv(train_target)
#	save_tf_record(x_train,tf_record_train,y = y_train)

#if not os.path.exists(tf_record_test):
#	x_test = get_videos_from_folder(test_folder)
#	save_tf_record(x_test,tf_record_test)	

### own code ###

# get data
x_train = np.load("x_train.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')    
y_train = np.load("y_train.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')    
x_test = np.load("x_test.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')    
#x_train = get_videos_from_folder(train_folder)
#y_train = get_target_from_csv(train_target)
#x_test = get_videos_from_folder(test_folder)

# get videos
X_train = np.zeros((len(x_train),210,100,100))
X_test = np.zeros((len(x_test),210,100,100))
for i in range(0,len(x_train)):
    X_train[i,0:len(x_train[i]),:,:] = x_train[i]
for i in range(0,len(x_test)):
    X_train[i,0:len(x_test[i]),:,:] = x_test[i]
X_train_reshape = X_train.reshape(len(X_train),210,100,100,1)
X_test_reshape = X_test.reshape(len(X_test),210,100,100,1)
y_train_encoded = keras.utils.to_categorical(y_train)
  
# 3D CNN for videos
filters = 1
units = 32
rate = 0.5
pool_size = (2, 2, 2)
kernel_size = (3, 3, 3)
strides = (1, 1, 1)
batch_size = 32
epochs = 1
validation_split = 0.1

model = tf.keras.Sequential()
model.add(keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(210, 100, 100, 1)))
model.add(keras.layers.SpatialDropout3D(rate, data_format=None))
model.add(keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(keras.layers.MaxPooling3D(pool_size=pool_size, strides=None, padding='valid', data_format=None))
model.add(keras.layers.Conv3D(filters=2*filters, kernel_size=kernel_size, strides=strides, padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
#model.add(keras.layers.SpatialDropout3D(rate, data_format=None))
model.add(keras.layers.Conv3D(filters=2*filters, kernel_size=kernel_size, strides=strides, padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(keras.layers.MaxPooling3D(pool_size=pool_size, strides=None, padding='valid', data_format=None))
model.add(keras.layers.Conv3D(filters=4*filters, kernel_size=kernel_size, strides=strides, padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
#model.add(keras.layers.SpatialDropout3D(rate, data_format=None))
model.add(keras.layers.Conv3D(filters=4*filters, kernel_size=kernel_size, strides=strides, padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(keras.layers.MaxPooling3D(pool_size=pool_size, strides=None, padding='valid', data_format=None))
model.add(keras.layers.Flatten(data_format=None))
model.add(keras.layers.Dense(units=2*units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
#model.add(keras.layers.Dropout(rate=rate, noise_shape=None, seed=None))
model.add(keras.layers.Dense(units=units, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
#model.add(keras.layers.Dropout(rate=rate, noise_shape=None, seed=None))
model.add(keras.layers.Dense(units=2, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
history = model.fit(x=X_train_reshape, y=y_train_encoded, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=None, validation_split=validation_split, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.savefig("accuracy.png")
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig("loss.png")
plt.show()

#y_predict_train = model.predict(x=X_train_reshape, batch_size=batch_size, verbose=0, steps=None)
#roc_auc_train = roc_auc_score(y_train, y_predict_train[:,1])

# make prediction
y_predict_video = model.predict(x=X_test_reshape, batch_size=batch_size, verbose=0, steps=None)
y_predict = y_predict_video[:,1]

### own code ###

save_solution(my_solution_file,y_predict)