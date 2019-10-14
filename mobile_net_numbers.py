import numpy as np

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL

# define paths for train, valid, and test data
train_path = 'sign_number/train'
valid_path = 'sign_number/valid'
test_path = 'sign_number/test'

train_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(224, 224), batch_size=10)
valid_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path, target_size=(224, 224), batch_size=10)
test_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size=(224, 224), batch_size=10, shuffle=False)

"""
mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()

x = mobile.layers[-6].output
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(inputs=mobile.input, outputs=predictions)
model.summary()

for layer in model.layers[:-23]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=18, validation_data=valid_batches, validation_steps=3, epochs=5, verbose=2)

keras_file = "sign_numbers_model.h5"
model.save(keras_file)
"""
base_model=tf.keras.applications.mobilenet.MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=tf.keras.layers.Dense(1024,activation='relu')(x) #dense layer 2
x=tf.keras.layers.Dense(512,activation='relu')(x) #dense layer 3
preds=tf.keras.layers.Dense(2,activation='softmax')(x) #final layer with softmax activation


mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()

x = mobile.layers[-6].output
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(inputs=mobile.input, outputs=predictions)
model.summary()

for layer in model.layers[:-23]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=18, validation_data=valid_batches, validation_steps=3, epochs=30, verbose=2)

keras_file = "sign_numbers_model.h5"
model.save(keras_file)
print(test_batches.class_indices)