import numpy as np

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL

# define paths for train, valid, and test data
train_path = 'asl-alphabet/training_data'
valid_path = 'asl-alphabet/testing_data'
test_path = 'asl-alphabet/testing_data'

train_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(224, 224), batch_size=10)
valid_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path, target_size=(224, 224), batch_size=10)
test_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size=(224, 224), batch_size=10, shuffle=False)

mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()

x = mobile.layers[-6].output
predictions = tf.keras.layers.Dense(29, activation='softmax')(x)
model = tf.keras.models.Model(inputs=mobile.input, outputs=predictions)
model.summary()

for layer in model.layers[:-23]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=18, validation_data=valid_batches, validation_steps=3, epochs=60, verbose=2)

keras_file = "sign_letters_model.h5"
model.save(keras_file)
print(test_batches.class_indices)