import numpy as np
import tensorflow as tf
import keras_preprocessing
import PIL


keras_file = "sign_numbers_model.h5"
loaded_model_mines = tf.keras.models.load_model(keras_file)
loaded_model = tf.keras.applications.mobilenet.MobileNet()

img_path = "photos/9/IMG_4128.JPG"

print(img_path)

img = keras_preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = keras_preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
pImg = tf.keras.applications.mobilenet.preprocess_input(img_array)
prediction_mines = loaded_model_mines.predict(pImg, batch_size=1000)[0].tolist()

print("prediction results: ")
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for i in range(len(prediction_mines)):
    print(labels[i],"\t",prediction_mines[i])

total = 0
for num in prediction_mines:
    total += num

print("prediction sum", total)