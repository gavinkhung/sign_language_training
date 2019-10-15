import numpy as np
import tensorflow as tf
import keras_preprocessing
import PIL


keras_file = "sign_letters_model.h5"
loaded_model_mines = tf.keras.models.load_model(keras_file)
loaded_model = tf.keras.applications.mobilenet.MobileNet()

img_path = "asl/testing_data/C/C_test.JPG"
#img_path = "asl-alphabet/testing_data/C/C_test.JPG"
#img_path = "photos/a.jpg"

print(img_path)

img = keras_preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = keras_preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
pImg = tf.keras.applications.mobilenet.preprocess_input(img_array)
prediction_mines = loaded_model_mines.predict(pImg, batch_size=1000)[0].tolist()

print("prediction results: ")
labels = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i in range(len(prediction_mines)):
    print(labels[i],"\t\t",prediction_mines[i])

total = 0
for num in prediction_mines:
    total += num

print("prediction sum", total)