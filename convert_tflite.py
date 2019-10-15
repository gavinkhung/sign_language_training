import tensorflow as tf

#keras_file = "sign_numbers_model"
keras_file = "sign_letters_model"
loaded_model = tf.keras.models.load_model(keras_file+".h5")

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()
open(keras_file+".tflite", "wb").write(tflite_model)