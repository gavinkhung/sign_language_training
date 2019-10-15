import tensorflow as tf
import tensorflowjs as tfjs

#keras_file = "sign_numbers_model"
keras_file = "sign_letters_model"
loaded_model = tf.keras.models.load_model(keras_file+".h5")

tfjs.converters.save_keras_model(loaded_model, keras_file+"_tfjs")