import tensorflow as tf
from tensorflow import keras

def preprocess(image, label):
    """
    Preprocess the image to the required input format.
    This function is used for both models: flowers and birds vs squirrels.
    """
    resized_image = tf.image.resize_with_pad(image,299,299)
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label
