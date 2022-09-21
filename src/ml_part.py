# Train a tensorflow model to classify chess pieces
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

MODELNAME = 'efficientnetv2-b3-21k.h5'
model = keras.models.load_model(MODELNAME, custom_objects={'KerasLayer': hub.KerasLayer})
IMAGE_SIZE = (300, 300)


def build_dataset(subset):
    return tf.keras.preprocessing.image_dataset_from_directory(
        '../assets/chess_pieces',
        validation_split=.20,
        subset=subset,
        label_mode="categorical",
        # Seed needs to provided when using validation_split and shuffle = True.
        # A fixed seed is used so that the validation set is stable across runs.
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=1)


train_ds = build_dataset("training")
class_names = tuple(train_ds.class_names)


def predict(img):
    print('Predicting...')
    image = np.array(img)
    image = tf.image.resize(image, (300, 300))
    image = tf.expand_dims(image, 0)
    prediction_scores = model.predict(image)
    print("Prediction scores:", prediction_scores)
    predicted_index = np.argmax(prediction_scores)
    print("Predicted index:", predicted_index)
    print("Predicted label: " + class_names[predicted_index])
    return class_names[predicted_index]


img = tf.keras.preprocessing.image.load_img('../assets/chess_pieces/black_queen/black_queen_1.png')
predict(img)
