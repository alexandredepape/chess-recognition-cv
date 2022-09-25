# Train a tensorflow model to classify chess pieces

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

MODELNAME = '../models/greatmodel.h5'
IMAGE_SIZE = (300, 300)


class PiecePredictor:
    def __init__(self):
        print("Loading model...")
        self.model = keras.models.load_model(MODELNAME, custom_objects={'KerasLayer': hub.KerasLayer})
        print("Model loaded")
        self.class_names = ['black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
                            'empty', 'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen',
                            'white_rook']


    def predict(self, img):
        print('Predicting...')
        image = np.array(img)
        image = tf.image.resize(image, (300, 300))
        image = tf.expand_dims(image, 0)
        image = image / 255.0
        prediction_scores = self.model.predict(image)
        print("Finished predicting")
        # print("Prediction scores:", prediction_scores)
        predicted_index = np.argmax(prediction_scores)
        # print("Predicted index:", predicted_index)
        # print("Predicted label: " + self.class_names[predicted_index])
        return self.class_names[predicted_index]
