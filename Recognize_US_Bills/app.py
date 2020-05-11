import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil

# print(tf.__version__)
# Declare a flask app
app = Flask(__name__)


# Load the model parameters
MODEL_PATH = 'model/best-model.tf'

# Load your own trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


def decode_preds(prob_index, prob):
    classes = ['Other', 'US $1', 'US $10',
                'US $100', 'US $20', 'US $5', 'US $50']
    most_probable_class = classes[prob_index]
    if prob >= 0.6 and most_probable_class != 0:
        if most_probable_class != 0:
            return classes[prob_index]
        else:
            return 'Not a US Dollar bill!'
    else:
        return 'Cannot be identified correctly. Please try again!'


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        # get the prediction with highest probability
        # print(preds)
        pred_proba = np.amax(preds)    # Max probability
        pred_proba_ind = np.argmax(preds)    # Index of Max probability
        pred_class = decode_preds(pred_proba_ind, pred_proba)

        # Serialize the result, you can add additional fields
        return jsonify(result=pred_class, probability=f"{pred_proba}")

    return None


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
