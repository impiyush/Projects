import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

# print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/usbills_resnet50v2_finetuned_init10_fine10.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


def decode_preds(prob_index, prob):
    classes = ['US $1', 'US $10', 'US $100', 'US $20', 'US $5', 'US $50']
    if prob >= 0.5:
        return classes[prob_index]
    else:
        return 'Cannot be identified. Please try again!'


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
