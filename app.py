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
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/your_model.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
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
        index=np.argmax(preds)
        print('preds',preds)
        target_map={"0": "Other garbage/disposable snack box ","1": "Other garbage/contaminated plastics ","2": "Other garbage/cigarette butts ","3": "Other garbage/toothpick ","4": "Other rubbish/Broken flowerpots and dishes ","5": "Other garbage/bamboo chopsticks ","6": "kitchen waste/leftovers ","7": "Kitchen waste/big bones","8": "Kitchen waste/fruit peel","9": "Kitchen waste/fruit pulp","10":"Kitchen waste/tea leaves","11":"Kitchen waste/vegetable leaves and roots","12": "Kitchen waste/eggshell","13":"Kitchen waste/fish bones","14": "Recyclables/charging bank ","15": "Recyclables/bags ","16": "Recyclables/cosmetics bottles ","17": "Recyclables/plastic toys","18": "Recyclables/plastic bowls","19": "Recyclables/plastic hangers","20": "Recyclables/express paper bags ","21": "Recyclables/plug wires ","22": "Recyclables/used clothes ","23": "Recyclables/cans ","24": "Recyclables/pillows","25": "Recyclables/plush toys","26": "Recyclable/shampoo bottle ","27": "Recyclables/glass ","28": "Recyclables/shoes ","29": "Recyclables/cutting board ","30": "Recyclables/cartons ","31": "Recyclables/seasoning bottles ","32": "Recyclables/bottles ","33": "Recyclables/metal food cans","34": "Recyclables/POTS","35": "Recyclables/edible oil drums ","36": "Recyclables/beverage bottles","37":"Hazardous waste/dry battery","38": "Hazardous waste/ointment","39": "Hazardous waste/expired drugs"}
        result=target_map[str(index)]
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        #result = str(pred_class[0][0][1])               # Convert to string
        #result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        print('result',result)
        print('preds',preds)
        print(preds[0,index])
        return jsonify(result=result, probability=str(preds[0,index]))

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
