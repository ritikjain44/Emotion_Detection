from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pickle
import cv2

app = Flask(__name__)

# Load the saved model
model_path = "C://Users//ritik//Downloads//emotion detection//models//EfficientNet_Unfreezing.h5"
model = tf.keras.models.load_model(model_path)

# Preprocess image function
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if len(gray_image.shape) < 3:
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    equalized_image = cv2.equalizeHist(blurred_image)
    _, thresholded_image = cv2.threshold(equalized_image, 127, 255, cv2.THRESH_BINARY)
    final_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB)
    return final_image

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Ensure image size matches the input size of the model
    img = preprocess_image(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    result = emotions[np.argmax(prediction)]
    return jsonify({'emotion': result})

# Home route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
