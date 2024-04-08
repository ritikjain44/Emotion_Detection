import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Define custom layers if necessary
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = tf.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

# Register custom objects when loading the model
with tf.keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
    model = tf.keras.models.load_model('C://Users//ritik//Downloads//Emotion_detection//Emotion_Detection//models//EfficientNet_Unfreezing.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Preprocess the image as needed
    return image

# Function to predict emotion
def predict_emotion(image):
    # Preprocess the image
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    
    # Perform prediction using the loaded model
    prediction = model.predict(img)
    
    emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    predicted_class = np.argmax(prediction)
    
    return emotions[predicted_class]

# Streamlit UI
st.title('Emotion Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        emotion = predict_emotion(image)
        st.write('Detected Emotion:', emotion)
