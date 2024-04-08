import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('EfficientNet_keras.keras')

# Function to preprocess the image
def preprocess_image(image):
    # Convert to RGB (if not already in RGB)
    image = image.convert("RGB")
    
    # Resize the image to the required input size of the model (e.g., 224x224)
    image = image.resize((224, 224))
    
    # Convert the image to a NumPy array
    img_array = np.array(image)
    
    # Normalize the pixel values to be in the range [0, 1]
    img_array = img_array / 255.0
    
    # Perform any additional preprocessing steps as needed
    
    return img_array

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
