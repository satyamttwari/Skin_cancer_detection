import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model_path = 'InceptionV3.h5'  # Replace with the path to your .h5 file
model = load_model(model_path)

# Define the class labels
class_labels = ["basal cell carcinoma", "non cancer", "melanoma"]

# Streamlit app
st.title("Skin Cancer Classifier")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
if st.button("Classify"):
    # Preprocess the image for the model
    image = image.resize((224, 224))  # Adjust the size based on your model's input size
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)
    # Make predictions
    prediction = model.predict(image)

   # Display the result
    st.write("Prediction:")
    st.write(f"Class: {class_labels[np.argmax(prediction)]}")