import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

st.title("🌱 Plant Disease Detection")
st.write("Upload a photo of a plant leaf to identify diseases")

# Download model from Google Drive (replace YOUR_FILE_ID)
@st.cache_resource
def load_model():
    model_path = "model.h5"
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI model... This may take a moment"):
            # Replace YOUR_FILE_ID with actual Google Drive file ID
            url = f"https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
            response = requests.get(url)
            with open(model_path, "wb") as f:
                f.write(response.content)
    return tf.keras.models.load_model(model_path)

# Replace these with your actual disease names
class_names = ["Healthy", "Early Blight", "Late Blight", "Powdery Mildew"]

# Load the AI model
try:
    model = load_model()
    st.success("✅ AI model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    st.stop()

# Image upload section
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    
    # Prepare image for AI analysis
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    with st.spinner("Analyzing leaf..."):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)
    
    # Show results
    st.subheader("🔍 Diagnosis Result")
    if confidence > 0.7:
        st.success(f"**{predicted_class}** detected with {confidence:.2%} confidence")
    else:
        st.warning(f"Low confidence ({confidence:.2%}) - please upload a clearer image")
