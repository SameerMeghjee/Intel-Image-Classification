import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("intel_cnn_model.h5")

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

st.title("🌍 Intel Image Classification")
st.write("Upload an image and let the model classify it into one of six categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((150,150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write(f"Prediction: **{class_names[class_index]}**")
    st.write(f"Confidence: {confidence*100:.2f}%")
