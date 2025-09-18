import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# -----------------------------
# 1. Download model if missing
# -----------------------------
model_path = "best_model.h5"
file_id = "1-qqf2-Zjy6qBv2ODfenikxj3vWiz6O6Z"   
url = f"https://drive.google.com/file/d/1-qqf2-Zjy6qBv2ODfenikxj3vWiz6O6Z/view?usp=drive_link={file_id}"

if not os.path.exists(model_path):
    st.write("📥 Downloading model...")
    gdown.download(url, model_path, quiet=False)

# -----------------------------
# 2. Load the full model
# -----------------------------
model = tf.keras.models.load_model(model_path)

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("🌍 Intel Image Classification")
st.write("Upload an image and let the model classify it into one of six categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((150,150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: **{class_names[class_index]}** ({confidence*100:.2f}% confidence)")
