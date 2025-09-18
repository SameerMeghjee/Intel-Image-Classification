import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# -----------------------------
# 1. Download weights if missing
# -----------------------------
weights_path = "intel_model_weights.h5"
file_id = "YOUR_FILE_ID"   # Replace with your Google Drive file ID
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(weights_path):
    st.write("📥 Downloading model weights...")
    gdown.download(url, weights_path, quiet=False)

# -----------------------------
# 2. Define model architecture
# -----------------------------
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    return model

# -----------------------------
# 3. Load weights
# -----------------------------
model = build_model()
model.load_weights(weights_path)

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# -----------------------------
# 4. Streamlit UI
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
