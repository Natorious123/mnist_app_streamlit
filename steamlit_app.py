import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="MNIST Camera", page_icon="🔢")
st.title("Camera MNIST classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model_10_fold.keras")

model = load_model()

img_data = st.camera_input("Take a photo of a single digit on a plain background")

if img_data is not None:
    img = Image.open(img_data).convert("L")  # grayscale

    # Center-crop to square
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((28, 28))

    arr = np.array(img).astype("float32") / 255.0
    arr = 1.0 - arr  # invert if you trained on white-on-black MNIST
    arr = arr.reshape(1, 28, 28, 1)

    probs = model.predict(arr, verbose=0)[0]
    pred = int(np.argmax(probs))
    st.metric("Prediction", pred, delta=f"confidence {probs[pred]:.2f}")


    st.image(img.resize((112, 112)), caption="Preprocessed 28×28", width=112)
