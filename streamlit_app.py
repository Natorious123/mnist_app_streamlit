import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import cv2
from PIL import Image, ImageOps
from skimage import exposure

st.set_page_config(page_title="Mobile MNIST", page_icon="🔢")

st.title("Mobile MNIST")

def crop_digit(image):
    # Step 1: Convert to grayscale
    image_np = np.array(image)
    print(type(image_np))
    print(image_np.shape)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Step 2: Threshold to binary
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Step 4: Crop to bounding box
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = binary[y:y+h, x:x+w]

        # Step 5: Resize while preserving aspect ratio
        scale = 23.0 / max(w, h)  # scale to fit within 24x24 box
        resized = cv2.resize(cropped, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # Step 6: Pad to 28x28
        padded = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - resized.shape[1]) // 2
        y_offset = (28 - resized.shape[0]) // 2
        padded[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
        processed_pil = Image.fromarray(padded)

        return processed_pil
    else:
        return np.zeros((28, 28), dtype=np.uint8)  # fallback: blank image


def preprocess_image(image):
    img = Image.open(image)#.convert('L')            # Convert to grayscale
    img = crop_digit(img)
    img = img.resize((28, 28))                            # Resize to 28x28
    #img = ImageOps.invert(img)
    img_array = np.array(img)
    #print("Image Min, Max Value", img_array.min(), img_array.max())
    img_array = exposure.rescale_intensity(img_array, in_range='image', out_range=(0, 255))  # Fix contrast 
    #img_array[img_array < 65] = 0
    img_array = img_array/255.0
    #img_array = np.where(img_array > 0.4, 1.0, 0.0)
    img_array = img_array.astype('float32')  # ? Only cast to float
    return img_array.reshape(1,28,28)

model = load_model("mnist_model_10_fold.keras")


img_data = st.camera_input("Take a photo of a single digit on a white background")

# Your camera input inside a container

# Inject CSS to center and style it
st.markdown("""
    <style>
    /* Center the camera input container */
    div[data-testid="stCameraInput"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        flex-direction: column !important;
        margin: auto !important;
        max-width: 360px !important;
    }

    /* Force portrait aspect ratio on the video preview */
    div[data-testid="stCameraInput"] video {
        aspect-ratio: 9 / 16 !important;
        width: 100% !important;
        height: auto !important;
        object-fit: cover !important;
        border-radius: 12px;
    }
        /* Center all text inside the main content */
    .main {
        text-align: center !important;
    }

    /* Optional: center buttons too */
    button {
        margin: auto !important;
        display: block !important;
    }


    /* Remove padding on mobile */
    @media screen and (max-width: 480px) {
        .main {
            padding: 0 !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

if img_data is not None:
    preprocessed = preprocess_image(img_data)

    prediction = model.predict(preprocessed)

    predicted_label = np.argmax(prediction)

    st.metric("Prediction", predicted_label)

    preprocessed_display = preprocessed.reshape(1,28,28,1)

    st.image(preprocessed_display, caption="Preprocessed 28×28")