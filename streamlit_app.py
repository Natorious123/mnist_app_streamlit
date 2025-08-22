import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import cv2
from PIL import Image, ImageOps
from skimage import exposure
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(layout= "centered", page_title="Mobile MNIST", page_icon="🔢")

#st.title("Mobile MNIST")
st.markdown("<h1 style='text-align: center; color: white;'>Mobile MNIST</h1>", unsafe_allow_html=True)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Draw a rectangle in the center
    h, w = img.shape[:2]
    x1, y1 = w//2 - 100, h//2 - 100
    x2, y2 = w//2 + 100, h//2 + 100
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def crop_digit(image):
    # Step 1: Convert to grayscale
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Step 2: Zoom crop (scale factor)
    scale = 1.8
    h, w = gray.shape[:2]
    new_w = int(w / scale)
    new_h = int(h / scale)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    crop = gray[y1:y1+new_h, x1:x1+new_w]
    global crop_to_display
    crop_to_display = crop

    # Step 3: Threshold to binary
    #binary = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY_INV)
    binary = ~crop
    binary = cv2.normalize(binary, None, 0, 255, cv2.NORM_MINMAX)
    binary[binary < 70] = 0
    global binary_to_display
    binary_to_display = binary
    

    # Step 4: Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        img_center = np.array([new_w / 2, new_h / 2])

        def contour_score(cnt, size_weight=1.0, dist_weight=3.0):
            x, y, bw, bh = cv2.boundingRect(cnt)
            blob_center = np.array([x + bw / 2, y + bh / 2])
            dist = np.linalg.norm(blob_center - img_center)
            area = cv2.contourArea(cnt)

            # Higher size_weight → favors bigger blobs more
            # Higher dist_weight → penalizes distance more
            return (area ** size_weight) / ((1 + dist) ** dist_weight)


        # Pick contour with best score
        cnt = max(contours, key=contour_score)

        # Step 5: Crop to bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = binary[y:y+h, x:x+w]

        # Step 6: Resize while preserving aspect ratio
        scale_factor = 23.0 / max(w, h)
        resized = cv2.resize(cropped, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_AREA)

        # Step 7: Pad to 28x28
        padded = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - resized.shape[1]) // 2
        y_offset = (28 - resized.shape[0]) // 2
        padded[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized

        return Image.fromarray(padded)

    else:
        return np.zeros((28, 28), dtype=np.uint8)  # fallback: blank image



def preprocess_image(image):
    img = Image.open(image)#.convert('L')            # Convert to grayscale
    #area = (100, 200, 100, 200)
    #img = img.crop(area)
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

st.markdown("<p style='text-align: center; color: white;'>Take a photo of a single digit on a white background, make sure the digit is dark, ideally written with a black marker/sharpie. The code crops to the largest blob, and treats any brightness around light gray or white as plain white.</p>", unsafe_allow_html=True)
img_data = st.camera_input("")
webrtc_streamer(key="digit-align", video_frame_callback=video_frame_callback)


# Your camera input inside a container

# Inject CSS to center and style it
st.markdown("""
    <style>

    </style>
""", unsafe_allow_html=True)

if img_data is not None:
    preprocessed = preprocess_image(img_data)

    prediction = model.predict(preprocessed)

    predicted_label = np.argmax(prediction)

    #st.metric("Prediction", predicted_label)
    st.markdown(f"""<p style='text-align: center; color: white; font-size: 24pt'>Prediction: {predicted_label}</p>""", unsafe_allow_html=True)

    preprocessed_display = preprocessed.reshape(1,28,28,1)
    col1, col2, col3 = st.columns([1, 2, 1]) 
    with col2:

        st.image(preprocessed_display, caption="Preprocessed 28×28")
        st.image(binary_to_display)
        st.image(crop_to_display)





































