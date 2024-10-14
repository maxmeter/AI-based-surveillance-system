import cv2
import numpy as np
from keras.models import load_model
import imutils
import requests
import os
import streamlit as st
import tempfile
import pywhatkit

lock = 0

# Function to calculate mean squared loss
def mean_squared_loss(x1, x2):
    difference = x1 - x2
    a, b, c, d, e = difference.shape
    n_samples = a * b * c * d * e
    sq_difference = difference ** 2
    Sum = sq_difference.sum()
    distance = np.sqrt(Sum)
    mean_distance = distance / n_samples
    return mean_distance

# Load the model
model = load_model("saved_model.h5")

# Streamlit UI
st.title("Abnormal Event Detection")

# Video file upload
uploaded_file = st.file_uploader("Upload Video", type=["avi"])

if uploaded_file is not None:
    # Save the uploaded file locally
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    # OpenCV video reader
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        st.write("Video file loaded successfully.")
    else:
        st.write("Failed to load video file.")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        # Check if the frame is None (end of video or error)
        if frame is None:
            break

        # Process each frame for abnormal event detection
        imagedump = []

        for i in range(10):
            # Resize the frame
            frame_resized = imutils.resize(frame, width=1000, height=1200)
            frame_resized = cv2.resize(frame_resized, (227, 227), interpolation=cv2.INTER_AREA)

            # Convert to grayscale
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray = (gray - gray.mean()) / gray.std()
            gray = np.clip(gray, 0, 1)
            imagedump.append(gray)

        imagedump = np.array(imagedump)
        imagedump.resize(227, 227, 10)
        imagedump = np.expand_dims(imagedump, axis=0)
        imagedump = np.expand_dims(imagedump, axis=4)

        # Model prediction
        output = model.predict(imagedump)
        frame_count += 1

        # Calculate loss
        loss = mean_squared_loss(imagedump, output)

        # Set thresholds
        lower_threshold = 0.000068
        upper_threshold = 0.00068

        # Calculate the weighted average threshold
        threshold1 = (upper_threshold + lower_threshold) / 2.36

        # Check if loss exceeds threshold
        if loss > threshold1:
            # Save the current frame as an image
            phone_number = "+"  # Replace with the recipient's phone number
            message = "Hello, this is a test message sent via security system! Abnormal activity detected."

            # Send the message
            if lock == 0:
                pywhatkit.sendwhatmsg_instantly(phone_number, message)
                lock = 1

            img_filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(img_filename, frame_resized)

            # Optional: Uncomment for Email or SMS notifications
            # SendMail(img_filename)
            # Send SMS notification
            # resp = requests.post('https://textbelt.com/text', {
            #     'phone': 'MOBILE NO',
            #     'message': 'Abnormal Event Detected',
            #     'key': 'textbelt',
            # })
            # st.write(resp.json())
            
            st.write('Abnormal Event Detected')

        # Display the video frame
        st.image(frame_resized, channels="BGR")

    # Release the video capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
