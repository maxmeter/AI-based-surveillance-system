import cv2
import numpy as np
from keras.models import load_model
import imutils
import requests
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import pywhatkit 
import streamlit as st
import tempfile
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

# Function to send email with an image attachment
def SendMail(img_file_name):
    img_data = open(img_file_name, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Abnormal Event Detected'
    msg['From'] = ''  # Replace with your sender email
    msg['To'] = ''  # Replace with recipient email

    text = MIMEText("An abnormal event has been detected. Please check the attached image.")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(img_file_name))
    msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(msg['From'], "")  # Replace with your email password
    s.sendmail(msg['From'], [msg['To']], msg.as_string())
    s.quit()

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
            message = "Hello, this is a test message sent via security system!. There is Abnormal activity detected"
            
            # Send the message
            if lock == 0:
                pywhatkit.sendwhatmsg_instantly(phone_number, message)
                lock = 1
            img_filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(img_filename, frame_resized)

            # Send email with the image attachment
            # SendMail(img_filename)

            # Send SMS notification
            resp = requests.post('https://textbelt.com/text', {
                'phone': 'MOBILE NO',
                'message': 'Abnormal Event Detected',
                'key': 'textbelt',
            })
            st.write(resp.json())
            st.write('Abnormal Event Detected')

        # Display the video frame
        st.image(frame_resized, channels="BGR")

        # Exit loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
