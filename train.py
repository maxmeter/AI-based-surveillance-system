from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Define the directory containing the videos
videos_directory = r'C:\Users\anish\Downloads\video-surviellance-project-code\Avenue_Dataset\Avenue Dataset\training_videos'

# Get a list of video files in the directory
video_files = [file for file in os.listdir(videos_directory) if file.endswith('.avi')]

# Define the path to save frames
frames_directory = os.path.join(videos_directory, 'frames')
os.makedirs(frames_directory, exist_ok=True)

# Iterate through each video file
for video_file in video_files:
    video_path = os.path.join(videos_directory, video_file)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Read and save frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_path = os.path.join(frames_directory, f'{video_file}_{frame_count:03d}.jpg')
        cv2.imwrite(frame_path, frame)
    
    # Release the video capture object
    cap.release()

# Preprocess and store frames
store_image = []
for frame_file in os.listdir(frames_directory):
    frame_path = os.path.join(frames_directory, frame_file)
    image = cv2.imread(frame_path)
    image = cv2.resize(image, (227, 227))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0  # Normalize
    store_image.append(gray)

# Convert list to numpy array
store_image = np.array(store_image)

# Define the model
stae_model = Sequential()
stae_model.add(Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid',
                      input_shape=(227, 227, 10, 1), activation='tanh'))
stae_model.add(Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tanh'))
stae_model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4,
                          recurrent_dropout=0.3, return_sequences=True))
stae_model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3,
                          return_sequences=True))
stae_model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', return_sequences=True,
                          dropout=0.5))
stae_model.add(Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid',
                               activation='tanh'))
stae_model.add(Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid',
                               activation='tanh'))

# Compile the model
stae_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Load training data
training_data = store_image  # Assuming you want to use the frames as training data
frames = training_data.shape[0]
frames = frames - frames % 10
training_data = training_data[:frames]

# Reshape training data
training_data = training_data.reshape(-1, 227, 227, 10, 1)

# Set training parameters
epochs = 5
batch_size = 1

# Define filepath to save the model
checkpoint_filepath = "saved_model.keras"

# Define callbacks
callback_save = ModelCheckpoint(checkpoint_filepath, monitor="mean_squared_error", save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
stae_model.fit(training_data, training_data,  # Assuming autoencoder training (input=output)
               batch_size=batch_size,
               epochs=epochs,
               callbacks=[callback_save, callback_early_stopping]
               )

# Save the model
stae_model.save("saved_model.h5")
