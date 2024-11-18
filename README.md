data set link-(https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)


---

# Abnormal Event Detection System

This project implements an **Abnormal Event Detection System** using deep learning techniques. The system processes video frames, detects anomalies in video sequences, and generates notifications upon detecting abnormal events. It uses a convolutional LSTM-based autoencoder model for anomaly detection.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Training](#training)
7. [Testing](#testing)
8. [Results](#results)
9. [Acknowledgments](#acknowledgments)

---

## Project Overview
The goal of this project is to detect anomalies in video streams, such as suspicious activities in a surveillance feed. The implemented system:
- Extracts frames from video files.
- Preprocesses the frames for training and inference.
- Trains a Convolutional LSTM Autoencoder model to identify deviations from normal patterns.

---

## Dataset
### Source:
- The **Avenue Dataset** was used for this project, containing training and testing videos of real-world surveillance footage.

### Data Structure:
1. **Training Videos**: Located in the `training_videos` directory.
2. **Extracted Frames**: Saved in the `frames` subdirectory within the same folder.

---

## Model Architecture
The model is a Convolutional LSTM Autoencoder with the following layers:
1. **Convolutional Layers**: For spatial feature extraction.
2. **LSTM Layers**: For capturing temporal patterns.
3. **Deconvolutional Layers**: To reconstruct input sequences.

**Input Shape**: `(227, 227, 10, 1)`  
- 227x227 resolution  
- 10 consecutive frames  
- Grayscale channel  

The model minimizes reconstruction loss to detect anomalies, as abnormal events result in higher reconstruction errors.

---

## Requirements
### Dependencies:
- Python 3.8 or higher
- Libraries:
  - `numpy`
  - `opencv-python`
  - `keras`
  - `tensorflow`
  - `imutils`
  - `streamlit`

### Installation:
Install the required libraries using:
```bash
pip install -r requirements.txt
```

---

## Usage
### 1. Clone the Repository:
```bash
git clone https://github.com/maxmeter/abnormal-event-detection.git
cd abnormal-event-detection
```

### 2. Extract Frames from Videos:
Run the script to preprocess videos and extract frames:
```bash
python extract_frames.py
```

### 3. Train the Model:
Train the Convolutional LSTM Autoencoder:
```bash
python train_model.py
```

### 4. Run the Detection System:
Start the detection system using Streamlit:
```bash
streamlit run detection_system.py
```

---

## Training
The training script preprocesses video frames and trains the autoencoder on normal patterns. 
1. Training data is normalized and resized to 227x227 resolution.
2. Frames are grouped into sequences of 10 for temporal modeling.
3. The model is trained to minimize reconstruction loss (`mean_squared_error`).

---

## Testing
Upload a video in the Streamlit interface to test anomaly detection:
1. Frames are extracted and preprocessed.
2. Each frame sequence is passed through the trained model.
3. Reconstruction loss is compared to a threshold to flag anomalies.

---

## Results
The system alerts the user and saves a snapshot of the detected abnormal event. Notifications can be sent via WhatsApp or email (optional).

---

## Acknowledgments
- **Dataset**: Avenue Dataset
- **Frameworks**: TensorFlow, Keras, OpenCV, Streamlit

Feel free to contribute to this project or report issues in the repository!

---
