 Sign2Text – Real-Time Sign Language Recognition
 Overview

Sign2Text is a deep learning–based system that converts hand signs into text using computer vision.
The system processes live webcam input and predicts sign language characters in real time.

This project combines CNN-based feature extraction with real-time inference to build an assistive communication tool.

 Problem Statement

Communication barriers exist between hearing-impaired individuals and non-sign language users.

The goal of this project is to:

Detect hand gestures

Classify sign language characters

Convert predictions into readable text in real time

 Model Architecture

The model follows a standard CNN pipeline:

Input Image → Convolution → ReLU → Pooling → Flatten → Dense → Output Layer

Key Components:

Convolution layers for feature extraction

ReLU activation

MaxPooling for dimensionality reduction

Fully connected layers for classification

 Dataset

Dataset Used: Sign Language MNIST

28x28 grayscale images

24 classes (A–Y excluding J & Z)

Preprocessed and normalized before training

(Note: Dataset is not included in this repo. It can be downloaded separately.)

🛠 Tech Stack

Python

OpenCV

TensorFlow / Keras (or PyTorch if applicable)

NumPy

Real-time webcam integration

📂 Project Structure
Sign2Text/
│
├── src/
│   ├── train_combined.py
│   ├── realtime_combined.py
│   ├── utils.py
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Installation

Clone the repository:

git clone https://github.com/AdityaNautiyal927/Sign2Text.git
cd Sign2Text

Install dependencies:

pip install -r requirements.txt
▶️ Run the Project
 cd src
 python realtime_combined.py

Train the model:

python src/train_combined.py

Run real-time detection:

python src/realtime_combined.py

 Results

Real-time sign classification using webcam

Accurate alphabet detection on clean inputs

Low latency inference

(You can later add accuracy metrics here.)

🔮 Future Improvements

Add word-level prediction

Add LSTM for sequence modeling

Improve robustness to lighting/background

Deploy as web application using FastAPI

Extend to regional sign datasets

👤 Author

Aditya Nautiyal
Aspiring AI Engineer | Machine Learning | Computer Vision