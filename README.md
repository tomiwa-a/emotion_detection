# Emotion Detection Project

A web-based application that detects facial emotions in real-time using a webcam. It uses a MobileNetV2 deep learning model and OpenCV for face tracking.

## Features

- **Real-time Emotion Detection**: Classifies 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
- **Face Tracking**: Uses Haar Cascades to detect and focus on faces for better accuracy.
- **Modern UI**: Clean, responsive interface with real-time confidence feedback.

## File Structure

```
emotion_detection/
├── app.py                      # Flask backend application
├── emotion_model_trained.h5    # Trained Keras model
├── haarcascade_frontalface_default.xml # OpenCV face detector
├── model_retrain.py            # Script to train the model
├── model_saving.py             # Script to create model architecture
├── requirements.txt            # Python dependencies
├── static/
│   └── style.css               # Frontend styling
└── templates/
    └── index.html              # Frontend HTML/JS
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Conda (recommended)

### 2. Installation

1.  Clone the repository:

    ```bash
    git clone <repository-url>
    cd emotion_detection
    ```

2.  Create and activate environment:

    ```bash
    conda create -n emotion_env python=3.9
    conda activate emotion_env
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Training the Model (Optional)

To make the model "smart" and accurate, you need to train it on the **FER2013** dataset.

1.  **Download Dataset**: Download the FER2013 dataset (e.g., from Kaggle).
2.  **Prepare Folder**: Extract the dataset so you have a `fer2013` folder in the project root.
    - Structure should look like: `emotion_detection/fer2013/train/` and `emotion_detection/fer2013/test/`.
3.  **Run Training**:
    ```bash
    python model_retrain.py
    ```
    This script will:
    - Load the images.
    - Train the MobileNetV2 model (default 5 epochs).
    - Save the new model as `emotion_model_trained.h5`.
4.  **Use Trained Model**: The `app.py` is already configured to load `emotion_model_trained.h5`. Restart the app to use the new brain.

### 4. Running the App

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5001/`.
