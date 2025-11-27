import tensorflow as tf
import cv2
from flask import Flask, render_template, request, jsonify
import os
import numpy as np

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
print("Loading model...")
model = tf.keras.models.load_model('emotion_model_trained.h5') 
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print("Model loaded!")

# Load Face Detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file:
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'capture.jpg')
        file.save(filepath)

        # Read Image
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect Faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_data = []
        
        # If faces detected, use the first one (largest usually) for prediction
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = img[y:y+h, x:x+w] # Crop face
            face_data = [int(x), int(y), int(w), int(h)] # For frontend
        else:
            face_roi = img # Fallback to full image
            face_data = []

        # Preprocess for Model
        face_roi = cv2.resize(face_roi, (48, 48)) 
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)

        # Prediction
        prediction = model.predict(face_roi)
        label_index = np.argmax(prediction)
        label = emotion_labels[label_index]
        confidence = float(np.max(prediction))
        
        # Simple description logic
        descriptions = {
            'Angry': "Take a deep breath. It's going to be okay.",
            'Disgust': "Something doesn't smell right...",
            'Fear': "You are safe here.",
            'Happy': "Keep smiling! Share your joy today.",
            'Sad': "It's okay to feel down sometimes.",
            'Surprise': "Wow! Didn't see that coming?",
            'Neutral': "Stay calm and carry on."
        }
        description = descriptions.get(label, "Emotion detected.")

        return jsonify({
            'emotion': label,
            'description': description,
            'confidence': f"{confidence:.2f}",
            'face_box': face_data
        })

if __name__ == '__main__':
    print("The server has started!")
    app.run(debug=True, port=5001)
