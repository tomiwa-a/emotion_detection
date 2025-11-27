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

        # Preprocess Image
        # MobileNetV2 expects (48, 48, 3) or (224, 224, 3) depending on how we saved it.
        # In model_saving.py we used input_shape=(48, 48, 3).
        
        img = cv2.imread(filepath)
        img = cv2.resize(img, (48, 48)) # Resize to match model input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB
        img = img / 255.0 # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=0) # Add batch dimension: (1, 48, 48, 3)

        # Prediction
        prediction = model.predict(img)
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
            'confidence': f"{confidence:.2f}"
        })

if __name__ == '__main__':
    print("The server has started!")
    app.run(debug=True, port=5001)
