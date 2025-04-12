import os
import flask
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import py7zr

# Create Flask app
app = Flask(__name__)

# Define the path to the model weights
COMPRESSED_MODEL_PATH = './models/best_weights_model.7z'
EXTRACTED_MODEL_PATH = './models/best_weights_model.keras'

# Function to extract the model from .7z parts
def extract_model():
    if not os.path.exists(EXTRACTED_MODEL_PATH):
        print("Extracting model weights...")
        with py7zr.SevenZipFile(COMPRESSED_MODEL_PATH, mode='r') as archive:
            archive.extractall(path='./models')
        print("Model extraction complete.")

# Load the model (ensure the model is extracted first)
model = None
def load_model():
    global model
    if model is None:
        print("Loading model...")
        extract_model()
        model = tf.keras.models.load_model(EXTRACTED_MODEL_PATH)
        print("Model loaded successfully.")

@app.route('/')
def home():
    return "Welcome to the Nodule Detector API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the model is loaded
    load_model()

    # Get the input data from the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read the image
        from PIL import Image
        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))  # Resize to the input size expected by the model
        img_array = np.array(image) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform prediction
        predictions = model.predict(img_array)
        result = {
            'predictions': predictions.tolist()
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app (this won't be used in Render, but useful for local testing)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
