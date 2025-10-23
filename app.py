"""
Flask API for Pneumonia Detection System
REST API for real-time chest X-ray classification
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import config

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

# Load the trained model
MODEL_PATH = os.path.join(config.MODEL_DIR, f'{config.MODEL_NAME}_best.h5')
model = None

def load_model():
    """Load the trained model"""
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully!")
    else:
        print(f"⚠ Model not found at {MODEL_PATH}")
        print("Please train the model first using train.py")

def preprocess_image(image_bytes):
    """
    Preprocess uploaded image for model prediction
    """
    # Open image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize(config.IMG_SIZE)
    
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Accepts image file and returns prediction with confidence
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    # Check if image is present in request
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image provided'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'error': 'Empty filename'
        }), 400
    
    try:
        # Read and preprocess image
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_idx]) * 100
        predicted_class = config.CLASS_NAMES[predicted_class_idx]
        
        # Prepare response
        response = {
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'probabilities': {
                'NORMAL': round(float(prediction[0][0]) * 100, 2),
                'PNEUMONIA': round(float(prediction[0][1]) * 100, 2)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Error processing image: {str(e)}'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'architecture': config.BACKBONE,
        'input_shape': list(config.IMG_SIZE) + [3],
        'classes': config.CLASS_NAMES,
        'total_parameters': int(model.count_params())
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the Flask app
    print("\n" + "=" * 50)
    print("PNEUMONIA DETECTION API SERVER")
    print("=" * 50)
    print("\nStarting server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /                  - Web interface")
    print("  GET  /api/health        - Health check")
    print("  POST /api/predict       - Make prediction")
    print("  GET  /api/model-info    - Model information")
    print("\nPress CTRL+C to stop the server")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
