"""
Quick Prediction Script
Test the trained model with a single image
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os
import config

def predict_image(image_path, model_path=None):
    """
    Make prediction on a single image
    
    Args:
        image_path: Path to the X-ray image
        model_path: Path to the trained model (optional)
    """
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, f'{config.MODEL_NAME}_best.h5')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}")
        return
    
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    print(f"Loading image: {image_path}")
    # Load and preprocess image
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(config.IMG_SIZE)
    
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    print("Making prediction...")
    # Predict
    prediction = model.predict(image_array, verbose=0)
    
    # Get results
    predicted_class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class_idx]) * 100
    predicted_class = config.CLASS_NAMES[predicted_class_idx]
    
    # Display results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nProbability Distribution:")
    print(f"  NORMAL:    {prediction[0][0] * 100:.2f}%")
    print(f"  PNEUMONIA: {prediction[0][1] * 100:.2f}%")
    print("=" * 50)
    
    # Visual indicator
    if predicted_class == 'NORMAL':
        print("✅ No pneumonia detected")
    else:
        print("⚠️  Pneumonia detected - Please consult a doctor")
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': {
            'NORMAL': float(prediction[0][0]) * 100,
            'PNEUMONIA': float(prediction[0][1]) * 100
        }
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_xray.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_image(image_path)
