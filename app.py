from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64   
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['*'])  # Allow all origins for now

# Global model variable
model = None

# Class names for predictions
CLASS_NAMES = [
    'Safe driving',
    'Texting - right hand', 
    'Talking on phone - right',
    'Texting - left hand',
    'Talking on phone - left',
    'Operating the radio',
    'Drinking',
    'Reaching behind',
    'Hair and makeup',
    'Talking to passenger'
]

CLASS_CODES = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

def create_demo_model():
    """Create a demo model for deployment if no trained model exists"""
    logger.info("Creating demo CNN model...")
    
    # Create a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Demo model created with {model.count_params()} parameters")
    return model

def load_model():
    """Load the trained model or create a demo model"""
    global model
    try:
        # Try to load a saved model first
        if os.path.exists('trained_model.h5'):
            logger.info("Loading saved model...")
            model = tf.keras.models.load_model('trained_model.h5')
            logger.info("Saved model loaded successfully")
        else:
            logger.info("No saved model found, creating demo model...")
            model = create_demo_model()
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.info("Creating fallback demo model...")
        model = create_demo_model()

def preprocess_image(image_data):
    """Preprocess image for prediction"""
    try:
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((64, 64))
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "Distracted Driver Detection API",
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__,
        "classes": len(CLASS_NAMES)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    try:
        # Check if model is loaded
        if model is None:
            load_model()
            
        if model is None:
            return jsonify({'error': 'Model not available'}), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        image_array = preprocess_image(data['image'])
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(image_array, verbose=0)
        
        # Get top prediction
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            top_predictions.append({
                'class': CLASS_CODES[idx],
                'class_name': CLASS_NAMES[idx],
                'confidence': float(prediction[0][idx])
            })
        
        logger.info(f"Prediction completed: {CLASS_NAMES[class_idx]} ({confidence:.3f})")
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': CLASS_CODES[class_idx],
                'class_name': CLASS_NAMES[class_idx],
                'confidence': confidence,
            },
            'top_predictions': top_predictions,
            'all_predictions': prediction[0].tolist()
        })
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all available classes"""
    classes_info = []
    for i, (code, name) in enumerate(zip(CLASS_CODES, CLASS_NAMES)):
        classes_info.append({
            'index': i,
            'code': code,
            'name': name
        })
    
    return jsonify({
        'classes': classes_info,
        'total': len(classes_info)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize model on startup
if __name__ == '__main__':
    logger.info("Starting Distracted Driver Detection API...")
    load_model()
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
