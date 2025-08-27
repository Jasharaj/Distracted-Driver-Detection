#!/usr/bin/env python3
"""
Distracted Driver Detection - Standalone Demo
Demonstrates the core ML functionality without Jupyter notebook
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, 
                                   Dropout, BatchNormalization, Add, Activation)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    """Load sample data and preprocess images"""
    print("Loading data...")
    
    # Load CSV metadata
    driver_imgs_df = pd.read_csv('driver_imgs_list/driver_imgs_list.csv')
    print(f"Loaded {len(driver_imgs_df)} image records")
    
    # Convert class names to integers
    class_mapping = {'c0': 0, 'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4, 
                    'c5': 5, 'c6': 6, 'c7': 7, 'c8': 8, 'c9': 9}
    driver_imgs_df['class_id'] = driver_imgs_df['classname'].map(class_mapping)
    
    # Load and preprocess images
    X = []
    y = []
    
    for idx, row in driver_imgs_df.iterrows():
        img_path = f"imgs/train/{row['img']}"
        if os.path.exists(img_path):
            # Load image
            img = image.load_img(img_path, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            
            X.append(img_array)
            y.append(row['class_id'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} images with shape {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, driver_imgs_df

def create_simple_cnn(input_shape=(64, 64, 3), num_classes=10):
    """Create a simple CNN model for demonstration"""
    
    input_layer = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Second convolutional block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Third convolutional block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    return model

def train_model(X, y):
    """Train the model with sample data"""
    print("\nTraining model...")
    
    # Convert labels to categorical
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=10)
    
    # For small dataset, use a smaller test split or train on all data
    if len(X) < 50:
        print("Small dataset detected - training on all data for demo")
        X_train, X_val = X, X
        y_train, y_val = y_categorical, y_categorical
    else:
        # Split data normally for larger datasets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Create and compile model
    model = create_simple_cnn()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print(f"\nModel created with {model.count_params():,} parameters")
    
    # Train model (short training for demo)
    history = model.fit(
        X_train, y_train,
        batch_size=4,  # Smaller batch for small dataset
        epochs=3,  # Very small number for demo
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training curves saved as 'training_curves.png'")

def demonstrate_predictions(model, X, y, class_names):
    """Show some predictions"""
    print("\nMaking predictions on sample images...")
    
    # Make predictions
    predictions = model.predict(X[:10])
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = y[:10]
    
    print("\nSample Predictions:")
    print("Image | Actual | Predicted | Confidence")
    print("-" * 45)
    
    for i in range(10):
        confidence = predictions[i][predicted_classes[i]]
        actual_class = class_names[actual_classes[i]]
        predicted_class = class_names[predicted_classes[i]]
        
        print(f"{i+1:5} | {actual_class:6} | {predicted_class:9} | {confidence:.3f}")

def main():
    """Main demonstration function"""
    print("🚗 Distracted Driver Detection - Demo")
    print("=" * 50)
    
    # Class names for interpretation
    class_names = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    
    try:
        # Load data
        X, y, df = load_and_preprocess_data()
        
        # Train model
        model, history = train_model(X, y)
        
        # Plot results
        plot_training_history(history)
        
        # Show predictions
        demonstrate_predictions(model, X, y, class_names)
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Try the Jupyter notebook for more detailed analysis")
        print("2. Download the full State Farm dataset for better results")
        print("3. Experiment with different model architectures")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Make sure you're in the project directory and have activated the virtual environment")

if __name__ == "__main__":
    main()
