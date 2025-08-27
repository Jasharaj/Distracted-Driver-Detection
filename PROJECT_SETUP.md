# Distracted Driver Detection ML Project - Setup Complete! 🚗🤖

## Project Status: ✅ READY TO RUN

This is a machine learning project for detecting distracted driving behaviors using computer vision and deep learning. The project uses a ResNet-50 convolutional neural network to classify driver actions into 10 different categories.

## 🎯 Project Overview

**Goal**: Classify driver images into one of 10 distracted driving categories:
- c0: Safe driving
- c1: Texting - right hand
- c2: Talking on phone - right hand  
- c3: Texting - left hand
- c4: Talking on phone - left hand
- c5: Operating the radio
- c6: Drinking
- c7: Reaching behind
- c8: Hair and makeup
- c9: Talking to passenger

## 🛠️ What's Been Set Up

### ✅ Environment & Dependencies
- **Python 3.10.11** with virtual environment (`.venv_new`)
- **TensorFlow 2.20.0** - Deep learning framework
- **Keras 3.11.3** - High-level neural network API
- **Scikit-learn** - Machine learning utilities
- **NumPy, Pandas, Matplotlib** - Data processing and visualization
- **Jupyter** - Notebook environment
- All dependencies tested and working ✓

### ✅ Project Structure
```
Distracted Driver Detection/
├── 📁 imgs/train/              # Sample training images (30 images)
├── 📁 driver_imgs_list/        # CSV metadata file
├── 📁 supp/                    # Support files (GIF demo)
├── 📝 Distracted Driver detection.ipynb  # Main Jupyter notebook
├── 📄 README.md                # Original project documentation
├── 🐍 resnets_utils.py         # ResNet utility functions
├── 🐍 create_sample_data.py    # Script to generate sample data
└── 🐍 test_project.py          # Project verification script
```

### ✅ Sample Data Created
- **30 sample images** generated for testing (64x64 RGB format)
- **CSV metadata file** with image classifications
- Sample data matches the expected format for the full dataset

## 🚀 How to Run the Project

### Option 1: Using Jupyter Notebook (Recommended)
1. **Activate the environment:**
   ```powershell
   .\.venv_new\Scripts\Activate.ps1
   ```

2. **Start Jupyter:**
   ```powershell
   jupyter notebook
   ```

3. **Open the notebook:**
   - Navigate to `Distracted Driver detection.ipynb`
   - Run cells sequentially starting from the imports

### Option 2: Using the Test Script
```powershell
.\.venv_new\Scripts\Activate.ps1
python test_project.py
```

### Option 3: Direct Python Execution
```powershell
.\.venv_new\Scripts\Activate.ps1
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} ready!')"
```

## 📊 Model Architecture

The project implements a **ResNet-50** architecture with:
- **50 layers** with residual connections
- **Input**: 64x64x3 RGB images
- **Output**: 10-class softmax classification
- **~1.6M parameters** in the sample model
- **Adam optimizer** with categorical crossentropy loss

## 🔧 Key Functions Implemented

1. **Data Processing:**
   - `CreateImgArray()` - Converts images to numerical arrays
   - `PlotClassFrequency()` - Visualizes class distribution
   - `DescribeImageData()` - Data statistics

2. **Model Architecture:**
   - `identity_block()` - ResNet identity blocks
   - `convolutional_block()` - ResNet convolutional blocks  
   - `ResNet50()` - Complete ResNet-50 model

3. **Evaluation:**
   - `LOGO()` - Leave-One-Group-Out cross-validation
   - Visualization and metrics tracking

## 📈 Expected Performance

Based on the original project results:
- **Training Loss**: 0.93
- **Validation Loss**: 3.79  
- **Holdout Loss**: 2.64

*Note: High losses are expected with the sample dataset. Better performance requires the full State Farm dataset and hyperparameter tuning.*

## 🎨 Visualizations Available

- Class frequency distributions
- Sample image displays
- Model architecture plots
- Training/validation curves
- Confusion matrices

## 📦 Using the Full Dataset

To use the complete State Farm Distracted Driver Detection dataset:

1. **Download from Kaggle:**
   - Visit: https://www.kaggle.com/c/state-farm-distracted-driver-detection
   - Download the training images and CSV file

2. **Replace sample data:**
   - Extract images to `imgs/train/`
   - Replace `driver_imgs_list.csv` with the full metadata

3. **Update parameters:**
   - Increase epochs for full training
   - Adjust batch size based on available memory
   - Consider data augmentation

## 🛠️ Troubleshooting

### If Jupyter won't start:
```powershell
.\.venv_new\Scripts\Activate.ps1
python -m pip install --upgrade jupyter ipykernel
python -m ipykernel install --user --name=distracted-driver
```

### If imports fail:
```powershell
python -m pip install --upgrade tensorflow keras numpy pandas matplotlib
```

### If memory issues occur:
- Reduce batch size in model training
- Use smaller image dimensions (32x32 instead of 64x64)
- Process data in smaller chunks

## 🎯 Next Steps

1. **Explore the notebook** - Run all cells to see the complete workflow
2. **Experiment with parameters** - Try different learning rates, epochs, etc.
3. **Add data augmentation** - Improve model generalization
4. **Download full dataset** - For production-quality results
5. **Try transfer learning** - Use pre-trained models for better performance

## 📝 Notes

- This project uses **sample/dummy data** for demonstration
- The ResNet implementation follows the original paper architecture
- Code has been updated for **TensorFlow 2.20** compatibility
- All deprecated imports have been fixed

---

**🎉 The project is fully configured and ready to run! Happy coding!**
