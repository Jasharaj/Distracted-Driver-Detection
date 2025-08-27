import numpy as np
from PIL import Image
import os

def create_sample_images():
    """Create sample images for testing the distracted driver detection model"""
    
    # List of sample image names from the CSV
    image_names = [
        'img_44733.jpg', 'img_72999.jpg', 'img_25094.jpg', 'img_69092.jpg', 'img_92629.jpg',
        'img_10001.jpg', 'img_10002.jpg', 'img_10003.jpg', 'img_20001.jpg', 'img_20002.jpg',
        'img_30001.jpg', 'img_30002.jpg', 'img_40001.jpg', 'img_40002.jpg', 'img_50001.jpg',
        'img_50002.jpg', 'img_60001.jpg', 'img_60002.jpg', 'img_70001.jpg', 'img_70002.jpg',
        'img_80001.jpg', 'img_80002.jpg', 'img_90001.jpg', 'img_90002.jpg', 'img_44734.jpg',
        'img_72998.jpg', 'img_10004.jpg', 'img_10005.jpg', 'img_20003.jpg', 'img_20004.jpg'
    ]
    
    # Create the images directory if it doesn't exist
    train_dir = 'imgs/train'
    os.makedirs(train_dir, exist_ok=True)
    
    # Generate random images (64x64x3) for testing
    for img_name in image_names:
        # Create a random image with some pattern to simulate different classes
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some simple patterns to differentiate classes (optional)
        # This creates different colored regions to simulate different driver poses
        class_id = int(img_name.split('_')[1][0]) if len(img_name.split('_')) > 1 else 0
        if class_id < 5:
            img_array[:, :, class_id % 3] = img_array[:, :, class_id % 3] // 2 + 100
        
        # Create PIL Image and save
        img = Image.fromarray(img_array)
        img_path = os.path.join(train_dir, img_name)
        img.save(img_path)
        print(f"Created sample image: {img_path}")

if __name__ == "__main__":
    create_sample_images()
    print("Sample images created successfully!")
