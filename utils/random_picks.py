import os
import random
import shutil

# Define paths
folder_a = "../datasets/celeba/fullsets/celeba_1024"  # Replace with your source folder path
folder_b = "../datasets/celeba/subsets/celeba_1024"  # Replace with your destination folder path

# Ensure folder B exists
os.makedirs(folder_b, exist_ok=True)

# Get list of image files (assuming common image extensions)
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
images = [f for f in os.listdir(folder_a) if f.lower().endswith(image_extensions)]

# Check if there are enough images
if len(images) < 200:
    print(f"Error: Only {len(images)} images found, need 200.")
else:
    # Randomly select 200 images
    selected_images = random.sample(images, 200)
    
    # Copy selected images to folder B
    for image in selected_images:
        src_path = os.path.join(folder_a, image)
        dst_path = os.path.join(folder_b, image)
        shutil.copy2(src_path, dst_path)
    
    print("Successfully copied 200 images to folder B.")