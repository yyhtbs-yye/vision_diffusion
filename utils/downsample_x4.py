import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
import time

def downsample_image(image_info):
    """
    Downsample an image using bicubic interpolation.
    
    Args:
        image_info (tuple): Contains (image_path, output_path, scale_factor).
    
    Returns:
        tuple: (image_path, success_status)
    """
    image_path, output_path, scale_factor = image_info
    
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return image_path, False
        
        # Get original dimensions
        h, w = img.shape[:2]
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Perform bicubic downsampling
        downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the downsampled image
        cv2.imwrite(output_path, downsampled)
        
        return image_path, True
    except Exception as e:
        return image_path, False

def process_images_parallel(image_files, input_dir, output_dir, scale_factor, num_processes=None):
    """
    Process images in parallel using multiple processes.
    
    Args:
        image_files (list): List of image file paths.
        input_dir (str): Input directory.
        output_dir (str): Output directory.
        scale_factor (float): Scale factor for downsampling.
        num_processes (int, optional): Number of processes to use. If None, uses CPU count.
    
    Returns:
        int: Number of successfully processed images.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Prepare arguments for each image
    process_args = []
    for image_path in image_files:
        # Create relative path for output
        rel_path = os.path.relpath(image_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        process_args.append((image_path, output_path, scale_factor))
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process images and track progress
        results = list(tqdm(
            pool.imap(downsample_image, process_args),
            total=len(process_args),
            desc=f"Downsampling images using {num_processes} processes"
        ))
    
    # Count successful operations
    success_count = sum(1 for _, success in results if success)
    
    # Get failed images
    failed_images = [path for path, success in results if not success]
    if failed_images:
        print(f"\nFailed to process {len(failed_images)} images:")
        for path in failed_images[:10]:  # Show only first 10 failed images
            print(f"  - {path}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description='Downsample images using bicubic interpolation with multiple processes.')
    parser.add_argument('--input_dir', type=str, default="./datasets/celeba/fullsets/celeba_1024", help='Directory containing images to downsample.')
    parser.add_argument('--output_dir', type=str, default="./datasets/celeba/fullsets/celeba_256", help='Directory to save downsampled images.')
    parser.add_argument('--scale', type=float, default=0.25, help='Scale factor for downsampling (0 < scale < 1).')
    parser.add_argument('--extensions', type=str, default='.jpg,.jpeg,.png,.bmp,.tiff', 
                        help='Comma-separated list of image extensions to process.')
    parser.add_argument('--processes', type=int, default=None, 
                        help='Number of processes to use. Default is number of CPU cores.')
    
    args = parser.parse_args()
    
    # Validate scale factor
    if args.scale <= 0 or args.scale >= 1:
        print("Error: Scale factor must be between 0 and 1.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get list of valid extensions
    valid_extensions = [ext.lower() for ext in args.extensions.split(',')]
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {args.input_dir} with extensions {valid_extensions}")
        return
    
    print(f"Found {len(image_files)} image files to process.")
    
    # Record start time
    start_time = time.time()
    
    # Process images in parallel
    success_count = process_images_parallel(
        image_files, 
        args.input_dir, 
        args.output_dir, 
        args.scale,
        args.processes
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print(f"\nDownsampling complete in {elapsed_time:.2f} seconds.")
    print(f"Successfully processed {success_count}/{len(image_files)} images.")
    print(f"Downsampled images saved to {args.output_dir}")
    
    if args.processes is None:
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = args.processes
    
    print(f"Used {num_processes} processes for parallel execution.")
    print(f"Average processing time per image: {elapsed_time/len(image_files):.4f} seconds")

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior on Windows
    multiprocessing.freeze_support()
    main()