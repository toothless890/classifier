import numpy as np
import cv2
import os
import multiprocessing

import variables

DIRECTORY = variables.DIRECTORY

def process_image(img_path, output_folder_fake, output_folder_real, count):
    if count > 1000:
        return count
    try:
        img = cv2.imread(img_path)  # Use OpenCV to read the image
        height, width, _ = img.shape
        middle_index = width // 2
        left_half = img[:, :middle_index, :]
        right_half = img[:, middle_index:, :]
        
        cv2.imwrite(os.path.join(output_folder_fake, f"{count}.png"), left_half)
        cv2.imwrite(os.path.join(output_folder_real, f"{count}.png"), right_half)

        if count % 50 == 0:
            print(count)
        return count + 1
    except Exception as e:
        print(f"Failed to process image {img_path}: {e}")
        return count

def process_images():
    input_folder = os.path.join(DIRECTORY, 'preprocess')
    output_folder_fake = os.path.join(DIRECTORY, 'training', 'fake')
    output_folder_real = os.path.join(DIRECTORY, 'training', 'real')
    
    os.makedirs(output_folder_fake, exist_ok=True)
    os.makedirs(output_folder_real, exist_ok=True)
    
    count = 0
    subfolders = sorted(os.listdir(input_folder))
    
    # Use multiprocessing to speed up image processing
    for folder in subfolders:
        img_files = [os.path.join(input_folder, folder, img_file) for img_file in os.listdir(os.path.join(input_folder, folder))]
        with multiprocessing.Pool() as pool:
            results = pool.starmap(process_image, [(img_file, output_folder_fake, output_folder_real, count + i) for i, img_file in enumerate(img_files)])
            count = max(results)  # Update the count after processing all images 
            

process_images()
