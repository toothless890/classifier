import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
import variables
import math

VALIDATION_SPLIT = variables.VALIDATION_SPLIT
SEED = variables.SEED
DIRECTORY = variables.DIRECTORY
IMAGE_SIZE = variables.IMAGE_SIZE
INPUTSHAPE = variables.INPUTSHAPE
RESHAPE = variables.RESHAPE

def copy_with_structure(source_dir, destination_dir):
    """ Copy files from source_dir to destination_dir, preserving the directory structure. """
    
    
    # Create the destination directory if it does not exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Copy the file
    shutil.copy2(source_dir, destination_dir)


def load_data(data_dir):
    """ Load data from the directory structure. """
    imagein = []
    imageout = []
    
    class_names = sorted(os.listdir(data_dir))
    for label, class_dir in enumerate(class_names): 
        inpath = os.path.join(data_dir, "fake")
        outpath = os.path.join(data_dir, "real")
        if not os.path.isdir(inpath):
            continue
        print(class_dir, ": ", str(len(os.listdir(inpath))))
        count = 0
    for img_file in os.listdir(inpath):
        if count >= 1000:
            break
        count +=1
        img_path = os.path.join(inpath, img_file)
        try:
            img = Image.open(img_path).convert('RGB')   # CHANGE 'RGB' TO 'L' FOR GRAYSCALE, 
                                                        # MAKE SURE TO CHANGE global INPUTSHAPE in variables.py
            img = img.resize(IMAGE_SIZE, resample=Image.Resampling.BICUBIC  )  # resize to 28x28 pixels
            img_array = np.array(img)
            imagein.append(img_array)

        except Exception as e:
            print(f"Failed to process image {img_path}: {e}")
            
        img_path = os.path.join(outpath, img_file)
        try:
            img = Image.open(img_path).convert('RGB')   # CHANGE 'RGB' TO 'L' FOR GRAYSCALE, 
                                                        # MAKE SURE TO CHANGE global INPUTSHAPE in variables.py
            img = img.resize(IMAGE_SIZE, resample=Image.Resampling.BICUBIC  )  # resize to 28x28 pixels
            img_array2 = np.array(img)
            imageout.append(img_array2)

        except Exception as e:
            print(f"Failed to process image {img_path}: {e}")
            
        
    return np.array(imagein), np.array(imageout)

if __name__ == "__main__":

    # Load all data
    data_dir = DIRECTORY+'training'
    x_data, y_data = load_data(data_dir)
    
    

    # Split the data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y_data)

    # Normalize the data (its more efficient)
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0 # moved back to main files for data compatibility
    
    # Ensure the data shape is correct for Keras
    # x_train = x_train.reshape(RESHAPE)
    # x_test = x_test.reshape(RESHAPE)
    x_data = x_data.reshape(RESHAPE)
    y_data = y_data.reshape(RESHAPE)
    # print('x_train shape:', x_train.shape)
    # print('y_train shape:', y_train.shape)
    # print('x_test shape:', x_test.shape)
    # print('y_test shape:', y_test.shape)

    np.savez_compressed(DIRECTORY+'/dataset.npz', x_data = x_data, y_data = y_data)#x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print('complete')