import matplotlib.pyplot as plt
import itertools
import numpy as np
import tensorflow as tf
import io
import os
# This file is the main file for storing variables that need to be replicated across multiple scripts
# Brief descriptions will be provided for each changable variable

"""
    MAKE SURE TO CHANGE DIRECTORY TO THE DIRECTORY OF THIS FILE
    Description of folder structure
    /classifier/
        /training/  < PUT TRAINING DATA IN THIS FOLDER
            /[category1] these categories can be named anything and have as many as you want
            /[category2]  
            /[category[n]]
            
        /input/     < PUT DATA THAT YOU WANT TO SORT HERE
            /[folder1] these can also be named anything
            /[folder2] the code that turns the training data into a dataset is the same as what turns the
            /[folder[n]] data that is desired to be sorted so it must be in a subfolder (the names of the folder doesnt change much)
            
        /output/   > DATA GETS SORTED INTO CATEGORIES FROM /training/
            /[category1]
            /[category2]  
            /[category[n]]
            
        /logs/
            /fit/ this is for tensorboard, please start tensorboard by running the shell command 
                                            "tensorboard --logdir=logs/fit" 
                                            in the root classifier directory (through the shell, not through python)
""" 


global DIRECTORY # of the dataset

########################################
DIRECTORY = "/home/ashto/classifier/"       ## change this to your folder
########################################

global CLASSNAMES
# 
# take the names of the folders in the training folder
# automatically processes all names in the training folder
CLASSNAMES = sorted(os.listdir(DIRECTORY+"training"))

# CLASSNAMES = ["1","2","3","4","5","6","7","8","9","0"]

#How many epochs to run for?
# set to high number to run until stopped
# (the NN gets saved in a checkpoint each epoch with an improvement, so you can resume training
#  at the expense of it starting a new tensorboard log)
global EPOCHS
EPOCHS = 1000000

# THIS DETERMINES THE SIZE OF THE IMAGE THAT THE NEURAL NETWORK TRAINS ON
# smaller values will run faster, but may be more limited in accuracy. 
# this also affects the size of the images in the test strip in tensorboard
global IMAGE_SIZE
IMAGE_SIZE = (128, 128)

# determines the 'shape' of the array formed for each image
# 3 means 3 color values (RGB)
# change to 1 for grayscale (you also must change 'RGB' to 'L' in prepData.py)
global INPUTSHAPE
INPUTSHAPE = (IMAGE_SIZE[1], IMAGE_SIZE[1], 3)


global SEED # seed for randomizing dataset order
SEED = 23265


# Usually datasets are split into training data and validating data
# SPLIT = the percentage (0.0-1.0) of data to be saved for validation
global VALIDATION_SPLIT 
SPLIT = 0.2

# validation data is not used to train the model, 
# but used to check that it can apply it's knowledge to images it hasnt trained on

# this counts the epochs, used primarily for generating test strips (series of images with categorizations and confidence levels)
global epochcounter
epochcounter = -1
# -1 so that it generates a test strip before it starts training, otherwise it will wait [default is 16] epochs

global checkpoint_filepath #where to put the weights of the model. Rename file to avoid overwriting if you want to preserve old models 
checkpoint_filepath = DIRECTORY + 'checkpoint.weights.h5'
# checkpoint_filepath = 'C:/Users/ashto/Documents/code/Python/classifier/sassunatest/checkpoint.weights.h5'


#Helper functions


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    """ CHATGPT + TENSORFLOW DOCS"""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image