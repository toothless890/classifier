import matplotlib.pyplot as plt
import itertools
import numpy as np
import tensorflow as tf
import io
import os
# This file is the main file for storing variables that need to be replicated across multiple scripts
# Brief descriptions will be provided for each changable variable


global DIRECTORY # of the dataset

########################################
DIRECTORY = "/home/ashto/cycle/"       ## change this to your folder
########################################

global CLASSNAMES
# 
# take the names of the folders in the training folder
# automatically processes all names in the training folder
CLASSNAMES = sorted(os.listdir(DIRECTORY+"training"))

#overrride
# CLASSNAMES = ["0", "1","2","3","4","5","6","7","8","9"]

#How many epochs to run for?
# set to high number to run until stopped
# (the NN gets saved in a checkpoint each epoch with an improvement, so you can resume training
#  at the expense of it starting a new tensorboard log)
global EPOCHS
EPOCHS = 300

# THIS DETERMINES THE SIZE OF THE IMAGE THAT THE NEURAL NETWORK TRAINS ON
# smaller values will run faster, but may be more limited in accuracy. 
# this also affects the size of the images in the test strip in tensorboard
global IMAGE_SIZE
IMAGE_SIZE = (256, 256)

# determines the 'shape' of the array formed for each image
# 3 means 3 color values (RGB)
# change to 1 for grayscale (you also must change 'RGB' to 'L' in prepData.py)
global INPUTSHAPE
INPUTSHAPE = (IMAGE_SIZE[1], IMAGE_SIZE[1], 3)
RESHAPE = (-1, IMAGE_SIZE[1], IMAGE_SIZE[1], 3)


global SEED # seed for randomizing dataset order
SEED = 23265


# Usually datasets are split into training data and validating data
# SPLIT = the percentage (0.0-1.0) of data to be saved for validation
global VALIDATION_SPLIT 
VALIDATION_SPLIT = 0.02

# validation data is not used to train the model, 
# but used to check that it can apply it's knowledge to images it hasnt trained on

# this counts the epochs, used primarily for generating test strips (series of images that show what the model is )
global epochcounter
epochcounter = -1
# -1 so that it generates a test strip before it starts training, otherwise it will wait [default is 16] epochs

global checkpoint_filepath #where to put the weights of the model. Rename file to avoid overwriting if you want to preserve old models 
# checkpoint_filepath = DIRECTORY + 'model.weights.h5'
checkpoint_filepath = DIRECTORY + 'model.weights.h5'


#Helper functions
#TODO: move this back to main.py as there's no need to use it and have to import tensorflow in prepData and other preprocessing steps

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
if __name__ == "__main__":
    print("this is the variables file and does not need to be run. \n please make sure to run the main, preprocess, or prep files instead!")