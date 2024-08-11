
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from keras import layers
import tensorflow as tf
import keras_cv
import tensorflow_datasets as tfds
import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import variables
import os
import shutil
import prepData
from keras import regularizers


CLASSNAMES = variables.CLASSNAMES

NUM_CLASSES = len(CLASSNAMES)
IMAGE_SIZE=variables.IMAGE_SIZE
input_shape = variables.INPUTSHAPE
checkpoint_filepath = variables.checkpoint_filepath
DIRECTORY = variables.DIRECTORY
SEED = variables.SEED
RESHAPE = variables.RESHAPE
makeNewData = False
makeNewData = True     # comment this out to toggle

#take data from /input/ and compile it to a dataset file
# files should be stored in a subfolder 
#               (maybe ill add another function, but im repurposing code
#               that takes in subfolders for training purposes as it was
#               the easiest, quickest solution)
#                   
if makeNewData:
    data_dir = DIRECTORY +'/input/'
    x_data, y_data, counts = prepData.load_data(data_dir)
    x_data = x_data.reshape(RESHAPE)
    np.savez_compressed(DIRECTORY+'newData.npz', x_data=x_data, y_data=y_data)

data = np.load(DIRECTORY+'newData.npz')
x_data = data['x_data']
y_data = data['y_data']

x_data = (x_data.astype("float32") /255)
y_data = np.expand_dims(y_data, -1)


model = keras.saving.load_model(checkpoint_filepath)
# model.load_weights(checkpoint_filepath)


# (2, 64, 64, 3) tf.Tensor([b'ok' b'not ok'], shape=(2,), dtype=string)
result = model.predict(x_data)


for i in range(len(x_data)):
    path = y_data[i][0]
    folder = "uncdecided"
    for x in CLASSNAMES:
        if (result[i][CLASSNAMES.index(x)]==(max(result[i]))):
            # print(result[i])
            folder = x
            break
    prepData.copy_with_structure(path, DIRECTORY+"output/"+folder)
    
    
    
