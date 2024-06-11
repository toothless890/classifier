
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
    x_data, y_data = prepData.load_data(data_dir)
    x_data = x_data.reshape((-1, 128, 128, 3))
    np.savez_compressed(DIRECTORY+'newData.npz', x_data=x_data, y_data=y_data)

data = np.load(DIRECTORY+'newData.npz')
x_data = data['x_data']
y_data = data['y_data']

# x_data = (x_data.astype("float32") /255)
y_data = np.expand_dims(y_data, -1)



data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5),
        layers.RandomBrightness(0.002),
        layers.RandomContrast(0.002),
        layers.RandomZoom(0.002),
        # layers.RandomCrop(0.2, 0.2),
        # layers.RandomTranslation(0.2, 0.2)

    ]
)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        data_augmentation,
        layers.Conv2D(32, kernel_size=(3, 3), activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(256, kernel_size=(3, 3), activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Dropout(0.5),
        layers.Dense(128, activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax") ,
    ]
)


model.load_weights(checkpoint_filepath)


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
    
    
    
