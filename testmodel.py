
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
from keras import backend as K
K.clear_session()

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


CLASSNAMES = variables.CLASSNAMES

NUM_CLASSES = len(CLASSNAMES)
IMAGE_SIZE=variables.IMAGE_SIZE
input_shape = variables.INPUTSHAPE
checkpoint_filepath = variables.checkpoint_filepath
DIRECTORY = variables.DIRECTORY
SEED = variables.SEED
RESHAPE = variables.RESHAPE
# makeNewData = False
# makeNewData = True     # comment this out to toggle

#take data from /input/ and compile it to a dataset file
# files should be stored in a subfolder 
#               (maybe ill add another function, but im repurposing code
#               that takes in subfolders for training purposes as it was
#               the easiest, quickest solution)
#                   
# if makeNewData:
#     data_dir = DIRECTORY +'/input/'
#     x_data, y_data, counts = prepData.load_data(data_dir)
#     x_data = x_data.reshape(RESHAPE)
#     np.savez_compressed(DIRECTORY+'newData.npz', x_data=x_data, y_data=y_data)

data = np.load(DIRECTORY+'/dataset.npz')
x_test = data['x_test']
y_test = data['y_test']

x_test = x_test.reshape(RESHAPE)
x_test = x_test.astype(np.float32)
x_test = (x_test / 127.5) - 1

y_test = y_test.reshape(RESHAPE)
y_test = y_test.astype(np.float32)
y_test = (y_test / 127.5) - 1

import model as modelBuilder

gen_G = modelBuilder.get_resnet_generator(name="generator_G")
gen_F = modelBuilder.get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = modelBuilder.get_discriminator(name="discriminator_X")
disc_Y = modelBuilder.get_discriminator(name="discriminator_Y")
model = modelBuilder.CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )
scheduler = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=1000,decay_rate=0.9)
model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.6),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.6),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.6),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.6),
    gen_loss_fn=modelBuilder.generator_loss_fn,
    disc_loss_fn=modelBuilder.discriminator_loss_fn,
)
try:
    model.load_weights(checkpoint_filepath)
    print("model loaded")
except Exception as e:
    print("model failed to load, exiting")
    exit()

result = model.gen_G(x_test)
    
# Plot and save images
num_images = 36
num_rows = 6
num_cols = 6

figure = plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    if i % 3 == 0:
        img = np.squeeze(x_test[i // 3])
    elif i % 3 == 1:
        img = np.squeeze(y_test[(i - 1) // 3])
    else:
        img = np.squeeze(result[(i - 2) // 3])
    
    img = (img * 127.5 + 127.5).astype(np.uint8)
    plt.imshow(img)
    plt.savefig("latestModel.png")