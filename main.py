import os
import sys
# os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['XLA_FLAGS'] = '--xla_hlo_profile'  # Reduces verbosity of XLA
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {‘0’, ‘1’, ‘2’}
import tensorflow as tf
import absl.logging
import keras_cv
import tensorflow_datasets as tfds
import keras
import variables
from keras import layers
import numpy as np
from keras import regularizers
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn.model_selection import train_test_split
# tf.config.optimizer.set_jit(False)
# tf.get_logger().setLevel('ERROR')
# absl.logging.set_verbosity(absl.logging.ERROR)
# absl.logging.info("starting logs")
# START TENSORBOARD:
#type in console (in classifier directory)> tensorboard --logdir=logs/fit


keras.mixed_precision.set_global_policy('mixed_float16')

tf.experimental.numpy.experimental_enable_numpy_behavior()

CLASSNAMES = variables.CLASSNAMES
NUM_CLASSES = len(CLASSNAMES)

IMAGE_SIZE=variables.IMAGE_SIZE

#usually scaled in powers of 2, reduce this number if running out of vram. increase for faster epochs
BATCH_SIZE = 4

input_shape = variables.INPUTSHAPE  


EPOCHS = variables.EPOCHS

checkpoint_filepath = variables.checkpoint_filepath

DIRECTORY = variables.DIRECTORY
SEED = variables.SEED
VALIDATION_SPLIT= variables.VALIDATION_SPLIT

# save the model if its a better model 
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    # monitor='val_loss',
    # mode='auto',
    # save_best_only=True
    )

# load the data processed by prepData
"""prepData.py MUST BE RUN BEFORE THE MAIN SCRIPT"""

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data("mnist.npz")

def load_and_preprocess_data( batch_size):
    # Load the .npz file
    try:
        data = np.load(DIRECTORY+'/dataset.npz')
    except:
        print("You must run prepData.py in order to train the model")
        exit

    x_data = data['x_data']
    y_data = data['y_data']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=VALIDATION_SPLIT, random_state=SEED)
    x_data = None
    y_data = None 
    x_train = (x_train.astype("float32") / 127.5) - 1
    x_test = (x_test.astype("float32") / 127.5) - 1
    y_train = (y_train.astype("float32") / 127.5) - 1
    y_test = (y_test.astype("float32") / 127.5) - 1

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = np.expand_dims(y_train, -1)
    y_test = np.expand_dims(y_test, -1)

    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    print("x_train shape:", x_train.shape)
    # Shuffle and batch the data
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
    
    return dataset, x_test, y_test

dataset, x_test, y_test = load_and_preprocess_data(BATCH_SIZE)


import model as modelBuilder

gen_G = modelBuilder.get_resnet_generator(name="generator_G")
gen_F = modelBuilder.get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = modelBuilder.get_discriminator(name="discriminator_X")
disc_Y = modelBuilder.get_discriminator(name="discriminator_Y")
model = modelBuilder.CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=modelBuilder.generator_loss_fn,
    disc_loss_fn=modelBuilder.discriminator_loss_fn,
)
try:
    model.load_weights(checkpoint_filepath)
    print("model loaded")
except Exception as e:
    print(e)
    print("model failed to load, training from scratch")
# create a test strip displayed in tensorboard
def show_test_dataset(a, b):
    # gen_G = modelBuilder.get_resnet_generator(name="generator_G")
    variables.epochcounter+=1   
    if (variables.epochcounter%16!=0):
        return
    
    figure = plt.figure(figsize=(10,10))
    # result = model.predict(x_test)
    result = model.gen_G(x_test)
    for i in range(36):
        # name = "undecided"
        
        plt.subplot(6, 6, i+1)
        plt.xticks([]) 
        plt.yticks([])
        plt.grid(False)
        
        if (i%3 == 0):
            img = (np.squeeze(x_test[i//3]))
            img = (img * 127.5 + 127.5).astype(np.uint8)
        elif(i%3 == 1):
            img = np.squeeze(y_test[(i-1)//3])
            img = (img * 127.5 + 127.5).astype(np.uint8)
        else:
            
            img = (np.squeeze(result[(i-2)//3]))
            img = (img * 127.5 + 127.5).astype(np.uint8)
        plt.imshow(img)
    # figure.subplots_adjust(hspace=0.2)
    fullImage = variables.plot_to_image(figure)
    with file_writer.as_default():
        tf.summary.image("latest classifications", fullImage, step=variables.epochcounter)
    return
# set up callbacks to run functions during events in training

log_dir = DIRECTORY+"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)
# show_test_dataset(1,3)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
drawImages = keras.callbacks.LambdaCallback(on_epoch_end= show_test_dataset)

print("training model")
# Train your model
model.fit(dataset, batch_size=BATCH_SIZE, epochs=10000, callbacks=[ model_checkpoint_callback, tensorboard_callback, drawImages]) #drawImages,
# model.fit(tf.data.Dataset.zip((train_horses, train_zebras)),epochs=1,callbacks=[plotter, model_checkpoint_callback],)\
print("completed training")
