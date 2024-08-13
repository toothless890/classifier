import os
import sys
# os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['XLA_FLAGS'] = '--xla_hlo_profile'  # Reduces verbosity of XLA
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {‘0’, ‘1’, ‘2’}
import tensorflow as tf
# import absl.logging
# import keras_cv
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import keras
import variables
# from keras import layers
import numpy as np
# from keras import regularizers
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

# START TENSORBOARD:
#type in console (in classifier directory)> tensorboard --logdir=logs/fit

from keras import backend as K
K.clear_session()

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


keras.mixed_precision.set_global_policy('mixed_float16')

tf.experimental.numpy.experimental_enable_numpy_behavior()

CLASSNAMES = variables.CLASSNAMES
NUM_CLASSES = len(CLASSNAMES)

IMAGE_SIZE=variables.IMAGE_SIZE
orig_img_size = (286, 286)
#usually scaled in powers of 2, reduce this number if running out of vram. increase for faster epochs
BATCH_SIZE = 1

INPUTSHAPE = variables.INPUTSHAPE  
RESHAPE = variables.RESHAPE

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

def load_and_preprocess_data():
    # Load the .npz file
    try:
        data = np.load(DIRECTORY+'/dataset.npz')
    except:
        print("You must run prepData.py in order to train the model")
        exit
    
    
    x_train = data['x_train']
    y_train = data['y_train']
    
    x_test = data['x_test']
    y_test = data['y_test']
    
    x_train = x_train.reshape(RESHAPE)
    x_train = x_train.astype(np.float32)
    x_train = (x_train / 127.5) - 1
    
    y_train = y_train.reshape(RESHAPE)
    y_train = y_train.astype(np.float32)
    y_train = (y_train / 127.5) - 1
    
    x_test = x_test.reshape(RESHAPE)
    x_test = x_test.astype(np.float32)
    x_test = (x_test / 127.5) - 1
    
    y_test = y_test.reshape(RESHAPE)
    y_test = y_test.astype(np.float32)
    y_test = (y_test / 127.5) - 1
    
    
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, x_test, y_test

def warmup_gpu():
    x = tf.random.normal([1, 64, 64, 3])
    y = tf.random.normal([1, 64, 64, 3])
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2))
    ])
    model(x)
    return


# def normalize_img(img):
#     img = tf.cast(img, dtype=tf.float32)
#     # Map values in the range [-1, 1]
#     return (img / 127.5) - 1.0

# def augment_image(image):
#     # Randomly flip the image horizontally
#     image = tf.image.random_flip_left_right(image)
   
#     # Randomly adjust brightness
#     image = tf.image.random_brightness(image, max_delta=0.1)
#     # Randomly adjust contrast
#     image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
#     # Randomly adjust saturation
#     image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
#     # Randomly adjust hue
#     image = tf.image.random_hue(image, max_delta=0.1)
    
#     image = tf.image.random_crip(image, size=[*INPUTSHAPE])
#     return image

dataset, x_test, y_test = load_and_preprocess_data()

import model as modelBuilder

gen_G = modelBuilder.get_resnet_generator(name="generator_G")
gen_F = modelBuilder.get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = modelBuilder.get_discriminator(name="discriminator_X")
disc_Y = modelBuilder.get_discriminator(name="discriminator_Y")

model = modelBuilder.CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )
scheduler = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.002,decay_steps=1000,decay_rate=0.9)

try:
    model.load_weights(checkpoint_filepath)
    print("model loaded")
except Exception as e:
    print(e)
    print("model failed to load, training from scratch")
    
    
model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.5),
    gen_loss_fn=modelBuilder.generator_loss_fn,
    disc_loss_fn=modelBuilder.discriminator_loss_fn,
)
# create a test strip displayed in tensorboard
def show_test_dataset(a, b):
    # Increment the epoch counter
    variables.epochcounter += 1

    # Perform actions only every few epochs
    if variables.epochcounter % 2 != 0: 
        return
    
    # Generate images
    # result = model.gen_G(x_test)
    
    # Plot and save images
    
    rows = 4
    cols = 4
    num_images = rows*cols
    #TODO: make this use batches, of a changable size (i tried isolating one image and it didnt work, investigate more!) 
    result = model.gen_G(x_test[0:(num_images//2)])
    figure = plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        if i % 2 == 0:
            img = np.squeeze(x_test[i // 2])
        # elif i % 3 == 1:
        #     img = np.squeeze(y_test[(i - 1) // 3])
        else:
            img = np.squeeze(result[(i - 1) // 2])
        
        img = (img * 127.5 + 127.5).astype(np.uint8)
        plt.imshow(img)
    # Save the figure to an image and write to TensorBoard
    full_image = variables.plot_to_image(figure)
    with file_writer.as_default():
        tf.summary.image("latest classifications", full_image, step=variables.epochcounter)

    # Clear the figure to free memory
    plt.close(figure)
# set up callbacks to run functions during events in training

log_dir = DIRECTORY+"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)
# show_test_dataset(1,3)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
drawImages = keras.callbacks.LambdaCallback(on_epoch_end= show_test_dataset)

print("warming up ")
warmup_gpu()

print("training model")
# Train your model
model.fit(dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[model_checkpoint_callback, tensorboard_callback, drawImages]) #drawImages,
print("completed training")
