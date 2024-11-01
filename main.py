import os
import sys
import io
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
#usually scaled in powers of 2, reduce this number if running out of vram. increase for faster epochs
BATCH_SIZE = 1


INPUTSHAPE = variables.INPUTSHAPE  
RESHAPE = variables.RESHAPE


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


def augment_image(image, label):
    # Apply random horizontal flip
    image = tf.image.random_flip_left_right(image)
    label = tf.image.random_flip_left_right(label)

    # Apply random zoom (cropping and resizing back to the original size)
    image = tf.image.resize(image,size=[ IMAGE_SIZE[0] + 20, IMAGE_SIZE[1] + 20])  # Add padding
    image = tf.image.random_crop(image, size=[*INPUTSHAPE])  # Crop back to original size
    
    label = tf.image.resize(label, size= [IMAGE_SIZE[0] + 20, IMAGE_SIZE[1] + 20])  # Add padding
    label = tf.image.random_crop(label, size=[*INPUTSHAPE])  # Crop back to original size

    
    # noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
    # image = tf.add(image, noise)
    # label = tf.add(label, noise)
    
    image = tf.image.random_hue(image, 0.05)
    label = tf.image.random_hue(label, 0.05)
    
    image = tf.image.random_saturation(image, 0.9, 1.1)
    label = tf.image.random_saturation(label, 0.9, 1.1)
    
    # Apply random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.1)
    label = tf.image.random_brightness(label, max_delta=0.1)
    
    # Apply random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    label = tf.image.random_contrast(label, lower=0.9, upper=1.1)
    
    # image = tf.image.random_jpeg_quality(image, 80, 100)
    # label = tf.image.random_jpeg_quality(label, 80, 100)
    
    return image, label


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
    
    dataset_x = tf.data.Dataset.from_tensor_slices(x_train)
    dataset_y = tf.data.Dataset.from_tensor_slices(y_train)
    
    dataset_x = dataset_x.shuffle(256, reshuffle_each_iteration=True)
    
    dataset_y = dataset_y.shuffle(256, reshuffle_each_iteration=True)
    # dataset = dataset.shuffle(buffer_size=256)
    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, x_test, y_test

def warmup_gpu():
    x = tf.random.normal([1, 256, 256, 3])
    y = tf.random.normal([1, 256, 256, 3])
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2))
    ])
    model(x)
    return

class CustomLossScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, final_lr, regularization_weight, total_epochs, decay_start_epoch=100):
        super(CustomLossScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.regularization_weight = regularization_weight
        self.total_epochs = total_epochs
        self.decay_start_epoch = decay_start_epoch
        self.lr = initial_lr

    def on_epoch_begin(self, epoch, logs=None):
        # Linearly decay learning rate after decay_start_epoch
        if epoch >= self.decay_start_epoch:
            decay_factor = (epoch - self.decay_start_epoch) / (self.total_epochs - self.decay_start_epoch)
            self.lr = self.initial_lr - decay_factor * (self.initial_lr - self.final_lr)
        
        # Adjust regularization weight (e.g., increase as training progresses)
        reg_weight = self.regularization_weight * (1 - epoch / self.total_epochs)
        
        # Apply the new learning rate to the optimizers
        self.model.gen_G_optimizer.learning_rate = self.lr
        self.model.gen_F_optimizer.learning_rate = self.lr
        self.model.disc_X_optimizer.learning_rate = self.lr
        self.model.disc_Y_optimizer.learning_rate = self.lr

        
        # If you are using custom regularization in the loss function, you can pass the reg_weight to your model
        self.model.regularization_weight = reg_weight

        print(f"Epoch {epoch + 1}/{self.total_epochs} - Learning Rate: {self.lr:.6f} - Regularization Weight: {reg_weight:.6f}")

# Usage in your training script

# Parameters
initial_lr = 0.0004
final_lr = 0.00001
regularization_weight = 0.01  # Adjust based on the desired smoothing effect
total_epochs = 600
decay_start_epoch = 100  # Start decaying after 100 epochs

# Initialize the custom scheduler
custom_loss_scheduler = CustomLossScheduler(
    initial_lr=initial_lr,
    final_lr=final_lr,
    regularization_weight=regularization_weight,
    total_epochs=total_epochs,
    decay_start_epoch=decay_start_epoch
)

dataset, x_test, y_test = load_and_preprocess_data()

import model as modelBuilder
downsample_blocks = 2
residual_blocks = 9
upsample_blocks = 2
disc_downsamples = 2
gen_G = modelBuilder.get_resnet_generator(name="generator_G", num_downsampling_blocks= downsample_blocks, num_upsample_blocks = upsample_blocks, num_residual_blocks = residual_blocks)
gen_F = modelBuilder.get_resnet_generator(name="generator_F", num_downsampling_blocks= downsample_blocks, num_upsample_blocks = upsample_blocks, num_residual_blocks = residual_blocks)

# Get the discriminators
disc_X = modelBuilder.get_discriminator(name="discriminator_X", num_downsampling= disc_downsamples)
disc_Y = modelBuilder.get_discriminator(name="discriminator_Y", num_downsampling= disc_downsamples)

model = modelBuilder.CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y, lambda_cycle=10.0, lambda_identity = 0.5)



try:
    model.load_weights(checkpoint_filepath)
    print("model loaded")
except Exception as e:
    print(e)
    print("model failed to load, training from scratch")
    
model.compile(
    gen_G_optimizer=keras.optimizers.AdamW(learning_rate=initial_lr, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.AdamW(learning_rate=initial_lr, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.AdamW(learning_rate=initial_lr, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.AdamW(learning_rate=initial_lr, beta_1=0.5),
    gen_loss_fn=modelBuilder.generator_loss_fn,
    disc_loss_fn=modelBuilder.discriminator_loss_fn,
)

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

# create a test strip displayed in tensorboard
def show_test_dataset(a, b):
    # Increment the epoch counter
    variables.epochcounter += 1

    # Perform actions only every few epochs
    # if variables.epochcounter % 2 != 0: 
    #     return
    # Plot and save images
    
    rows = 6
    cols = 6
    num_images = rows*cols
    #TODO: make this use batches, of a changable size (i tried isolating one image and it didnt work, investigate more!) 
    image = None
    result = None
    figure = plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        if i % 3 == 0:
            img = np.squeeze(x_test[i // 3])
        elif i % 3 == 1:
            image = x_test[((i - 1) // 3):((i - 1) // 3)+1]
            result = model.gen_G(image)
            img = np.squeeze(result)
            
        else:
            # image = x_test[((i - 1) // 3):((i - 1) // 3)+1]
            # result = model.gen_G(image)
            img = np.squeeze(model.gen_F(result))
            
        img = (img * 127.5 + 127.5).astype(np.uint8)
        plt.imshow(img)
    # Save the figure to an image and write to TensorBoard
    full_image = plot_to_image(figure)
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
model.fit(dataset, batch_size=BATCH_SIZE, epochs=total_epochs, callbacks=[model_checkpoint_callback, tensorboard_callback, drawImages, custom_loss_scheduler]) #drawImages,
print("completed training")