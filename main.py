import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

import tensorflow as tf
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

# START TENSORBOARD:
#type in console (in classifier directory)> tensorboard --logdir=logs/fit

tf.experimental.numpy.experimental_enable_numpy_behavior()

CLASSNAMES = variables.CLASSNAMES
NUM_CLASSES = len(CLASSNAMES)

IMAGE_SIZE=variables.IMAGE_SIZE
#usually scaled in powers of 2, reduce this number if running out of vram. increase for faster epochs
BATCH_SIZE = 32

input_shape = variables.INPUTSHAPE  


EPOCHS = variables.EPOCHS

checkpoint_filepath = variables.checkpoint_filepath

DIRECTORY = variables.DIRECTORY
SEED = variables.SEED
VALIDATION_SPLIT= variables.VALIDATION_SPLIT

# save the model if its a better model 
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    # save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# currently unused. custom optimizer
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.001,
#     decay_steps=10000,
#     decay_rate=0.99)

# optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.99) 

# load the data processed by prepData
"""prepData.py MUST BE RUN BEFORE THE MAIN SCRIPT"""
try:
    data = np.load(DIRECTORY+'/dataset.npz')
except:
    print("You must run prepData.py in order to train the model")
    exit

x_data = data['x_data']
y_data = data['y_data']
bias = data['bias']


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y_data)

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data("mnist.npz")

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

#augment the data in order to simulate more training data and reduce overfitting. 
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5),
        layers.RandomBrightness(0.002),
        layers.RandomContrast(0.002),
        layers.RandomZoom(0.002),
        # layers.RandomTranslation(0.2, 0.2)
    ]
)

#try to load a model if avaliable to resume training

try:
    model = keras.saving.load_model(checkpoint_filepath)
    print("model loaded")

except:
    print("loading failed")

#create the network. modify at risk of your sanity
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            data_augmentation,
            layers.Conv2D(32, kernel_size=(3, 3), activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # layers.Conv2D(128, kernel_size=(3, 3), activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            
            
            # layers.Conv2D(256, kernel_size=(3, 3), activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
            # layers.MaxPooling2D(pool_size=(2, 2)),
            # layers.Dropout(0.5),
            
            # layers.Dense(32, activation=keras.activations.leaky_relu, kernel_regularizer=regularizers.l2(1e-2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax") ,
        ]
    )

    # compile the model  for training
    model.compile(loss="categorical_crossentropy", optimizer= "adamw", metrics=["accuracy"])



# create a test strip displayed in tensorboard
def show_test_dataset(a, b):
    variables.epochcounter+=1   
    if (variables.epochcounter%16!=0):
        return
    
    figure = plt.figure(figsize=(10,15))
    result = model.predict(x_test)
    for i in range(30):
        name = "undecided"
        for x in CLASSNAMES:
            if (result[i][CLASSNAMES.index(x)]==(max(result[i]))):
                name = x
                break
        plt.subplot(5, 6, i+1, title = str(name)+ ": \n"+ str(max(result[i])))
        plt.xticks([]) 
        plt.yticks([])
        plt.grid(False)
        
        
        img = (np.squeeze(x_test[i]))
        plt.imshow(img)
    figure.subplots_adjust(hspace=0.2)
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
class_weight = {}
try:
    
    for x in range(len(bias)):
        class_weight[CLASSNAMES[x]] = bias[x]
except:
    pass

# Train your model
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=class_weight, validation_data = (x_test, y_test), callbacks=[drawImages, model_checkpoint_callback, tensorboard_callback])

