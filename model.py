from __future__ import division
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, BatchNormalization
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.regularizers import l2 # L2-regularisation


def atan(x):
    return tf.multiply(tf.atan(x), 2)


img_height = 66
img_width = 200
l2_lambda = 0.001

image_size = (img_height, img_width)
input_shape = image_size + (3, )

def create_model():
    img_input = Input(input_shape)
    # ---------------------------------------------------- #
    # Step 1: Initializing the cnn as a sequential layers
    # ---------------------------------------------------- #
    model = Sequential()
    # ---------------------------------------------------- #
    # Step 2: Convolutional Layers
    #      2.1: Convolution
    #      2.2: Pooling
    # ---------------------------------------------------- #
    # First Convolutional Layer
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same", activation='relu',\
                     input_shape=input_shape, kernel_regularizer=l2(l2_lambda))) # padding = "valid"
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # Second Convolutional Layer
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="same", activation='relu', \
                     kernel_regularizer = l2(l2_lambda)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # Third Convolutional Layer
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="same", activation='relu', \
                     kernel_regularizer=l2(l2_lambda)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # Fourth Convolutional Layer
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', \
                     kernel_regularizer=l2(l2_lambda)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # Fifth Convolutional Layer
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', \
                     kernel_regularizer=l2(l2_lambda)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # sixth Convolutional Layer
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', \
                     kernel_regularizer=l2(l2_lambda)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # ---------------------------------------------------- #
    # Step 3: Flattening the CNN
    # ---------------------------------------------------- #
    model.add(Flatten())
    # ---------------------------------------------------- #
    # Step 4: Full Connection
    # ---------------------------------------------------- #
    # First fully-connected layer
    model.add(Dense(units=1164, activation='relu', \
                    kernel_regularizer=l2(l2_lambda)))
    # Second fully-connected layer
    model.add(Dense(units=100, activation='relu', \
                    kernel_regularizer=l2(l2_lambda)))
    model.add((Dropout(0.5)))
    # Third fully-connected layer
    model.add(Dense(output_dim=50, activation='relu', \
                    kernel_regularizer=l2(l2_lambda)))
    model.add((Dropout(0.5)))
    # Fourth fully-connected layer
    model.add(Dense(units=10, activation='relu', \
                    kernel_regularizer=l2(l2_lambda)))
    model.add((Dropout(0.5)))
    # ---------------------------------------------------- #
    # Step 5: Output Layer
    # ---------------------------------------------------- #
    model.add(Dense(units=1, activation=atan, \
                    kernel_regularizer=l2(l2_lambda)))
    # ---------------------------------------------------- #
    # Step 6: Compile the CNN
    # ---------------------------------------------------- #
    model.compile(optimizer=Adam(lr=.0001), loss="mse", metrics=['accuracy'])

    print(model.summary())
    return model
