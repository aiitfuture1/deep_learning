from __future__ import division
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D


img_height = 66
img_width = 200

image_size = (img_height, img_width)
input_shape = image_size + (3, )
#print(input_shape)
def create_model():
    # ---------------------------------------------------- #
    # Step 1: Initializing the cnn as a sequential layers
    # ---------------------------------------------------- #
    cnn = Sequential()
    # ---------------------------------------------------- #
    # Step 2: Convolutional Layers
    #      2.1: Convolution
    #      2.2: Pooling
    # ---------------------------------------------------- #
    # First Convolutional Layer
    cnn.add(Convolution2D(24, (5, 5), padding='same', activation='relu', input_shape=input_shape))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Second Convolutional Layer
    cnn.add(Convolution2D(36, (5, 5), padding='same', activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Third Convolutional Layer
    cnn.add(Convolution2D(48, (5, 5), padding='same', activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Fourth Convolutional Layer
    cnn.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Fifth Convolutional Layer
    cnn.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # ---------------------------------------------------- #
    # Step 3: Flattening the CNN
    # ---------------------------------------------------- #
    cnn.add(Flatten())
    # ---------------------------------------------------- #
    # Step 4: Full Connection
    # ---------------------------------------------------- #
    # First fully-connected layer
    cnn.add(Dense(units=1164, activation='relu'))
    # Second fully-connected layer
    cnn.add(Dense(units=100, activation='relu'))
    # Third fully-connected layer
    cnn.add(Dense(units=50, activation='relu'))
    # Fourth fully-connected layer
    cnn.add(Dense(units=10, activation='relu'))
    # ---------------------------------------------------- #
    # Step 5: Output Layer
    # ---------------------------------------------------- #
    cnn.add(Dense(units=1, activation='relu'))
    # ---------------------------------------------------- #
    # Step 6: Compile the CNN
    # ---------------------------------------------------- #
    cnn.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print('--------------------------------------------------------------')
    print "Model is created and compiled"
    print('--------------------------------------------------------------')
    print cnn.summary()
    print('--------------------------------------------------------------')
    return cnn