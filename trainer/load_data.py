from __future__ import division
from collections import OrderedDict
import re
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime

from tensorflow.python.lib.io import file_io
from keras.callbacks import ModelCheckpoint, TensorBoard, History
from keras.preprocessing.image import load_img, img_to_array

import model

def train_model(train_file = os.path.abspath('.') , job_dir = os.path.abspath('.'), **args):
    # -----------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------
    # Path to files
    img_data_dir = train_file + '/imgs'
    csv_data_dir = train_file + '/csv'
    #out_dir = os.path.abspath('./output')
    save_dir = job_dir + '/models'

    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('--------------------------------------------------------------')
    print('Using training images located at {}'.format(img_data_dir))
    print('Using training output located at {}'.format(csv_data_dir))
    print('Using logs_path located at {}'.format(logs_path))
    print('Using save_dir located at {}'.format(save_dir))
    print('--------------------------------------------------------------')


    # Size of input images
    # input Layer is of size 3@66x200
    img_height = 66
    img_width = 200
    img_channels = 3
    batch_size = 100
    training_steps = 2000
    # -----------------------------------------------------------------------------
    # Load images and angles onto memory
    # -----------------------------------------------------------------------------
    xx = []
    yy = []

    print("Loading steering angles ...")
    yy = pd.read_csv(csv_data_dir + "/" + "steering.csv", header = None)
    yy = np.array(yy)
    print("Number of loaded output: {}".format(len(yy)))

    listing = [f for f in os.listdir(img_data_dir) if not f.startswith('.')]
    listing = sorted(listing, key=lambda x: (int(re.sub('\D','',x)),x))

    print("Loading road images ...")
    for file in listing:
        img = load_img(img_data_dir + "/" + file, target_size=(img_height, img_width))
        img = img_to_array(img)
        img = np.array(img)
        xx.append(img / 255.) # Normalizing image values
    print("Number of loaded input: {}".format(len(xx)))
    # -----------------------------------------------------------------------------
    # Divide data into training, validation, and test datasets
    # training: 60%
    # validation: 20%
    # test: 20%
    # -----------------------------------------------------------------------------
    imgs = OrderedDict()
    wheels = OrderedDict()

    imgs['train'] = []
    imgs['val']    = []
    imgs['test']  = []
    wheels['train'] = []
    wheels['val']    = []
    wheels['test']  = []

    print("Dividing data into train, val, and test data set ...")

    for i in range(int(len(xx)*0.6)):
        imgs['train'].append(xx[i])
        wheels['train'].append(yy[i])

    for i in range(int(len(xx)*0.6), int(len(xx)*0.8)):
        imgs['val'].append(xx[i])
        wheels['val'].append(yy[i])

    for i in range(int(len(xx)*0.8), len(xx)):
        imgs['test'].append(xx[i])
        wheels['test'].append(yy[i])

    xtrain = np.asarray(imgs['train'])
    ytrain = np.asarray(wheels['train'])
    xval = np.asarray(imgs['val'])
    yval = np.asarray(wheels['val'])
    xtest = np.asarray(imgs['test'])
    ytest = np.asarray(wheels['test'])

    print("Size of training data set: {}".format(len(xtrain)))
    print("Size of validation data set: {}".format(len(xval)))
    print("Size of test data set: {}".format(len(xtest)))
    # -----------------------------------------------------------------------------
    # Create the network and callbacks
    # -----------------------------------------------------------------------------
    my_model = model.create_model()
    # checkpoint
    filepath = save_dir + "/" + "{epoch:02d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir = save_dir, histogram_freq = 0, batch_size = batch_size, \
                              write_graph = True, write_grads=False, write_images=False, \
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    history = History()
    callbacks_list = [checkpoint, tensorboard, history]

    # -----------------------------------------------------------------------------
    # Train the network
    # -----------------------------------------------------------------------------
    my_model.fit(xtrain, ytrain, batch_size=batch_size, epochs=training_steps,\
                 verbose=0, callbacks=callbacks_list, validation_data=(xval, yval))

    my_model.save('model.h5')
    # Save model.h5 on to google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)