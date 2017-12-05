import load_data
import model
from keras.callbacks import ModelCheckpoint, TensorBoard, History
import numpy as np
from time import time
import matplotlib.pyplot as plt




# Size of input images
# input Layer is of size 3@66x200
img_height = 66
img_width = 200
img_channels = 3
batch_size = 100
num_epochs = 2000

save_dir = './logs'

def trainGenerator(batch_size):
    while 1:
        xtrain, ytrain = load_data.loadTrainBatch(batch_size)
        xtrain = np.asarray(xtrain)
        ytrain = np.asarray(ytrain)
        yield xtrain, ytrain

def validationGenerator(batch_size):
    while 1:
        xval, yval = load_data.loadValBatch(batch_size)
        xval = np.asarray(xval)
        yval = np.asarray(yval)
        yield xval, yval

# -----------------------------------------------------------------------------
# Create the network and callbacks
# -----------------------------------------------------------------------------
my_model = model.create_model()

# checkpoint
filepath = save_dir + "/" + "{epoch:02d}-{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir = save_dir + "/{}".format(time()), histogram_freq = 0, batch_size = batch_size, \
                          write_graph = True, write_grads=False, write_images=False, \
                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

history = History()
callbacks_list = [history, tensorboard]
# -----------------------------------------------------------------------------
# Train the network
# -----------------------------------------------------------------------------

iters_train = load_data.num_train_images
iters_train = int(iters_train / batch_size)

iters_test = load_data.num_val_images
iters_test = int(iters_test / batch_size)

my_model.fit_generator(trainGenerator(batch_size), steps_per_epoch=iters_train, epochs=num_epochs,\
                       verbose=1, callbacks=callbacks_list, validation_data=validationGenerator(batch_size), \
                       validation_steps=iters_train)


my_model.save('model.h5')

# summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


