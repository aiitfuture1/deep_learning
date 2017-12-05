import scipy.misc
import random
import cv2

# xs : Array that holds the address to all frames
# ys : Array that holds steering angle values
xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

# read steering_angles.txt
# first value in each line is name of a frame and
# second value in each line is turning radius
# e.g.
# 0.png -1.0 
with open("driving_dataset/steering_angles.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        ys.append(float(line.split()[1]) * scipy.pi / 180) # steering angle in radians

# Find number of images for trainig and validation sets:
# trainig    = 80% of dataset
# validation = 20% of dataset
num_imgs       = len(xs)
num_train_imgs = int(num_imgs * 0.8)
num_val_imgs   = int(num_imgs * 0.2)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[0 : num_train_imgs]
train_ys = ys[0 : num_train_imgs]

val_xs = xs[-num_val_imgs : ]
val_ys = ys[-num_val_imgs : ]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def loadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # Normalize each image -> /255.0
        img = scipy.misc.imread(train_xs[(train_batch_pointer + i)% num_train_images])
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)/255.0
        x_out.append(img_yuv)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_imgs]])
    train_batch_pointer += batch_size
    return x_out, y_out

def loadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imread(train_xs[(val_batch_pointer + i) % num_train_images])/255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
