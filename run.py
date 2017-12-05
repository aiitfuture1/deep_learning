import tensorflow as tf
import scipy.misc
import model
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd


from keras import backend as K
K.set_image_dim_ordering('tf')

def atan(x):
    return tf.multiply(tf.atan(x), 2)

# Overlay steering wheel image on top of video frame
def image_overlay(s_img, l_img, x_offset , y_offset):
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    alpha_s = 1.0 
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])
    return l_img

# Open steering wheel image and resize it
img = cv2.imread('steering_wheel.png')
img = cv2.resize(img, (int(img.shape[0]/4), int(img.shape[1]/4)), interpolation = cv2.INTER_AREA)

rows,cols,dim = img.shape
# Initial smoothed angle
smoothed_angle = 0

# Open a trained model
model = load_model('model/model.h5',custom_objects={'atan': atan})


# Capture video frames
cap = cv2.VideoCapture('test_video/10_front.mp4')
# Read actual angles from csv file
y = pd.read_csv(r"test_video/10_steering.csv", usecols=[2], dtype=float).values[1:] /1.5


# Variables for videowriter
i = 0
j = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('test_video.avi',fourcc, 20.0, (1280,720))
while(i < 2699):
    ret, frame = cap.read()
    image = scipy.misc.imread("test_dataset/{}.png".format(j + 24300))/255.0
    j += 1
    image = np.expand_dims(image, axis=0)
    degrees = model.predict(x=image, batch_size=1, verbose=1) * 180 / scipy.pi

    print("Predicted steering angle: " + str(degrees) + " degrees")
    print("Human steering angle: " + str(y[i]) + " degrees")

    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))

    frame = image_overlay(dst, frame, 50, 70)
    cv2.putText(frame, str("Predicted angle: {:2.2f}".format(degrees[0][0])), (15, 30),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str("Human angle: {:2.2f}".format(y[i][0])), (15, 55),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    i += 1 % len(y)
    video.write(frame)

cap.release()
cv2.destroyAllWindows()
video.release()



