# Steering   Direction   of   Self-Driving   Cars Via   Deep   Learning

Aknowledgment:

1) https://github.com/lexfridman/deeptesla
2) https://github.com/SullyChen/Autopilot-TensorFlow
3) https://github.com/tmaila/autopilot

We started from implementing NVIDIA’s network architecture for end-to-end self-driving cars that was published in their paper [1]. To train this CNN we used a set of data provided by MIT [2], which includes over 1 hour of driving videos and corresponding steering angles with time stamps.

We randomly selected 80% of our data as our training set and used the remaining 20% for cross examination and testing and tried to improve the above network. Development and training was first done on one of our group members personal computer and later we used an EC2 instance on and Amazon Web Services(AWS). 

The code was done in Python using the machine learning libraries Tensorflow and Keras, and OpenCV was used for preprocessing the input data.

Finally, we visualized our result and compared it with human recorded output to measure the accuracy of our trained network. 


[1] M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. Jackel, M. Monfort, U. Muller, J. Zhang, X. Zhang, J. Zhao and K. Zieba, "End to End Learning for Self-Driving Cars", 2016.

[2] L. Fridman, “Deeptesla”, 2016, Github, Github repository, https://github.com/lexfridman/deeptesla/tree/master/epochs. [Accessed: 02- Nov- 2017]
