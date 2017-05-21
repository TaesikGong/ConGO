import DataToVideo
import DataToImg
import train_example_conv
import numpy as np
mnist = np.load('mnist_test_seq.npy')
# mnist is the name of the image
# Assign your data to mnist

DataToVideo.MakeVideo(mnist)
#DataToImg.MakeImage(mnist)

