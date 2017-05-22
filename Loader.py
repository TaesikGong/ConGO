import DataToVideo
#import DataToImg
#import train_example_conv
import numpy as np
import tensorflow as tf
import threading, cv2
import util.rnn_ops_conv as rnn
import data.moving_mnist as mm_data
import os
import sys
from datetime import datetime
#import DataToVideo
#mnist = np.load('mnist_test_seq.npy')
# mnist is the name of the image
# Assign your data to mnist

#DataToVideo.MakeVideo(mnist)
#DataToImg.MakeImage(mnist)
opts = mm_data.BouncingMNISTDataHandler.options()
opts.batch_size = 40  # 80
opts.image_size = 64
opts.num_digits = 2
opts.num_frames = 20##first half is for input, latter is ground-truth
opts.step_length = 0.1
min_loss = np.inf
moving_mnist = mm_data.BouncingMNISTDataHandler(opts)
batch_generator = moving_mnist.GetBatchThread()
x_batch = batch_generator.next()            
inp_vid, fut_vid = np.split(x_batch, 2, axis=1)
inp_vid, fut_vid = np.expand_dims(inp_vid, -1), np.expand_dims(fut_vid, -1)
fut_loss = 12345
ResultData = []
ResultData.append(np.squeeze((x_batch * 255).astype(np.uint8))[0].reshape((20,64,64)))
ResultData.append(np.squeeze((x_batch * 255).astype(np.uint8))[0].reshape((20,64,64)))
ResultData.append(fut_loss)
DataToVideo.MakeVideo(ResultData)