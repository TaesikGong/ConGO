import h5py, sys
import numpy as np
import threading

"""
This code is part of "https://github.com/emansim/unsupervised-videos"
"""
class BouncingMNISTDataHandler(object):
  """Data Handler that creates Bouncing MNIST dataset on the fly."""
  class options:
    def __init__(self):
      self.num_frames = None
      self.batch_size = None
      self.image_size = None
      self.num_digits = None
      self.step_length = None

  def __init__(self, options):
    self.seq_length_ = options.num_frames
    self.batch_size_ = options.batch_size
    self.image_size_ = options.image_size
    self.num_digits_ = options.num_digits
    self.step_length_ = options.step_length
    self.dataset_size_ = 10000  # The dataset is really infinite. This is just for validation.
    self.digit_size_ = 28
    self.frame_size_ = self.image_size_ ** 2

    try:
      f = h5py.File('data/mnist.h5')
    except:
      print 'Please set the correct path to MNIST dataset'
      sys.exit()

    self.data_ = f['train'].value.reshape(-1, 28, 28)
    f.close()
    self.indices_ = np.arange(self.data_.shape[0])
    self.row_ = 0
    np.random.shuffle(self.indices_)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    pass

  def GetRandomTrajectory(self, batch_size):
    length = self.seq_length_
    canvas_size = self.image_size_ - self.digit_size_

    # Initial position uniform random inside the box.
    y = np.random.rand(batch_size)
    x = np.random.rand(batch_size)

    # Choose a random velocity.
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros((length, batch_size))
    start_x = np.zeros((length, batch_size))
    for i in xrange(length):
      # Take a step along velocity.
      y += v_y * self.step_length_
      x += v_x * self.step_length_

      # Bounce off edges.
      for j in xrange(batch_size):
        if x[j] <= 0:
          x[j] = 0
          v_x[j] = -v_x[j]
        if x[j] >= 1.0:
          x[j] = 1.0
          v_x[j] = -v_x[j]
        if y[j] <= 0:
          y[j] = 0
          v_y[j] = -v_y[j]
        if y[j] >= 1.0:
          y[j] = 1.0
          v_y[j] = -v_y[j]
      start_y[i, :] = y
      start_x[i, :] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

  def Overlap(self, a, b):
    """ Put b on top of a."""
    return np.maximum(a, b)
    # return b

  def __GetBatchThreadWorker(self, verbose=False, outputs=None):
    outputs[0] = self.GetBatch(verbose=verbose)

  def GetBatchThread(self, verbose=False):
    if NameError:#???
      outputs = [[]]
      th = threading.Thread(target=self.__GetBatchThreadWorker, args=(verbose, outputs))
      th.start()
      while True:
        th.join()
        res = outputs[:]

        th = threading.Thread(target=self.__GetBatchThreadWorker, args=(verbose, outputs))
        th.start()
        yield res[0]

  def GetBatch(self, verbose=False):
    start_y, start_x = self.GetRandomTrajectory(self.batch_size_ * self.num_digits_)

    # minibatch data
    data = np.zeros((self.batch_size_, self.seq_length_, self.image_size_, self.image_size_), dtype=np.float32)

    for j in xrange(self.batch_size_):
      for n in xrange(self.num_digits_):

        # get random digit from dataset
        ind = self.indices_[self.row_]
        self.row_ += 1
        if self.row_ == self.data_.shape[0]:
          self.row_ = 0
          np.random.shuffle(self.indices_)
        digit_image = self.data_[ind, :, :]

        # generate video
        for i in xrange(self.seq_length_):
          top = start_y[i, j * self.num_digits_ + n]
          left = start_x[i, j * self.num_digits_ + n]
          bottom = top + self.digit_size_
          right = left + self.digit_size_
          data[j, i, top:bottom, left:right] = self.Overlap(data[j, i, top:bottom, left:right], digit_image)

    return data

if __name__ == '__main__':
  opts = BouncingMNISTDataHandler.options()
  opts.batch_size = 1
  opts.image_size = 64
  opts.num_digits = 2
  opts.num_frames = 32
  opts.step_length = 0.1

  while True:
    data_handler = BouncingMNISTDataHandler(opts)
    d = data_handler.GetBatch()
    d_r = d.reshape(-1, opts.num_frames, opts.image_size, opts.image_size)
    d_r *= 255
    d_r = d_r.astype(np.uint8)

    import cv2

    for i in range(opts.num_frames):
      cv2.imshow('digits', d_r[0,i])
      cv2.waitKey(100)
