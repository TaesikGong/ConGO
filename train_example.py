import tensorflow as tf
import threading, cv2
import numpy as np
import util.rnn_ops as rnn
import data.moving_mnist as mm_data
import os
import sys
from datetime import datetime

def vid_show_thread(output_vid):
    for i in range(output_vid.shape[0]):
        cv2.imshow('vid', output_vid[i])
        cv2.waitKey(100)


class pred_model:
    def __init__(self):
        with tf.device('/GPU:0'):
            self.input_frames = tf.placeholder(tf.float32, shape=[None, None, 64, 64, 1],
                                               name='input_frames')  # batch,sqe,x,y,features
            self.fut_frames = tf.placeholder(tf.float32, shape=[None, None, 64, 64, 1], name='future_frames')
            self.keep_prob = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='keep_prob')
            self.weight_decay = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='weight_decay')
            self.learning_rate = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='learning_rate')

            # data refinement
            s = tf.shape(self.input_frames)

            input_flatten = tf.reshape(self.input_frames, [s[0], s[1], 64 * 64 * 1])
            fut_flatten = tf.reshape(self.fut_frames, [s[0], s[1], 64 * 64 * 1])

            input_norm = input_flatten / 1.
            fut_norm = fut_flatten / 1.

            # cell decleration
            enc_cell = self.__lstm_cell(2048, 2)
            fut_cell = self.__lstm_cell(2048, 2)

            # encode frames
            enc_o, enc_s = rnn.custom_dynamic_rnn(enc_cell, input_norm, name='enc_rnn', scope='enc_cell')

            # state mapping
            repr = enc_cell.zero_state(s[0], tf.float32)

            repr = (
                tf.contrib.rnn.LSTMStateTuple(enc_s[0][0], repr[0][1]),
                tf.contrib.rnn.LSTMStateTuple(enc_s[1][0], repr[1][1]))

            # future prediction
            fut_dummy = tf.zeros_like(fut_norm)
            fut_o, fut_s = rnn.custom_dynamic_rnn(fut_cell, fut_dummy, output_conditioned=False, output_dim=4096,
                                                  output_activation=tf.identity,
                                                  initial_state=repr, name='dec_rnn', scope='dec_cell')

            # future ground-truth (0 or 1)
            fut_logit = tf.greater(fut_norm, 0.)

            # loss calculation
            self.fut_loss = \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fut_o,
                                                        labels=tf.cast(fut_logit, tf.float32))
            self.fut_loss = tf.reduce_mean(tf.reduce_sum(self.fut_loss, [2]))

            # optimizer
            self.optimizer = self.__adam_optimizer_op(
                self.fut_loss + self.weight_decay * self.__calc_weight_l2_panalty())

            # output future frames as uint8
            self.fut_output = tf.cast(tf.clip_by_value(tf.sigmoid(fut_o) * 255, 0, 255), tf.uint8)

    def __lstm_cell(self, cell_dim, num_multi_cells):
        cells = [tf.contrib.rnn.core_rnn_cell.LSTMCell(cell_dim, use_peepholes=True, forget_bias=0.,
                                                       initializer=tf.random_uniform_initializer(-0.01, 0.01))
                 for _ in range(num_multi_cells)]

        drop_cells = [tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell,
                                                                  input_keep_prob=self.keep_prob,
                                                                  output_keep_prob=self.keep_prob)

                      for cell in cells]

        multi_cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(drop_cells)
        return multi_cell

    def __momentum_optimizer_op(self, cost):
        o = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        op = o.minimize(cost)
        return op

    def __adam_optimizer_op(self, cost):
        o = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        op = o.minimize(cost)
        return op

    def __calc_weight_l2_panalty(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainable_vnames = [v.name for v in trainable_vars]

        l2_loss = 0.
        for i in range(len(trainable_vnames)):
            if trainable_vnames[i].split('/')[-1].startswith('weights'):
                l2_loss += tf.nn.l2_loss(trainable_vars[i])

        return l2_loss


if __name__ == '__main__':
    opts = mm_data.BouncingMNISTDataHandler.options()
    opts.batch_size = 80
    opts.image_size = 64
    opts.num_digits = 2
    opts.num_frames = 20
    opts.step_length = 0.1
    min_loss = np.inf
    moving_mnist = mm_data.BouncingMNISTDataHandler(opts)
    batch_generator = moving_mnist.GetBatchThread()

    net = pred_model()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    saver = tf.train.Saver(max_to_keep=2)
    dir_name = "weights"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) # make directory if not exists

    with tf.Session(config=sess_config) as sess:
        if len(sys.argv) > 1 and sys.argv[1]:
            saver.restore(sess, sys.argv[1])
        else:
            tf.global_variables_initializer().run()

        for step in range(100000):
            x_batch = batch_generator.next()
            inp_vid, fut_vid = np.split(x_batch, 2, axis=1)

            inp_vid, fut_vid = np.expand_dims(inp_vid, -1), np.expand_dims(fut_vid, -1)

            _, fut_loss = sess.run([net.optimizer, net.fut_loss],
                                   feed_dict={net.input_frames: inp_vid,
                                              net.fut_frames: fut_vid})

            print ("[step %d] loss: %f" % (step, fut_loss))
            if fut_loss < min_loss - 5: # THRESHOLD
                saver.save(sess, dir_name+"/{}__step{}__loss{:f}".format(
                    str(datetime.now()).replace(' ','_'),
                    step,
                    fut_loss
                ))
                min_loss = fut_loss

            if step % 40 == 0:
                o_vid = sess.run(net.fut_output, feed_dict={net.input_frames: inp_vid,
                                                            net.fut_frames: fut_vid})

                o_vid = o_vid[0].reshape([opts.num_frames // 2, opts.image_size, opts.image_size])
                output_vid = np.concatenate(
                    (np.squeeze((x_batch * 255).astype(np.uint8))[0][0:opts.num_frames // 2], o_vid), axis=0)
                threading.Thread(target=vid_show_thread, args=([output_vid])).start()
