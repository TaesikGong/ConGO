import tensorflow as tf
import threading, cv2
import numpy as np
import util.rnn_ops_conv as rnn
import data.moving_mnist as mm_data
import os
import sys
from datetime import datetime
import DataToVideo
import DataToImg
def vid_show_thread(output_vid):
    for i in xrange(output_vid.shape[0]):
        cv2.imshow('vid', output_vid[i])
        cv2.waitKey(100)

class pred_model:
    def __init__(self, batch_size=1):
        with tf.device('/cpu:0'):
        ####with tf.device('/cpu:0'):
            self.input_frames = tf.placeholder(tf.float32, shape=[None, None, 64, 64, 1], name='input_frames')
            self.fut_frames = tf.placeholder(tf.float32, shape=[None, None, 64, 64, 1], name='future_frames')
            self.keep_prob = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='keep_prob')
            self.weight_decay = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='weight_decay')
            self.learning_rate = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='learning_rate')
            self.test_case = tf.placeholder(tf.bool, name='test_case')

            # data refinement
            s = tf.shape(self.input_frames)

            input_flatten = tf.reshape(self.input_frames, [s[0], s[1], 64, 64, 1])
            fut_flatten = tf.reshape(self.fut_frames, [s[0], s[1], 64, 64, 1])

            input_norm = input_flatten / 1.
            fut_norm = fut_flatten / 1.

            # cell declaration
            print('cell declaration...')

            dim1 = 16
            dim2 = 64
            cell_dim = 256
            bias_start = 0.0
            enc_cell = self.__lstm_cell(cell_dim, 2)  # expressive power: 2048
            fut_cell = self.__lstm_cell(cell_dim, 2)
            recon_cell = self.__lstm_cell(cell_dim, 2)


            def conv_to_input(input, name):

                with tf.variable_scope(name):
                    # tensor = tf.identity(input, name)

                    cv1_f = tf.get_variable("weights_cv1_f", shape=[3, 3, 1, dim1],
                                              initializer=tf.random_uniform_initializer(-0.01, 0.01))
                    cv1_b = tf.get_variable("weights_cv1_b", shape=[dim1],
                                            initializer=tf.constant_initializer(bias_start))
                    cv1 = tf.nn.relu(tf.nn.conv2d(input, cv1_f, strides=[1, 2, 2, 1], padding='VALID') + cv1_b)

                    cv2_f = tf.get_variable("weights_cv2_f", shape=[3, 3, dim1, dim2],
                                            initializer=tf.random_uniform_initializer(-0.01, 0.01))
                    cv2_b = tf.get_variable("weights_cv2_b", shape=[dim2],
                                            initializer=tf.constant_initializer(bias_start))
                    cv2 = tf.nn.relu(tf.nn.conv2d(cv1, cv2_f, strides=[1, 2, 2, 1], padding='VALID') + cv2_b)

                    cv3_f = tf.get_variable("weights_cv3_f", shape=[3, 3, dim2, cell_dim],
                                            initializer=tf.random_uniform_initializer(-0.01, 0.01))
                    cv3_b = tf.get_variable("weights_cv3_b", shape=[cell_dim],
                                            initializer=tf.constant_initializer(bias_start))
                    cv3 = tf.nn.relu(tf.nn.conv2d(cv2, cv3_f, strides=[1, 2, 2, 1], padding='VALID') + cv3_b)

                    return cv3

            def conv_to_output(input, name):

                with tf.variable_scope(name):

                    # input = ?,7,7,2048
                    shape1 = [batch_size, 15, 15, dim2]
                    dcv1_f = tf.get_variable("weights_dcv1_f", shape=[3, 3, dim2, cell_dim],
                                             initializer=tf.random_uniform_initializer(-0.01, 0.01))
                    dcv1_b = tf.get_variable("weights_dcv1_b", shape=[dim2],
                                             initializer=tf.constant_initializer(bias_start))
                    dcv1 = tf.nn.relu(tf.nn.conv2d_transpose(input, dcv1_f, output_shape=shape1,
                                                             strides=[1, 2, 2, 1], padding='VALID') + dcv1_b)

                    shape2 = [batch_size, 31, 31, dim1]
                    dcv2_f = tf.get_variable("weights_dcv2_f", shape=[3, 3, dim1, dim2],
                                             initializer=tf.random_uniform_initializer(-0.01, 0.01))
                    dcv2_b = tf.get_variable("weights_dcv2_b", shape=[dim1],
                                             initializer=tf.constant_initializer(bias_start))
                    dcv2 = tf.nn.relu(tf.nn.conv2d_transpose(dcv1, dcv2_f, output_shape=shape2,
                                                             strides=[1, 2, 2, 1], padding='VALID') + dcv2_b)

                    shape3 = [batch_size, 64, 64, 1]
                    dcv3_f = tf.get_variable("weights_dcv3_f", shape=[3, 3, 1, dim1],
                                             initializer=tf.random_uniform_initializer(-0.01, 0.01))
                    dcv3_b = tf.get_variable("weights_dcv3_b", shape=[1],
                                             initializer=tf.constant_initializer(bias_start))
                    dcv3 = tf.nn.conv2d_transpose(dcv2, dcv3_f, output_shape=shape3, strides=[1, 2, 2, 1], padding='VALID') + dcv3_b

                    return dcv3

            # encode frames
            print('encode frames...')
            enc_o, enc_s = rnn.custom_dynamic_rnn(enc_cell, input_norm, input_operation=conv_to_input,
                                                  name='enc_rnn', scope='enc_cell')


            #TODO: multi cell
            #copy c_states
            repr = enc_cell.zero_state(s[0], tf.float32)
            repr = (
                tf.contrib.rnn.LSTMStateTuple(enc_s[0][0], repr[0][1]),#[cell][c/h]
                tf.contrib.rnn.LSTMStateTuple(enc_s[1][0], repr[1][1]))


            #TODO:shift right
            dummy = tf.expand_dims(tf.zeros_like(input_norm[:, 0]), axis=1)#bx1xhxwxd
            input_norm_reverse = input_norm
            input_norm_reverse = tf.reverse(input_norm_reverse, [1])#2 or 1?
            input_norm_shifted = tf.concat([dummy, input_norm_reverse], 1)
            input_norm_shifted = input_norm_shifted[:, :-1]

            #input_norm_reverse = tf.reshape(input_norm_reverse, tf.shape(input_norm))

            recon_out, recon_st = rnn.custom_dynamic_rnn(recon_cell, input_norm_shifted, input_operation=conv_to_input,
                                                           output_operation=conv_to_output, output_conditioned=False,
                                                           output_dim=None, output_activation=tf.identity,
                                                           initial_state=repr, name='dec_rnn_recon', scope='dec_cell_recon')

            # future ground-truth (0 or 1)
            recon_logit = tf.greater(input_norm_reverse, 0.)

            # loss calculation
            self.recon_loss = \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=recon_out,
                                                        labels=tf.cast(recon_logit, tf.float32))
            self.recon_loss = tf.reduce_mean(tf.reduce_sum(self.recon_loss, [2, 3, 4]))  # ?,?,4096 -> ?,?,64,64,1
######

            # future prediction
            # TODO:shift right

            print('future prediction...')

            fut_norm_shifted = tf.concat([dummy, fut_norm], 1)
            fut_norm_shifted = fut_norm_shifted[:, :-1]
            fut_out_tr, fut_st_tr = rnn.custom_dynamic_rnn(fut_cell, fut_norm_shifted, input_operation=conv_to_input,
                                                     output_operation=conv_to_output, output_conditioned=False,
                                                     output_dim=None, output_activation=tf.identity,
                                                     initial_state=repr, name='dec_rnn_fut', scope='dec_cell_fut', reuse=False)


            fut_dummy_te = tf.zeros_like(input_norm)
            fut_out_te, fut_st_te = rnn.custom_dynamic_rnn(fut_cell, fut_dummy_te, input_operation=conv_to_input,
                                                     output_operation=conv_to_output, output_conditioned=True,
                                                     output_dim=None, output_activation=tf.identity,
                                                     recurrent_activation=tf.sigmoid,
                                                     initial_state=repr, name='dec_rnn_fut', scope='dec_cell_fut', reuse=True)


            fut_o, fut_s = tf.cond(self.test_case, lambda: (tf.convert_to_tensor(fut_out_te), tf.convert_to_tensor(fut_st_te)),
                                   lambda: (tf.convert_to_tensor(fut_out_tr), tf.convert_to_tensor(fut_st_tr)), name=None)

            # future ground-truth (0 or 1)
            fut_logit = tf.greater(fut_norm, 0.)

            self.fut_loss = \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fut_o,
                                                        labels=tf.cast(fut_logit, tf.float32))
            self.fut_loss = tf.reduce_mean(tf.reduce_sum(self.fut_loss, [2, 3, 4]))  # ?,?,4096 -> ?,?,64,64,1

            # optimizer
            print('optimization...')
            self.optimizer = self.__adam_optimizer_op(
                (self.fut_loss + self.recon_loss)) #+ self.weight_decay * self.__calc_weight_l2_panalty())

            # output future frames as uint8
            print('output future frames...')
            self.fut_output = tf.cast(tf.clip_by_value(tf.sigmoid(fut_o) * 255, 0, 255), tf.uint8)

    def __lstm_cell(self, cell_dim, num_multi_cells):

        cells = [rnn.ConvLSTMCell([None, 7, 7, cell_dim], [5, 5], use_peepholes=True, forget_bias=0.,
                                                       initializer=tf.random_uniform_initializer(-0.01, 0.01))
                 for _ in xrange(num_multi_cells)]

        multi_cell = rnn.MultiRNNCell(cells)
        return multi_cell

        return cell

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
        for i in xrange(len(trainable_vnames)):
            if trainable_vnames[i].split('/')[-1].startswith('weights'):
                l2_loss += tf.nn.l2_loss(trainable_vars[i])

        return l2_loss



if __name__ == '__main__':
    opts = mm_data.BouncingMNISTDataHandler.options()
    opts.batch_size = 1  # 80
    opts.image_size = 64
    opts.num_digits = 2
    opts.num_frames = 20##first half is for input, latter is ground-truth
    opts.step_length = 0.1
    min_loss = np.inf
    moving_mnist = mm_data.BouncingMNISTDataHandler(opts)
    batch_generator = moving_mnist.GetBatchThread()

    net = pred_model(batch_size=opts.batch_size)

    ####
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True


    saver = tf.train.Saver(max_to_keep=3)
    dir_name = "weights_ccc"
    Vdir_name = "./Video_ccc/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name) # make directory if not exists
    mnist = np.load('./data/moving_mnist.npy')
    mnist = mnist.astype(np.float) / 255 # 0~ 255 -> 0 ~ 1	
    with tf.Session(config=sess_config) as sess:
    ####with tf.Session() as sess:
        init_step = 0
        if len(sys.argv) > 1 and sys.argv[1]:
            import re
            restored_step = re.search('step(\d+)', sys.argv[1])
            init_step = int(restored_step.group(1))#load previous steps
            saver.restore(sess, sys.argv[1])

        else:
            tf.global_variables_initializer().run()
        sumloss = 0
        numIter = 2
        for step in xrange(0, numIter):
            #x_batch = batch_generator.next()
            ###x_batch = next(batch_generator)
            x_batch = mnist[step].reshape(1,20,64,64)
            inp_vid, fut_vid = np.split(x_batch, 2, axis=1)

            inp_vid, fut_vid = np.expand_dims(inp_vid, -1), np.expand_dims(fut_vid, -1)

            fut_loss_cross= sess.run([net.fut_loss],
                                   feed_dict={net.input_frames: inp_vid,
                                              net.fut_frames: fut_vid,
                                              net.test_case: True})

            print ("[step %d] Train loss CE: %f" % (step, fut_loss_cross[0]))            
            o_vid= sess.run([net.fut_output],feed_dict={net.input_frames: inp_vid,net.fut_frames: fut_vid,net.test_case: True})
            saver.save(sess, dir_name + "/{}__step{}__loss{:f}".format(
				str(datetime.now()).replace(' ', '_'),step,fut_loss_cross[0]))
            min_loss = fut_loss_cross[0]
            sumloss+=min_loss
            o_vid = o_vid[0].reshape([opts.num_frames // 2, opts.image_size, opts.image_size])
            output_vid = np.concatenate((np.squeeze((x_batch * 255).astype(np.uint8))[0:opts.num_frames // 2], o_vid), axis=0)
            ResultData = []
            ResultData.append(output_vid)
            ResultData.append(np.squeeze((x_batch * 255).astype(np.uint8)).reshape((20,64,64)))
            ResultData.append(str(min_loss))
            DataToVideo.MakeVideo(ResultData,step,Vdir_name)
        print("Average Loss:" + str(sumloss / np.float(numIter)))