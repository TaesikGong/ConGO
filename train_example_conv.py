import tensorflow as tf
import threading, cv2
import numpy as np
import util.rnn_ops_conv as rnn
import data.moving_mnist as mm_data
import os
import sys
from datetime import datetime

def vid_show_thread(output_vid):
    for i in range(output_vid.shape[0]):
        cv2.imshow('vid', output_vid[i])
        cv2.waitKey(100)

#comment
class pred_model:
    def __init__(self, batch_size=80):
        with tf.device('/gpu:0'):
        ###with tf.device('/cpu:0'):
            self.input_frames = tf.placeholder(tf.float32, shape=[None, None, 64, 64, 1], name='input_frames')
            self.fut_frames = tf.placeholder(tf.float32, shape=[None, None, 64, 64, 1], name='future_frames')
            self.keep_prob = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='keep_prob')
            self.weight_decay = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='weight_decay')
            self.learning_rate = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='learning_rate')
            # self.learning_rate = tf.Variable(1e-2, dtype=tf.float32, trainable=False, name='learning_rate')
            self.test_case = tf.placeholder(tf.bool, name='test_case')

            # data refinement
            s = tf.shape(self.input_frames)

            #TODO:old
            # input_flatten = tf.reshape(self.input_frames, [s[0], s[1], 64 * 64 * 1])
            # fut_flatten = tf.reshape(self.fut_frames, [s[0], s[1], 64 * 64 * 1])

            #TODO:conv
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
            repr_cell = self.__lstm_cell(cell_dim, 2)


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

            #TODO: single cell
            # repr = enc_s
            ###TODO: reverse input_orm


            input_norm_reverse = input_norm
            print("input_norm_reverse", input_norm_reverse)
            input_norm_reverse = tf.reverse(input_norm_reverse, [2])#2 or 1?
            print("input_norm_reverse", input_norm_reverse)
            input_norm_reverse = tf.reshape(input_norm_reverse, tf.shape(input_norm))
            print("input_norm_reverse", input_norm_reverse)

            repr_out, repr_st = rnn.custom_dynamic_rnn(repr_cell, input_norm_reverse, input_operation=conv_to_input,
                                                           output_operation=conv_to_output, output_conditioned=False,
                                                           output_dim=None, output_activation=tf.identity,
                                                           initial_state=repr, name='dec_rnn', scope='dec_cell')

            # future ground-truth (0 or 1)
            repr_logit = tf.greater(input_norm_reverse, 0.)

            # loss calculation
            self.repr_loss = \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=repr_out,
                                                        labels=tf.cast(repr_logit, tf.float32))

            self.repr_loss = tf.reduce_mean(tf.reduce_sum(self.repr_loss, [2, 3, 4]))  # ?,?,4096 -> ?,?,64,64,1
######
            # future prediction

            print('future prediction...')

            #fut_dummy = tf.zeros_like(input_norm) # need to be changed "if we use conditioned"
            #fut_dummy[:,0] = tf.zeros_like(batch,64,64,1)
            #fut_dummy[:,1-9] = fut_frames[:, 1-9, :, :, :]
            #fut_dummy = tf.zeros_like(enc_o)
            #TODO: output_dim = None!

            # fut_o, fut_s = rnn.custom_dynamic_rnn(fut_cell, fut_dummy, input_operation=conv_to_input,
            #                                       output_operation=conv_to_output, output_conditioned=False,
            #                                       output_dim=None, output_activation=tf.identity,
            #                                       initial_state=repr, name='dec_rnn', scope='dec_cell')


            fut_out_tr, fut_st_tr = rnn.custom_dynamic_rnn(fut_cell, input_norm, input_operation=conv_to_input,
                                                     output_operation=conv_to_output, output_conditioned=False,
                                                     output_dim=None, output_activation=tf.identity,
                                                     initial_state=repr, name='dec_rnn', scope='dec_cell', reuse=True)

            fut_dummy_te = tf.zeros_like(input_norm)
            fut_out_te, fut_st_te = rnn.custom_dynamic_rnn(fut_cell, fut_dummy_te, input_operation=conv_to_input,
                                                     output_operation=conv_to_output, output_conditioned=True,
                                                     output_dim=None, output_activation=tf.identity,
                                                     recurrent_activation=tf.sigmoid,
                                                     initial_state=repr, name='dec_rnn', scope='dec_cell', reuse=True)

            def l1_loss(tensor, weight=1.0, scope=None):
                """Define a L1Loss, useful for regularize, i.e. lasso.

                Args:
                  tensor: tensor to regularize.
                  weight: scale the loss by this factor.
                  scope: Optional scope for op_scope.

                Returns:
                """
                with tf.name_scope(scope, 'L1Loss', [tensor]):
                #with tf.op_scope([tensor], scope, 'L1Loss'): #WARNING:tensorflow:tf.op_scope(values, name, default_name) is deprecated, use tf.name_scope(name, default_name, values)
                    weight = tf.convert_to_tensor(weight,
                                                  dtype=tensor.dtype.base_dtype,
                                                  name='loss_weight')
                    loss = tf.multiply(weight, tf.reduce_sum(tf.abs(tensor)), name = 'value')
                    #tf.add_to_collection(LOSSES_COLLECTION, loss)
                    return loss

            #print("tr: ", fut_out_, )
            fut_o, fut_s = tf.cond(self.test_case, lambda: (tf.convert_to_tensor(fut_out_te), tf.convert_to_tensor(fut_st_te)), lambda: (tf.convert_to_tensor(fut_out_tr), tf.convert_to_tensor(fut_st_tr)), name=None)
            # print(fut_o, fut_s)

            # future ground-truth (0 or 1)
            fut_logit = tf.greater(fut_norm, 0.)

            #fut_o
            # loss calculation
            self.fut_loss_old = \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fut_o,
                                                        labels=tf.cast(fut_logit, tf.float32))

            self.fut_loss = \
                l1_loss(tf.subtract(fut_o, fut_norm))

            ## fut_o: ?,?,4096
            ## fut_logit: ?,?,4096

            self.fut_loss_old = tf.reduce_mean(tf.reduce_sum(self.fut_loss_old, [2, 3, 4]))#?,?,4096 -> ?,?,64,64,1


            # optimizer
            print('optimization...')
            self.optimizer = self.__adam_optimizer_op(
                #####??self.fut_loss + self.weight_decay * self.__calc_weight_l2_panalty())
                self.fut_loss + self.repr_loss + self.weight_decay * self.__calc_weight_l2_panalty())

            # output future frames as uint8
            print('output future frames...')
            self.fut_output = tf.cast(tf.clip_by_value(tf.sigmoid(fut_o) * 255, 0, 255), tf.uint8)

    def __lstm_cell(self, cell_dim, num_multi_cells):

        #TODO:old
        # cells = [tf.contrib.rnn.core_rnn_cell.LSTMCell(cell_dim, use_peepholes=True, forget_bias=0.,
        #                                               initializer=tf.random_uniform_initializer(-0.01, 0.01))
        #          for _ in range(num_multi_cells)]
        #
        # drop_cells = [tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell,
        #                                                           input_keep_prob=self.keep_prob,
        #                                                           output_keep_prob=self.keep_prob) for cell in cells]
        # multi_cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(drop_cells)
        # return multi_cell

        #TODO:multi cell
        cells = [rnn.ConvLSTMCell([None, 7, 7, cell_dim], [5, 5], use_peepholes=True, forget_bias=0.,
                                                       initializer=tf.random_uniform_initializer(-0.01, 0.01))
                 for _ in range(num_multi_cells)]

        multi_cell = rnn.MultiRNNCell(cells)
        return multi_cell

        # TODO:single cell
        # cell = rnn.ConvLSTMCell([None, 7, 7, cell_dim], [3, 3], use_peepholes=True, forget_bias=0.,
        #                           initializer=tf.random_uniform_initializer(-0.01, 0.01))
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
        for i in range(len(trainable_vnames)):
            if trainable_vnames[i].split('/')[-1].startswith('weights'):
                l2_loss += tf.nn.l2_loss(trainable_vars[i])

        return l2_loss


if __name__ == '__main__':
    opts = mm_data.BouncingMNISTDataHandler.options()
    opts.batch_size = 40  # 80
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


    saver = tf.train.Saver(max_to_keep=2)
    dir_name = "weights_ccc_lossl1"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name) # make directory if not exists

    with tf.Session(config=sess_config) as sess:
    ###with tf.Session() as sess:
        init_step = 0
        if len(sys.argv) > 1 and sys.argv[1]:
            import re
            restored_step = re.search('step(\d+)', sys.argv[1])
            init_step = int(restored_step.group(1))#load previous steps
            saver.restore(sess, sys.argv[1])

        else:
            tf.global_variables_initializer().run()

        for step in range(init_step, 500000):
            x_batch = batch_generator.next()
            ###x_batch = next(batch_generator)
            inp_vid, fut_vid = np.split(x_batch, 2, axis=1)

            inp_vid, fut_vid = np.expand_dims(inp_vid, -1), np.expand_dims(fut_vid, -1)

            _, fut_loss_tr = sess.run([net.optimizer, net.fut_loss],
                                   feed_dict={net.input_frames: inp_vid,
                                              net.fut_frames: fut_vid,
                                              net.test_case: False})

            print ("[step %d] Train loss: %f" % (step, fut_loss_tr))
            # if fut_loss_tr < min_loss - 5: # THRESHOLD
            #     saver.save(sess, dir_name+"/{}__step{}__loss{:f}".format(
            #         str(datetime.now()).replace(' ','_'),
            #         step,
            #         fut_loss_tr
            #     ))
            #     min_loss = fut_loss_tr

            if step % 40 == 0:
                o_vid, fut_loss_te = sess.run([net.fut_output, net.fut_loss], feed_dict={net.input_frames: inp_vid,
                                                            net.fut_frames: fut_vid,
                                                            net.test_case: True})
                print("type of fut_loss_te:", type(fut_loss_te))
                print ("[step %d] Test loss: %f" % (step, fut_loss_te))

                if fut_loss_te < min_loss - 5:  # THRESHOLD
                    saver.save(sess, dir_name + "/{}__step{}__loss{:f}".format(
                        str(datetime.now()).replace(' ', '_'),
                        step,
                        fut_loss_te
                    ))
                    min_loss = fut_loss_te

                o_vid = o_vid[0].reshape([opts.num_frames // 2, opts.image_size, opts.image_size])
                output_vid = np.concatenate(
                    (np.squeeze((x_batch * 255).astype(np.uint8))[0][0:opts.num_frames // 2], o_vid), axis=0)
                threading.Thread(target=vid_show_thread, args=([output_vid])).start()
