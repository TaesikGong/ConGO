import tensorflow as tf
import numpy as np

class Discriminator:
    def __init__(self, n_hidden, p_g, p_data):
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)# Variables declared before descriminator

        n_input = 10*64*64
        learning_rate = learning_rate = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='learning_rate')

        '''
        TODO: 
        4(?). delete variables from encoder?
        '''

        P_G = tf.reshape(p_g, [-1, 64, 64, 1])
        P_DATA = tf.reshape(p_data, [-1, 64, 64, 1])
        p_g_conv = self.conv_to_input(P_G, 'p_g')
        p_data_conv = self.conv_to_input(P_DATA, 'p_data')
        p_g_conv_flat = tf.reshape(p_g_conv, [-1, 7*7*256])
        p_data_conv_flat = tf.reshape(p_data_conv, [-1, 7*7*256])

        self.D_W1 = tf.Variable(tf.random_normal([7*7*256, n_hidden], stddev=0.01))
        self.D_b1 = tf.Variable(tf.zeros([n_hidden]))

        self.D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
        self.D_b2 = tf.Variable(tf.zeros([1]))

        self.D_fake, self.D_logit_fake = self.discriminator(p_g_conv_flat)
        self.D_real, self.D_logit_real = self.discriminator(p_data_conv_flat)

        self.D_loss_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logit_real, labels=tf.ones_like(self.D_logit_real)), [2, 3, 4]))
        self.D_loss_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logit_fake, labels=tf.zeros_like(self.D_logit_fake)), [2, 3, 4]))
        self.G_loss_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logit_fake, labels=tf.ones_like(self.D_logit_fake)), [2, 3, 4]))
        self.G_loss_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logit_real, labels=tf.zeros_like(self.D_logit_real)), [2, 3, 4]))

        self.D_loss = (self.D_loss_real + self.D_loss_fake)
        self.G_loss = 0.01 * (self.G_loss_real + self.G_loss_fake)



        theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2,
                   self.cv1_f, self.cv1_b, 
                   self.cv2_f, self.cv2_b, 
                   self.cv3_f, self.cv3_b,
                   ]

        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=g_vars)
        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=theta_D)


    def discriminator(self, inputs):
        hidden_layer = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(hidden_layer, self.D_W2) + self.D_b2
        discrimination = tf.sigmoid(D_logit)
        return discrimination, D_logit

    def conv_to_input(self, input_, name):
        dim1 = 16
        dim2 = 64
        cell_dim = 256
        bias_start = 0.0
        with tf.variable_scope(name):
            self.cv1_f = tf.get_variable("weights_cv1_f", shape=[3, 3, 1, dim1],
                                    initializer=tf.random_uniform_initializer(-0.01, 0.01))
            self.cv1_b = tf.get_variable("weights_cv1_b", shape=[dim1],
                                    initializer=tf.constant_initializer(bias_start))
            self.cv1 = tf.nn.relu(tf.nn.conv2d(input_, self.cv1_f, strides=[1, 2, 2, 1], padding='VALID') + self.cv1_b)

            self.cv2_f = tf.get_variable("weights_cv2_f", shape=[3, 3, dim1, dim2],
                                    initializer=tf.random_uniform_initializer(-0.01, 0.01))
            self.cv2_b = tf.get_variable("weights_cv2_b", shape=[dim2],
                                    initializer=tf.constant_initializer(bias_start))
            self.cv2 = tf.nn.relu(tf.nn.conv2d(self.cv1, self.cv2_f, strides=[1, 2, 2, 1], padding='VALID') + self.cv2_b)

            self.cv3_f = tf.get_variable("weights_cv3_f", shape=[3, 3, dim2, cell_dim],
                                    initializer=tf.random_uniform_initializer(-0.01, 0.01))
            self.cv3_b = tf.get_variable("weights_cv3_b", shape=[cell_dim],
                                    initializer=tf.constant_initializer(bias_start))
            self.cv3 = tf.nn.relu(tf.nn.conv2d(self.cv2, self.cv3_f, strides=[1, 2, 2, 1], padding='VALID') + self.cv3_b)

            return self.cv3
