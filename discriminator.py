import tensorflow as tf
import numpy as np

class Discriminator:
    def __init__(self, n_hidden, p_g, p_data):
        n_input = 10*64*64
        learning_rate = 0.0002

        P_G = tf.reshape(p_g, [-1, 10*64*64])
        P_DATA = tf.reshape(p_data, [-1, 10*64*64])
        
        self.D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
        self.D_b1 = tf.Variable(tf.zeros([n_hidden]))

        self.D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
        self.D_b2 = tf.Variable(tf.zeros([1]))

        self.D_gene = self.discriminator(P_G)
        self.D_real = self.discriminator(P_DATA)

        self.loss_D = tf.reduce_mean(tf.log(self.D_real) + tf.log(1 - self.D_gene))
        self.loss_G = tf.reduce_mean(tf.log(self.D_gene))

        D_var_list = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        self.train_D = tf.train.AdamOptimizer(learning_rate).minimize(-self.loss_D, var_list = D_var_list)

    def discriminator(self, inputs):
        hidden_layer = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        discrimination = tf.sigmoid(tf.matmul(hidden_layer, self.D_W2) + self.D_b2)
        return discrimination

