import tensorflow as tf
import numpy as np

class Discriminator:
    def __init__(self, n_hidden, p_g, p_data):
        n_input = 10*64*64
        learning_rate = learning_rate = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='learning_rate')

        P_G = tf.reshape(p_g, [-1, 10*64*64])
        P_DATA = tf.reshape(p_data, [-1, 10*64*64])

        self.D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
        self.D_b1 = tf.Variable(tf.zeros([n_hidden]))

        self.D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
        self.D_b2 = tf.Variable(tf.zeros([1]))

        self.D_fake, self.D_logit_fake = self.discriminator(P_G)
        self.D_real, self.D_logit_real = self.discriminator(P_DATA)

        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logit_real, labels = tf.ones_like(self.D_logit_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logit_fake, labels = tf.zeros_like(self.D_logit_fake)))
        
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logit_fake, labels = tf.ones_like(self.D_logit_fake)))

        theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list = theta_D)

    def discriminator(self, inputs):
        hidden_layer = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(hidden_layer, self.D_W2) + self.D_b2
        discrimination = tf.sigmoid(D_logit)
        return discrimination, D_logit
