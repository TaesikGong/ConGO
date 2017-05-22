import tensorflow as tf

def basic_conv2d(in_tensor, dim, name, reuse=False, trainable=True, relu=True, filt=(3, 3), stride=(1, 1)):
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable("weights", shape=[filt[0], filt[1], in_tensor.get_shape().as_list()[3], dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=trainable)
        biases = tf.get_variable("biases", shape=[dim],
                                 initializer=tf.constant_initializer(0.0),
                                 trainable=trainable)
        conv = tf.nn.conv2d(in_tensor, weights,
                            strides=[1, stride[0], stride[1], 1], padding='SAME')
        if relu:
            return tf.nn.relu(conv + biases)
        else:
            return conv + biases


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.random_normal_initializer(stddev=stddev))

    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                              strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

    if with_w:
      return deconv, w, biases
    else:
      return deconv