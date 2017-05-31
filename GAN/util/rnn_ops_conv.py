"""
"""
import tensorflow as tf
from time import gmtime, strftime

def custom_dynamic_rnn(cell, inputs, output_dim=None, output_conditioned=False,
                       sequence_length=None, initial_state=None,
                       output_activation=tf.nn.relu, recurrent_activation=tf.identity,
                       input_operation=tf.identity, output_operation=tf.identity,
                       parallel_iterations=None, swap_memory=False, name=None, scope=None, reuse=False):
    """
    :param cell: Tensorflow RNN cell
    :param inputs: 3D or 5D tensor - batch x sequence x some dimensions...
    :param output_dim: 1D tensor
    :param output_conditioned: boolean indicates input at time t is output at t-1
    :param sequence_length: 1D tensor sequence length
    :param initial_state: initial rnn state
    :param output_activation: output activation (default tf.identity) only used when output_dimension is not None
    :param recurrent_activation: recurrent activation (default tf.identity)
    :param input_operation: input operation (default tf.identity) applied to inputs before a RNN
    :param output_operation: output operation (default tf.identity) applied to output of rnn
    :param parallel_iterations: parameter for tf.while_loop
    :param swap_memory: parameter for tf.while_loop
    :param name: scope of entire rnn block
    :param scope: scope of RNN unit
    :param reuse: for re-using pre-built RNN
    :return: output, state
    """
    if parallel_iterations is None:
        parallel_iterations = 10

    with tf.variable_scope(name or 'custom_rnn', reuse=reuse):
        inputs_shape = tf.shape(inputs)

        if not output_conditioned:
            ground_conditioned = True

        if sequence_length is None:
            if (not output_conditioned) or ground_conditioned:
                sequence_length = inputs_shape[1]

        # parameter initialization
        time = tf.constant(0, dtype=tf.int32)

        if initial_state is not None:
            state = initial_state
        else:
            state = cell.zero_state(inputs_shape[0], tf.float32)

        outputs_ta = tf.TensorArray(dtype=tf.float32, size=sequence_length)

        # first input
        input = inputs[:, 0]

        #TODO:conv for reshape 64,64,2048 -> 64,64,1
        # def init_weights(shape):
        #     return tf.Variable(tf.random_normal(shape, stddev=0.01))
        # matrix = init_weights([1, 1, 2048, 1])

        # one timestep of RNN
        def _time_step(time, input, outputs_ta, state):
            input = input_operation(input, name='input_operation')
            output, state = cell(input, state, scope=scope)

            if output_dim is not None:
                output = output_activation(fc(output, output_dim, 'rnn_output'))

            #TODO:my
            # else:
            #     output = tf.nn.conv2d(output, matrix, strides=[1, 1, 1, 1], padding='SAME')


            output = output_operation(output, name='output_operation')
            outputs_ta = outputs_ta.write(time, output)
            time = time + 1

            if output_conditioned and (not ground_conditioned):
                input = recurrent_activation(output)
            if (not output_conditioned) or ground_conditioned:
                input = inputs[:, tf.mod(time, sequence_length)]

            return time, input, outputs_ta, state

        # do the iterations
        _, _, outputs_ta, state = tf.while_loop(
            cond=lambda time, *_: tf.less(time, sequence_length),
            body=_time_step,
            loop_vars=(time, input, outputs_ta, state),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        gathered = outputs_ta.gather(tf.range(0, sequence_length))

        # reset the order of dimension & return
        if len(gathered.get_shape().as_list()) == 3:
            return tf.transpose(gathered, [1, 0, 2]), state
        else:
            return tf.transpose(gathered, [1, 0, 2, 3, 4]), state


def fc(tensor, dim, name):
    in_shape = tensor.get_shape().as_list()

    with tf.variable_scope(name):
        weights = tf.get_variable("weights", shape=[in_shape[-1], dim],
                                  initializer=tf.random_uniform_initializer(-0.01, 0.01))
        biases = tf.get_variable("biases", shape=[dim],
                                 initializer=tf.constant_initializer(0.0))

        fc = tf.nn.xw_plus_b(tensor, weights, biases)
        return fc


class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_shape(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_shape(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self.shape

        zeros = tuple(tf.zeros([batch_size, shape[1], shape[2], shape[3]])
                               for _ in range(len(self.state_shape)))
        zeros = tf.contrib.rnn.LSTMStateTuple(zeros[0], zeros[1])


        return zeros


# ConvLSTMCell
# original: https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow
# modified: Woobin Im
class ConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, use_peepholes=False,
                 forget_bias=1.0, input_size=None, activation=tf.nn.tanh,
                 initializer=None):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          activation: Activation function of the inner states.
        """
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)

        if input_size is not None:
            tf.logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self._use_peepholes = use_peepholes##
        self._forget_bias = forget_bias
        self._activation = activation
        self._initializer = initializer ##


    @property
    def state_shape(self):
        #TODO:original
        return tf.contrib.rnn.LSTMStateTuple(self.shape, self.shape)

    @property
    def output_shape(self):
        return self.shape

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__,
                               initializer=self._initializer):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = state

            concat = _conv([inputs, h], self.filter_size, self.shape[3] * 4, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=3, value=concat, num_or_size_splits=4)

            # Diagonal connections
            if self._use_peepholes:
                dtype = i.dtype
                w_f_diag = tf.get_variable(
                    "w_f_diag", shape=[1, self.shape[1], self.shape[2], self.shape[3]], dtype=dtype)
                w_i_diag = tf.get_variable(
                    "w_i_diag", shape=[1, self.shape[1], self.shape[2], self.shape[3]], dtype=dtype)
                w_o_diag = tf.get_variable(
                    "w_o_diag", shape=[1, self.shape[1], self.shape[2], self.shape[3]], dtype=dtype)

            if self._use_peepholes:
                new_c = (c * tf.nn.sigmoid(f + self._forget_bias + w_f_diag * c)
                         + tf.nn.sigmoid(i + w_i_diag * c) * self._activation(j))
            else:
                new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i)
                         * self._activation(j))

            if self._use_peepholes:
                new_h = self._activation(new_c) * tf.nn.sigmoid(o + w_o_diag * new_c)
            else:
                new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            return new_h, new_state


class MultiRNNCell(ConvRNNCell):
    """RNN cell composed sequentially of multiple simple cells."""

    def __init__(self, cells):
        """Create a RNN cell composed sequentially of a number of RNNCells.
        Args:
          cells: list of RNNCells that will be composed in this order.
        Raises:
          ValueError: if cells is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")

        self._cells = cells

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)

    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        with tf.variable_scope(scope or "multi_rnn_cell"):
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("cell_%d" % i):
                    cur_state = state[i]
                    cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, new_states


def _conv(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    """convolution:
    Args:
      args: a 4D Tensor or a list of 4D, batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      num_features: int, number of features.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 4D Tensor with shape [batch h w num_features]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.


    with tf.variable_scope(scope or "Conv"):

        matrix = tf.get_variable(
          "filters", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
        # filter = [5, 5, 4096, 8192]
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(values=args, axis=3), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "biases", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term
