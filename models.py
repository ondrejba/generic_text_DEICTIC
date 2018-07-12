import tensorflow as tf
import tensorflow.contrib.layers as layers


def _mlp(hiddens, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def mlp(hiddens=[]):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, *args, **kwargs)

def embed(index_t, num_items, embedding_size):

    embeddings_var = tf.get_variable("embeddings", [num_items, embedding_size], trainable=True)
    embedded_t = tf.nn.embedding_lookup(embeddings_var, index_t)

    return embedded_t

def multiplex(vector_t, size):
    """
    Go from batch_size x num_filters to batch_size x size x size x num_filters by repeating the vector
    size ^ 2 times.
    :param vector_t:    Batch of input vectors.
    :param size:        Height and width.
    :return:            Tensor of rank 4 that can be added to the output of the convolution.
    """

    vector_t = tf.expand_dims(vector_t, axis=1)
    vector_t = tf.expand_dims(vector_t, axis=1)

    vector_t = tf.tile(vector_t, (1, size, size, 1))

    return vector_t

def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            return state_score + action_scores_centered
        else:
            return action_scores

def _cnn_to_mlp_symbolic(convs, hiddens, deictic_input, abstract_input, num_actions, num_abstract_actions,
                         abstract_embedding_size, scope, reuse=False):

    with tf.variable_scope(scope, reuse=reuse):

        out = deictic_input
        with tf.variable_scope("convnet"):

            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)

        out = layers.flatten(out)

        with tf.variable_scope("embedding"):

            embedded_abstract_action_t = embed(abstract_input, num_abstract_actions, abstract_embedding_size)
            embedded_abstract_action_t = tf.nn.relu(embedded_abstract_action_t)

        out = tf.concat((out, embedded_abstract_action_t), axis=1)

        with tf.variable_scope("action_value"):

            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        return action_scores


def _cnn_to_mlp_symbolic_multiplex(convs, hiddens, deictic_input, abstract_input, num_actions, num_abstract_actions,
                                   abstract_embedding_size, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):

        out = deictic_input
        with tf.variable_scope("convnet"):

            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)

        with tf.variable_scope("embedding"):

            embedded_abstract_action_t = embed(abstract_input, num_abstract_actions, abstract_embedding_size)
            embedded_abstract_action_t = tf.nn.relu(embedded_abstract_action_t)

            multiplex_t = multiplex(embedded_abstract_action_t, tf.shape(out)[1])

        out += multiplex_t
        out = layers.flatten(out)


        with tf.variable_scope("action_value"):

            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        return action_scores

def cnn_to_mlp(convs, hiddens, dueling=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, *args, **kwargs)

def cnn_to_mlp_symbolic(convs, hiddens):

    return lambda *args, **kwargs: _cnn_to_mlp_symbolic(convs, hiddens, *args, **kwargs)

def cnn_to_mlp_symbolic_multiplex(convs, hiddens):

    return lambda *args, **kwargs: _cnn_to_mlp_symbolic_multiplex(convs, hiddens, *args, **kwargs)