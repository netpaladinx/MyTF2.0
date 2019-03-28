import tensorflow as tf


if __name__ == '__main__':

    # RNN cell
    # 1. individual component within RNN layer
    # 2. calculates the output for the current timestep
    # 3. contains weights
    # 4. defines the numerical calculation
    # 5. User cannot use RNN cell alone (must work with RNN layer interface)

    # RNN layer
    # 1. connects previous timestep
    # 2. drives the run loop
    # 3. returns the final outputs and states

    # TF RNN API
    # 1. static (fixed length, using a symbolic loop) / dynamic (use tf.while_loop, faster, more memory)
    # 2. single_direction / bidirectional (unifying two)
    # 3. tf.nn.raw_rnn (custom run loop given input and output states) (removed in 2.0)
    # 4. tf.nn.static_state_saving_rnn (too long to fit into the memory) (similar to "stateful" RNN in keras)

    # Keras RNN layer
    # 1. base: tf.keras.layers.RNN (the parameter unroll: static/dynamic; Bidirectional wrapper)
    # 2. Keras RNN is object based
    # 3. requires input to be a 3D tensor with shape [batch_size, timestep, input_dim]
    # 4. by default, returns the output for the last timestep

    print(tf.keras)
    from tensorflow.python.keras.api._v2 import keras

    cell = keras.layers.SimpleRNNCell(10)
    rnn = keras.layers.RNN(cell, unroll=True)  # static rnn
    rnn = keras.layers.RNN(cell, unroll=False)  # dynamic rnn
    rnn = keras.layers.Bidirectional(keras.layers.RNN(cell, unroll=True))  # bidirectional static rnn
    rnn = keras.layers.Bidirectional(keras.layers.RNN(cell, unroll=False))  # bidirectional dynamic rnn
    rnn = keras.layers.RNN(cell, stateful=True)  # stateful rnn
    print(rnn)

    num_layers = 3
    model = keras.Sequential()
    for i in range(num_layers):
        model.add(keras.layers.Bidirectional(keras.layers.RNN(cell, unroll=True)))  # stack bidirectional static rnn

    for i in range(num_layers):
        model.add(keras.layers.Bidirectional(keras.layers.RNN(cell, unroll=False)))  # stack bidirectional dynamic rnn

    # Keras RNN cell
    keras.layers.SimpleRNNCell  # vanilla RNN
    keras.layers.LSTMCell  # no peephole, clipping. projection
    keras.layers.GRUCell
    keras.layers.StackedRNNCells
