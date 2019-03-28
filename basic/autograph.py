import tensorflow as tf


@tf.function
def add(a, b):
    return a + b


@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)


conv_layer = tf.keras.layers.Conv2D(100, 3)

@tf.function
def conv_fn(image):
    return conv_layer(image)


lstm_cell = tf.keras.layers.LSTMCell(10)

@tf.function
def lstm_fn(input, state):
    return lstm_cell(input, state)


a = tf.Variable(1.0)
b = tf.Variable(2.0)  # create a variable out of tf.function (recommended)

@tf.function
def f(x, y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b


@tf.function
def f2(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x


@tf.function
def f3(x):
    for i in range(10):  # statically unrolled
        tf.print(x)
    for i in tf.range(10):  # dynamically
        tf.print(x)


@tf.function
def f4(x):
    ta = tf.TensorArray(tf.float32, size=10)
    for i in tf.range(10):
        x += x
        ta = ta.write(i, x)
    return ta.stack()


if __name__ == '__main__':
    print(add(tf.ones([2, 2]), tf.ones([2, 2])))
    print(dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2])))

    v = tf.Variable(1.)
    with tf.GradientTape() as tape:
        result = add(v, 1.)
    print(tape.gradient(result, v))

    import timeit

    image = tf.zeros([1, 200, 200, 100])
    # warm up
    conv_layer(image);
    conv_fn(image)
    print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
    print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
    print("Note how there's not much difference in performance for convolutions")

    input = tf.zeros([10, 10])
    state = [tf.zeros([10, 10])] * 2
    # warm up
    lstm_cell(input, state);
    lstm_fn(input, state)
    print("eager lstm:", timeit.timeit(lambda: lstm_cell(input, state), number=10))
    print("function lstm:", timeit.timeit(lambda: lstm_fn(input, state), number=10))

    print(f(1., 2.))

    print(f2(tf.random.uniform([10])))
    print(tf.autograph.to_code(f2))
