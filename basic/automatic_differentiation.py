import tensorflow as tf


def gradient_tapes():
    x = tf.ones((2, 2))

    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.reduce_sum(x)
        z = tf.multiply(y, y)

    dz_dx = t.gradient(z, x)  # by default, the resources held by a GradientTape are released as soon as
                              # GradientTape.gradient() method is called.
    print(z)
    print(x)
    print(dz_dx)

    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.reduce_sum(x)
        z = tf.multiply(y, y)

    dz_dy = t.gradient(z, y)
    print(dz_dy)

    x = tf.constant([1., 3.])
    with tf.GradientTape(persistent=True) as t:
        t.watch(x)
        y = x * x
        z = y * y

    dz_dx = t.gradient(z, x)
    dy_dx = t.gradient(y, x)
    del t

    print(y)
    print(z)
    print(dz_dx)
    print(dy_dx)


def recording_control_flow():
    def f(x, y):
        output = 1.
        for i in range(y):
            if i > 1 and i < 5:
                output = tf.multiply(output, x)
        return output

    def grad(x, y):
        with tf.GradientTape() as t:
            t.watch(x)
            out = f(x, y)
        return t.gradient(out, x)

    x = tf.convert_to_tensor(2.)
    print(grad(x, 6))
    print(grad(x, 4))


def high_order_gradients():
    x = tf.Variable(1.)

    with tf.GradientTape() as t:
        with tf.GradientTape() as t2:
            y = x * x * x
        dy_dx = t2.gradient(y, x)
    d2y_dx2 = t.gradient(dy_dx, x)

    print(dy_dx)
    print(d2y_dx2)


if __name__ == '__main__':
    #gradient_tapes()

    #recording_control_flow()

    high_order_gradients()