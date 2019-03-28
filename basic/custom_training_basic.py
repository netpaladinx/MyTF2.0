import tensorflow as tf


def use_variables():
    v = tf.Variable(1.)
    print(v.numpy())

    v.assign(3.)
    print(v.numpy())

    v.assign(tf.square(v))
    print(v.numpy())


# Define the model
class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.)
        self.b = tf.Variable(0.)

    def __call__(self, x):
        return self.W * x + self.b


# Define the loss
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


# Define the train
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


if __name__ == '__main__':
    #use_variables()

    model = Model()
    print(model(3.).numpy())

    # Obtain training data

    TRUE_W = 3.
    TRUE_b = 2.
    NUM_EXAMPLES = 1000

    inputs = tf.random.normal(shape=[NUM_EXAMPLES])
    noise = tf.random.normal(shape=[NUM_EXAMPLES])
    outputs = inputs * TRUE_W + TRUE_b + noise

    # Plot

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.show()

    print('Current loss: ')
    print(loss(model(inputs), outputs).numpy())

    # Define a training loop

    Ws, bs = [], []  # collect the history
    epochs = range(10)
    for epoch in epochs:
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(model(inputs), outputs)

        train(model, inputs, outputs, learning_rate=0.1)
        print('Epoch: {:2d}: W={:1.2f} b={:1.2f}, loss={:2.5f}'.format(
            epoch, Ws[-1], bs[-1], current_loss
        ))

    # Plot
    plt.plot(epochs, Ws, 'r', epochs, bs, 'b')
    plt.plot([TRUE_W] * len(epochs), 'r--', [TRUE_b] * len(epochs), 'b--')
    plt.legend(['W', 'b', 'true W', 'true b'])
    plt.show()

