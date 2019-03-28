import tensorflow as tf
import tensorflow.keras as keras


def keras_layer():
    layer = keras.layers.Dense(10)
    print(layer(tf.ones([10, 5])))
    print(layer.variables)
    print(layer.trainable_variables)
    print(layer.kernel)
    print(layer.bias)


# Implement a custom layer
class MyDenseLayer(keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel', shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


# Implement a model by composing layers
class ResnetIdentityBlock(keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = keras.layers.Conv2D(filters1, (1,1))
        self.bn2a = keras.layers.BatchNormalization()

        self.conv2b = keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = keras.layers.BatchNormalization()

        self.conv2c = keras.layers.Conv2D(filters3, (1,1))
        self.bn2c = keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


if __name__ == '__main__':
    #keras_layer()

    layer = MyDenseLayer(10)
    print(layer(tf.ones([10, 5])))
    print(layer.trainable_variables)

    block = ResnetIdentityBlock(1, [1,2,3])
    print(block(tf.ones([1,2,3,3])))
    print([x.name for x in block.trainable_variables])

    block_2 = keras.Sequential([keras.layers.Conv2D(1, (1,1)),
                                keras.layers.BatchNormalization(),
                                keras.layers.Conv2D(2, 1, padding='same'),
                                keras.layers.BatchNormalization(),
                                keras.layers.Conv2D(3, (1,1)),
                                keras.layers.BatchNormalization()])
    print(block_2(tf.ones([1,2,3,3])))