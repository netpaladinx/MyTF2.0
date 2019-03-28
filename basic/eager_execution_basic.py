import numpy as np
import tensorflow as tf


def print_tensors():
    print(tf.add(1, 2))
    print(tf.add([1, 2], [3, 4]))
    print(tf.square(5))
    print(tf.reduce_sum([1, 2, 3]))
    print(tf.io.encode_base64("hello world"))
    print(tf.square(2) + tf.square(3))

    x = tf.matmul([[1]], [[2, 3]])
    print(x.shape)
    print(x.dtype)


def with_numpy():
    ndarray = np.ones([3, 3])

    # numpy arrays to Tensors
    tensor = tf.multiply(ndarray, 42)
    print(tensor)

    # Tensors to numpy arrays
    print(np.add(tensor, 1))

    # Tensors to numpy arrays (explicitly)
    print(tensor.numpy())


def with_device():
    x = tf.random.uniform([3, 3])

    print(x)
    print(tf.test.is_gpu_available())
    print(x.device.endswith('GPU:0'))

    import time

    def time_matmul(x):
        start = time.time()
        for loop in range(10):
            tf.matmul(x, x)

        result = time.time() - start
        print("10 loops: {:.2f}ms".format(1000*result))

    # Force execution on CPU
    with tf.device("CPU:0"):
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith("CPU:0")
        time_matmul(x)


def with_dataset():
    # Create a source Dataset from a tensor

    ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
    ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

    for x in ds_tensors:
        print(x)

    # Create a source Dataset from a file

    import tempfile
    _, filename = tempfile.mkstemp()
    with open(filename, 'w') as f:
        f.write(
"""Line 1
Line 2
Line 3
"""
        )

    ds_file = tf.data.TextLineDataset(filename)
    ds_file = ds_file.batch(2)

    for x in ds_file:
        print(x)


if __name__ == '__main__':
    #print_tensors()

    #with_numpy()

    #with_device()

    with_dataset()