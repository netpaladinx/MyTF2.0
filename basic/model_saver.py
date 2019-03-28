"""
    Use tf.keras to save TensorFlow models.
"""
import os

import tensorflow as tf
import tensorflow.keras as keras


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation=keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=keras.activations.softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':

    # Get an example dataset

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    # Build a model

    model = create_model()
    model.summary()

    # Save checkpoints during training

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels),
              callbacks=[cp_callback])

    # Load weights from the checkpoint

    model_2 = create_model()
    loss, acc = model_2.evaluate(test_images, test_labels)
    print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

    model_2.load_weights(checkpoint_path)
    loss, acc = model_2.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    # Checkpoint callback options

    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True,
                                                  # Save weights, every 5-epochs
                                                  period=5)  # default: only save the 5 most recent checkpoints

    model = create_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels), verbose=0,
              callbacks=[cp_callback])
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)

    # Reset the model and load the latest checkpoint

    model_2 = create_model()
    model_2.load_weights(latest)
    loss, acc = model_2.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    # Manually save weights

    model_2.save_weights('./checkpoints/my_checkpoint')

    model_3 = create_model()
    model_3.load_weights('./checkpoints/my_checkpoint')

    loss, acc = model_3.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    # Save the entire model

    model = create_model()
    model.fit(train_images, train_labels, epochs=5)

    # save to a HDF5 file
    # - weights
    # - model's configuration (architecture)
    # - optimizer configuration (loss the state of the optimizer, so need to re-compile after loading)
    model.save('my_model.h5')

    # recreate the exact same model, including weights and optimizer.
    new_model = keras.models.load_model('my_model.h5')
    new_model.summary()
    loss, acc = new_model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    