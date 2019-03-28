import os
import time

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras


def preprocess():
    # Download the Shakespeare data file
    path_to_file = keras.utils.get_file('shakespeare.txt',
                                        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    # Load the text data
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    print('Length of text: {} characters'.format(len(text)))
    print(text[:250])

    # Create a vocabulary of characters
    vocab = sorted(set(text))  # character-based
    print('{} unique characters'.format(len(vocab)))

    # Make bidirectional indexing
    char2idx = {u: i for i, u in enumerate(vocab)}  # dict: character => idx
    idx2char = np.array(vocab)  # np.array, shape: (n_characters,)

    # Convert to the integer data (vectorizing the text)
    text_as_int = np.array([char2idx[c] for c in text])  # shape: (len_text,)

    print('{')
    for char, _ in zip(char2idx, range(20)):
        print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    print('  ...\n}')
    print('{} ---- characters mapped to int --- > {}'.format(repr(text[:13]), text_as_int[:13]))

    return text, text_as_int, char2idx, idx2char, vocab


# Create training examples and targets
def create_dataset(text, text_as_int, char2idx, idx2char, vocab):
    # Slice text to sequences

    seq_length = 100
    examples_per_epoch = len(text) // seq_length

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    for item in sequences.take(5):
        print(repr(''.join(idx2char[item.numpy()])))

    # Split sequences to training text and target text

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data: ', repr(''.join(idx2char[target_example.numpy()])))

    return dataset


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),  # `stateful` needs batch_size
        keras.layers.Dense(vocab_size)
    ])
    return model


def loss_fn(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def train(dataset, char2idx, idx2char, vocab):
    # Create training batches
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Build the model
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024
    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
    model.summary()

    # Try the model
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)  # (sequence_length, 1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()  # (sequence_length,)
        print(sampled_indices)

        print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
        print()
        print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

        # Train the model
        example_batch_loss = loss_fn(target_example_batch, example_batch_predictions)
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("scalar loss:      ", example_batch_loss.numpy().mean())

    # Compile the model
    model.compile(optimizer='adam', loss=loss_fn)

    # Configure checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    # Execute the training
    EPOCHS = 3  # 10
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # Restore the latest checkpoint (need to rebuild the model because of the change of batch_size)
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))  # explicitly call `.build()`
    model.summary()

    # The prediction loop
    print(generate_text(model, u"ROMEO: ", char2idx, idx2char))


optimizer = keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(inp, target, model):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(target, predictions))  # from_logits=False
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# Customized Training
def custom_train(dataset, vocab):
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

    EPOCHS = 3
    BATCH_SIZE = 64

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

    for epoch in range(EPOCHS):
        start = time.time()

        for (batch_n, (inp, target)) in enumerate(dataset):
            loss = train_step(inp, target)

            if batch_n % 100 == 0:
                print("Epoch {} Batch {} Loss {}".format(epoch + 1, batch_n, loss))

        if (epoch + 1) % 5 == 0:
            model.save_weights(checkpoint_prefix.format(epoch=epoch))

        print("Epoch {} Loss {:.4f}".format(epoch + 1, loss))
        print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

    model.save_weights(checkpoint_prefix.format(epoch=epoch))


def generate_text(model, start_string, char2idx, idx2char):
    num_generate = 1000

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predicted_id = tf.random.categorical(predictions / temperature, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


if __name__ == '__main__':
    text, text_as_int, char2idx, idx2char, vocab = preprocess()

    dataset = create_dataset(text, text_as_int, char2idx, idx2char, vocab)

    train(dataset, char2idx, idx2char, vocab)
