import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import tqdm


def download_MS_COCO():
    annotation_zip = keras.utils.get_file('captions.zip', cache_subdir=os.path.abspath('.'),
                                          origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract=True)
    annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = keras.utils.get_file(name_of_zip, cache_subdir=os.path.abspath('.'),
                                         origin='http://images.cocodataset.org/zips/train2014.zip',
                                         extract=True)
        PATH = os.path.dirname(image_zip) + '/train2014/'
    else:
        PATH = os.path.abspath('.') + '/train2014/'
    return annotation_file, PATH


def create_dataset(annotation_file, PATH):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

    num_examples = 30000
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]
    return train_captions, img_name_vector


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = keras.applications.inception_v3.preprocess_input(img)  # place the pixels in the range of -1 of 1
    return img, image_path


def create_feature_model_based_on_inception_v3():
    image_model = keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = keras.Model(new_input, hidden_layer)
    return image_features_extract_model


def caching_inception_v3_features(img_name_vector):
    encode_train = sorted(set(img_name_vector))
    print(len(encode_train))

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    image_features_extract_model = create_feature_model_based_on_inception_v3()

    for img, path in tqdm.tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode('utf-8')
            np.save(path_of_feature, bf.numpy())

    return image_features_extract_model


def preprocess_captions(train_captions, img_name_vector):
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)

    top_k = 5000
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token='<unk>',
                                                   filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_length = calc_max_length(train_seqs)

    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, cap_vector,
                                                                        test_size=0.2, random_state=0)
    print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))
    return img_name_train, img_name_val, cap_train, cap_val, tokenizer, max_length


def create_tf_dataset(tokenizer, img_name_train, cap_train):
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = len(tokenizer.word_index) + 1
    num_steps = len(img_name_train) // BATCH_SIZE


    def map_func(img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        return img_tensor, cap

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, embedding_dim, BATCH_SIZE, units, vocab_size


class BahdanauAttention(keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, features, hidden):
        # features: (batch_size, 64, embedding_dim)
        # hidden: (batch_size, hidden_size)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = keras.layers.Dense(embedding_dim)  # (batch_size, 64, embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()

        self.units = units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(units, return_sequences=True, return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc1 = keras.layers.Dense(self.units)
        self.fc2 = keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def train():
    features_shape = 2048
    attention_features_shape = 64

    img, image_path = download_MS_COCO()
    print(img, image_path)

    train_captions, img_name_vector = create_dataset(img, image_path)
    img_name_train, img_name_val, cap_train, cap_val, tokenizer, max_length = preprocess_captions(train_captions, img_name_vector)

    image_features_extract_model = caching_inception_v3_features(img_name_vector)
    dataset, embedding_dim, BATCH_SIZE, units, vocab_size = create_tf_dataset(tokenizer, img_name_train, cap_train)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = keras.optimizers.Adam()
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def loss_fn(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    checkpoint_path = "./image_captioning_checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

    loss_plot = []

    @tf.function
    def train_step(img_tensor, target):
        loss = 0

        hidden = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_fn(target[:, i], predictions)

                # teaching forcing
                dec_input = tf.expand_dims(target[:, 1], 1)

        total_loss = (loss / int(target.shape[1]))
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    EPOCHS = 20
    num_steps = len(img_name_train) // BATCH_SIZE

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, batch, batch_loss.numpy() / int(target.shape[1])))

        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch+1, total_loss/num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()

    return max_length, attention_features_shape, decoder, encoder, image_features_extract_model, tokenizer, img_name_val, cap_val


def evaluate(image, max_length, attention_features_shape, decoder, encoder, image_features_extract_model, tokenizer):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def try_image(*args):
    image_url = 'https://tensorflow.org/images/surf.jpg'
    image_extention = image_url[-4:]
    image_path = keras.utils.get_file('image'+image_extention, origin=image_url)
    result, attention_plot = evaluate(image, *args)
    print('Prediction Caption:', ' '.join(result))
    plot_attention(image_path, result, attention_plot)
    Image.open(image_path)


if __name__ == '__main__':
    max_length, attention_features_shape, decoder, encoder, image_features_extract_model, tokenizer, img_name_val, cap_val = train()

    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image, max_length, attention_features_shape, decoder, encoder, image_features_extract_model, tokenizer)

    print('Real Caption:', real_caption)
    print('Prediction Caption:', ' '.join(result))
    plot_attention(image, result, attention_plot)

    Image.open(img_name_val[rid])
