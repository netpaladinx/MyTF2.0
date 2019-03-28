import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras


if __name__ == '__main__':

    # Download the dataset

    train_dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'
    train_dataset_fp = keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                            origin=train_dataset_url)
    print("Local copy of the dataset file: {}".format(train_dataset_fp))

    # Inspect the data

    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    feature_names = column_names[:-1]
    label_name = column_names[-1]
    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

    # Create a dataset

    batch_size = 32
    train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_fp, batch_size,
                                                          column_names=column_names,
                                                          label_name=label_name,
                                                          num_epochs=1)
    features, labels = next(iter(train_dataset))
    print(features)
    plt.scatter(features['petal_length'].numpy(),
                features['sepal_length'].numpy(),
                c=labels.numpy(),
                cmap='viridis')
    plt.xlabel("Petal length")
    plt.ylabel("Sepal lenght")
    #plt.show()

    # Pack features

    def pack_features_vector(features, labels):
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    train_dataset = train_dataset.map(pack_features_vector)
    features, labels = next(iter(train_dataset))
    print(features)

    # Create a model

    model = keras.Sequential([
        keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(3)
    ])

    # Use the model

    predictions = tf.argmax(model(features), axis=1)
    print("Prediction: {}".format(predictions))
    print("Labels: {}".format(labels))

    # Define a loss

    def loss(model, x, y):
        y_ = model(x)
        return tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, y_, from_logits=True))

    l = loss(model, features, labels)
    print("Loss test: {}".format(l))

    # Calculate the gradients

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


