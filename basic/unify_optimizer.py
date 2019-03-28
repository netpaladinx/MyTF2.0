import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras

if __name__ == '__main__':

    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, name='SGD')

    optimizer = keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-8, decay=0.0, name='Adadelta')

    optimizer = keras.optimizers.Adagrad(learning_rate=0.001, epsilon=1e-8, decay=0.0, initial_accumulator_value=0.1, name='Adagrad')

    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False, name='Adam')

    optimizer = keras.optimizers.Ftrl(learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
                                      l1_regularization_strength=0.0, l2_regularization_strength=0.0, l2_shrinkage_regularization_strength=0.0,
                                      name='Ftrl')

    optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8, decay=0.0, centered=False, name='RMSProp')

    optimizer = keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, name='Adamax')

    optimizer = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, name='Nadam')