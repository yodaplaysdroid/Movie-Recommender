import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import os


# GPU Configuration
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Number of GPU: ', len(physical_devices))
print(physical_devices)

if len(physical_devices) != 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Initializing Neural Network Model
nn = keras.models.Sequential(keras.layers.Dense(units=32, input_shape=(24,), activation='relu'))
nn.add(keras.layers.Dense(64))
nn.add(keras.layers.LeakyReLU(alpha=0.05))
nn.add(keras.layers.Dense(128))
nn.add(keras.layers.LeakyReLU(alpha=0.05))
nn.add(keras.layers.Dense(256))
nn.add(keras.layers.LeakyReLU(alpha=0.05))
nn.add(keras.layers.Dense(512))
nn.add(keras.layers.LeakyReLU(alpha=0.05))
nn.add(keras.layers.Dense(1024))
nn.add(keras.layers.LeakyReLU(alpha=0.05))
nn.add(keras.layers.Dense(units=9, activation='softmax'))


# Defining NN Optimizer, Learning Rate and Loss Function
nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Loading Train data
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_train, y_train = shuffle(X_train, y_train)


# Training NN model
nn.fit(x=X_train, y=y_train, validation_split=0.1, batch_size=10, epochs=10, verbose=1)


# Export model
nn.save('data/model.h5')