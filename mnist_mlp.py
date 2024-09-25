import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.utils import to_categorical


'''Write Python code that 1) makes a full connected neural network that takes
the MNIST images as 784-dimensional vectors input. The network produces 10
outputs each representing one class. Start with something simple, such as only
5 hidden layer neurons.

Then, 2) your code should train a network with the MNIST trainining data. Set
a suitable learning rate and number of epochs. Plot the training loss curve to
confirm that the network learns.'''

def preprocess_data(x_train, x_test, y_train, y_test):
    # Flattens the 28 x 28 px images to a 784-dimensional vectors and 
    # normalizes pixel values.
    x_train_flattened = x_train.reshape(x_train.shape[0], -1) / 255
    x_test_flattened = x_test.reshape(x_test.shape[0], -1) / 255
    y_train = to_categorical(y_train, num_classes=10) # One-hot encoding
    y_test = to_categorical(y_test, num_classes=10)

    return x_train_flattened, x_test_flattened, y_train, y_test


def plot_data(tr_hist):
    plt.plot(tr_hist.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training'], loc='upper right')
    plt.show()


def main():
    # Load the MNIST data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test, y_train, y_test = preprocess_data(x_train, x_test, y_train, y_test)

    model = Sequential()
    model.add(Dense(64, input_dim=784, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))

    sgd = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    num_of_epochs = 10

    tr_hist = model.fit(x_train, y_train, epochs=num_of_epochs, verbose=1)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    plot_data(tr_hist)

if __name__ == "__main__":
    main()