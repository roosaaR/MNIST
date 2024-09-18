import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier



def class_acc(pred,gt):
    pred = np.array(pred)
    gt = np.array(gt)

    correctpreds = np.sum(pred == gt)
    totalpreds = len(gt)

    accuracy = correctpreds / totalpreds

    return accuracy

def test_accuracy():
    num_test_samples = len(y_test)
    random_predictions = np.random.randint(0, 10, num_test_samples)

    accuracy = class_acc(random_predictions, y_test)
    print(f"Accuracy of random predictions on MNIST test set: {accuracy:.2f}")

def preprocess_data(x_train, x_test):
    # Flatten the images and normalize pixel values to the range [0, 1]
    x_train_flattened = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test_flattened = x_test.reshape(x_test.shape[0], -1) / 255.0

    return x_train_flattened, x_test_flattened

def train_data(x_train_flattened, y_train):
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)  # You can change n_neighbors to whatever you like

    # Train the classifier
    knn.fit(x_train_flattened, y_train)


def main():
    # Load the MNIST data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_flattened, x_test_flattened = preprocess_data(x_train, y_train)

    train_data(x_train_flattened, y_train)

if __name__ == "__main__":
    main()