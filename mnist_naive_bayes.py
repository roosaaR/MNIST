import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier


def class_acc(pred,gt):
    # Calculates class accuracy
    pred = np.array(pred)
    gt = np.array(gt)
    correctpreds = np.sum(pred == gt)
    totalpreds = len(gt)
    accuracy = correctpreds / totalpreds

    return accuracy


def preprocess_data(x_train, x_test):
    # Flatten the images and normalize pixel values to the range [0, 1]
    x_train_flattened = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test_flattened = x_test.reshape(x_test.shape[0], -1) / 255.0

    return x_train_flattened, x_test_flattened

def train_data(x_train, y_train):
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)  

    # Train the classifier
    knn.fit(x_train, y_train)

    return knn

def test_data(knn, x_test, y_test):
    y_pred = knn.predict(x_test)
    accuracy = class_acc(y_pred, y_test)  
    print(f"Classification accuracy is {accuracy:.2f}")

def calculate_means(x_train, classes):
    num_classes = 10

    means = []

    for dataclass in range(num_classes): 
        class_data = x_train[classes == dataclass]
        mean = np.mean(class_data, axis=0)
        means.append(mean)

    return np.array(means)

def calculate_variances(x_train, classes):
    num_classes = 10
    variances = []

    for dataclass in range(num_classes): 
        class_data = x_train[classes == dataclass]
        variance = np.var(class_data, axis=0)
        variances.append(variance)

    return np.array(variance)



def main():
    # Load the MNIST data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_flattened, x_test_flattened = preprocess_data(x_train, x_test)

    knn = train_data(x_train_flattened, y_train)

    test_data(knn, x_test_flattened, y_test)

    means = calculate_means(x_train_flattened, y_train)
    variances = calculate_variances(x_train_flattened, y_train)

if __name__ == "__main__":
    main()