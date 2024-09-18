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

def test_class_acc():
    # Tests the class_acc function
    random_preds = np.random.randint(0, 10, 1000)  # Random predictions between 0 and 9
    random_gt = np.random.randint(0, 10, 1000)  # Random ground truth labels between 0 and 9
    accuracy = class_acc(random_preds, random_gt)
    print(f"Accuracy with random predictions: {accuracy:.2f}")


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
    test_class_acc()
    y_pred = knn.predict(x_test)
    accuracy = class_acc(y_pred, y_test)  
    print(f"Classification accuracy is {accuracy:.2f}")


def main():
    # Load and read the MNIST data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_flattened, x_test_flattened = preprocess_data(x_train, x_test)

    knn = train_data(x_train_flattened, y_train)

    test_data(knn, x_test_flattened, y_test)

if __name__ == "__main__":
    main()