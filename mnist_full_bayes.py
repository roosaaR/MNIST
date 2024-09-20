import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal


def preprocess_data(x_train, x_test):
    # Flattens the images and normalizes pixel values to the range [0, 1]
    x_train_flattened = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test_flattened = x_test.reshape(x_test.shape[0], -1) / 255.0

    return x_train_flattened, x_test_flattened

def add_noise(x_train):
    # Adds Zero mean white noise to training data
    noise = np.random.normal(loc=0.0, scale=0.1, size=x_train.shape)
    x_train_noise = x_train + noise
    
    return np.clip(x_train_noise, 0.0, 1.0)


def train_data(x_train, y_train):
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)  

    # Train the classifier
    knn.fit(x_train, y_train)

    return knn

def class_acc(pred,gt):
    # Calculates class accuracy
    pred = np.array(pred)
    gt = np.array(gt)
    correctpreds = np.sum(pred == gt)
    totalpreds = len(gt)
    accuracy = correctpreds / totalpreds

    return accuracy

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

def calculate_covariance(x_train, classes):
    num_classes = 10

    covariances = []

    for dataclass in range(num_classes):
        class_data = x_train[classes == dataclass]
        covariance = np.cov(class_data, rowvar=False)
        rank = np.linalg.matrix_rank(covariance)
        print(f"Rank of class {dataclass} covariance matrix: {rank}")
        covariances.append(covariance)

    return covariances


def predict_class(x_test, means, covariances):
    num_classes = len(means)
    probabilities = []
    
    for class_label in range(num_classes):
        # Create a multivariate normal distribution for the class
        distribution = multivariate_normal(mean=means[class_label], cov=covariances[class_label])
        # Compute the log probability (logpdf) of the test data belonging to this class
        log_prob = distribution.logpdf(x_test)
        probabilities.append(log_prob)
    
    # Return the class with the highest log likelihood
    return np.argmax(probabilities)

def test_naive_bayes(x_test, y_test, means, variances):

    correct_predictions = 0

    for i in range(len(x_test)):
        predicted_class = predict_class(x_test[i], means, variances)
        if predicted_class == y_test[i]:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(x_test)

    print(f"Naive Bayes classification accuracy: {accuracy:.2f}")



def main():
    # Load the MNIST data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_flattened, x_test_flattened = preprocess_data(x_train, x_test)

    x_train_noise = add_noise(x_train_flattened)

    #knn = train_data(x_train_noise, y_train)

    #test_data(knn, x_test_flattened, y_test)

    means = calculate_means(x_train_noise, y_train)
    covariances = calculate_covariance(x_train_noise, y_train)

    test_naive_bayes(x_test_flattened, y_test, means, covariances)


if __name__ == "__main__":
    main()