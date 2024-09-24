import numpy as np
import tensorflow as tf
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

def class_acc(pred,gt):
    # Calculates class accuracy
    pred = np.array(pred)
    gt = np.array(gt)
    correctpreds = np.sum(pred == gt)
    totalpreds = len(gt)
    accuracy = correctpreds / totalpreds

    return accuracy

def calculate_means(x_train, classes):
    # Calculates means for MNIST classes
    num_classes = 10
    means = np.array([np.mean(x_train[classes == dataclass], 
                              axis=0) for dataclass in range(num_classes)])
    
    return means

def calculate_covariance(x_train, classes):
    # Calculates covariances for MNIST classes
    num_classes = 10
    covariances = np.array([np.cov(x_train[classes == dataclass], rowvar=False) 
                            + np.eye(x_train.shape[1]) * 1e-6  # Regularization
                            for dataclass in range(num_classes)])
    
    return covariances

def predict_class(x_test, means, covariances):
    num_classes = len(means)
    num_samples = x_test.shape[0]
    log_likelihoods = np.zeros((num_samples, num_classes))

    # Compute log-likelihoods for each class for each test sample
    mvns = [multivariate_normal(mean=means[class_label], cov=covariances[class_label]) for class_label in range(num_classes)]
    log_likelihoods = np.array([mvn.logpdf(x_test) for mvn in mvns]).T
    
    # Return the predicted class with the highest log-likelihood
    return np.argmax(log_likelihoods, axis=1)

def test_full_bayes(x_test, y_test, means, covariances):
    # Predict the class for all test samples at once
    predicted_classes = predict_class(x_test, means, covariances)

    # Calculate accuracy
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Full Bayes classification accuracy: {accuracy:.2f}")

def main():
    # Load the MNIST data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_flattened, x_test_flattened = preprocess_data(x_train, x_test)
    x_train_noise = add_noise(x_train_flattened)

    means = calculate_means(x_train_noise, y_train)
    covariances = calculate_covariance(x_train_noise, y_train)

    test_full_bayes(x_test_flattened, y_test, means, covariances)

if __name__ == "__main__":
    main()