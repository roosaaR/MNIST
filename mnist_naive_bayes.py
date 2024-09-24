import numpy as np
import tensorflow as tf

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

def add_noise(x_train):
    # Adds white noise to the training data
    noise = np.random.normal(loc=0.0, scale=0.1, size=x_train.shape)
    x_train_noise = x_train + noise
    
    return np.clip(x_train_noise, 0.0, 1.0)

def calculate_means(x_train, classes):
    # Calculates means for MNIST classes
    num_classes = 10
    means = []

    for dataclass in range(num_classes): 
        class_data = x_train[classes == dataclass]
        mean = np.mean(class_data, axis=0)
        means.append(mean)

    return np.array(means)

def calculate_variances(x_train, classes):
    # Calculates variances for MNIST classes
    num_classes = 10
    variances = []

    for dataclass in range(num_classes): 
        class_data = x_train[classes == dataclass]
        variance = np.var(class_data, axis=0)
        variances.append(variance)

    return np.array(variances)

def log_likelihood(x, mean, var):
    # Calculates logaritmic likehood for every test sample
    epsilon=1e-8 # To avoid division by zero
    var = var + epsilon
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

    return log_likelihood

def predict_class(x, means, variances):
    # Returns the class with the highest log likelihood
    num_classes = len(means)
    log_likelihoods = []
    
    for i in range(num_classes):
        log_likelihood_k = log_likelihood(x, means[i], variances[i])
        log_likelihoods.append(log_likelihood_k)
    
    return np.argmax(log_likelihoods)

def test_naive_bayes(x_test, y_test, means, variances):
    #Tests the naive bayes classifier and calculates classification accuracy
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

    means = calculate_means(x_train_noise, y_train)
    variances = calculate_variances(x_train_noise, y_train)

    test_naive_bayes(x_test_flattened, y_test, means, variances)

if __name__ == "__main__":
    main()