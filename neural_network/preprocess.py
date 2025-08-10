def preprocess_data():
    """Imports the MNIST data set from Keras, normalises the data, 
    flattens the 28x28 matrix to a 1D array and one-hot encodes the image labels
    Args:
        
    Returns:
    """
    import numpy as np
    import keras
    from keras.datasets import mnist

    # Load and preprocess the data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Flatten the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # One Hot encode the image labels
    # Code shamelessly borrowed from: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy

    n_values = np.max(y_train) + 1
    y_train_encoded = np.eye(n_values)[y_train]
    n_values = np.max(y_test) + 1
    y_test_encoded = np.eye(n_values)[y_test]

    return X_train, y_train, X_test, y_test, y_train_encoded, y_test_encoded

