import numpy as np
from activation_functions import relu, softmax
def init_params():
    """Initialises the weights and biases for the network as arrays of random floats between -0.5 and 0.5

    Args:
        none

    Returns:
        W1: (np.ndarray) an array of shape 32x784, representing the initial weights of the first layer.
        B1: (np.ndarray) an array of shape 32x1, representing the initial biases of the connections of the first layer.
        W2: (np.ndarray) an array of shape 10x32, representing the initial weights of the second layer of the network.
        B2: (np.ndarray) an array of shape 10x1, representing the initial biases of the connections of the second layer.

    """
    # For reproducibility
    np.random.seed(42)

    params = {}

    params['W1'] = np.random.rand(32, 784)-0.5
    params['B1'] = np.random.rand(32, 1)-0.5
    params['W2'] = np.random.rand(10, 32)-0.5
    params['B2'] = np.random.rand(10, 1)-0.5

    return params

def forward_prop(params, X):
    """ Performs forward propagation through the network.

    Args:
    X: (np.ndarray) an array of shape 728xm representing the input data (train or test set)
    params: (np.ndarray) a series of arrays representing the weighs and biases of the network

    Returns:
    cache_values: (dict) a dictionary of arrays holding the inputs for each node pre and post activation
    A2: (np.ndarray) Output of the network

    """
    cache_values = {}

    # Input layer has no activation function. The input is the output.
    A0 = X.T
    cache_values['A0'] = A0

    # First hidden layer

    Z1 = np.dot(params['W1'], A0)+ params['B1']

    # Apply activation function
    A1 = relu(Z1)

    # Store values for backprop later
    cache_values['Z1'] = Z1
    cache_values['A1'] = A1

    # Output layer - get params
    W2 = params['W2']
    B2 = params['B2']

    Z2 = np.dot(W2, A1)+B2

    # Apply activation function
    A2 = softmax(Z2)

    # Store output layer values
    cache_values['Z2'] = Z2
    cache_values['A2'] = A2

    return cache_values, A2