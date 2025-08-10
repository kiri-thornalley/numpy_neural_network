def deriv_relu(X):
    """ Derivative of the ReLU function """
    return X > 0

# Error Calculation
def cross_entropy_loss(cache_values, Y):
    """
    Compute the cross-entropy loss for softmax output.

    Args:
        cache_values:(dict) Dictionary containing forward pass values, including 'A2'.
            'A2' should have shape (n_classes, n_samples) — softmax probabilities.
        Y: (ndarray)One-hot encoded true labels, shape (n_samples, n_classes).

    Returns:
        float: Mean cross-entropy loss over all samples.
    """
    m = Y.shape[0]  # number of samples
    probs = cache_values['A2'].T  # shape (n_samples, n_classes)
    loss = -np.sum(Y * np.log(probs + 1e-15)) / m
    return loss

def cross_entropy_loss_grad(cache_values, Y):
    """Compute the gradient of the cross-entropy loss w.r.t. the logits for softmax output."""
    network_error = cache_values['A2'] - Y.T
    return network_error

# Calculate gradients through backpropagation
def backward_prop(Y, network_error, cache_values, params):
    """ Performs backward propagation through the network
    Args:
      Y: True labels
      network_error: (np.ndarray) Error between predicted and true values
      cache_values: (np.ndarray) Values of each node pre- and post activation
      params: Weights and biases of the network

    Returns:
      gradients:


    """
    m = Y.shape[1]
    A0 = cache_values['A0']
    A1 = cache_values['A1']
    Z1 = cache_values['Z1']

    gradients= {}

    # Second layer (output)
    dZ2 = cross_entropy_loss_grad(cache_values, Y)
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    dB2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # Store values
    gradients['dZ2'] = dZ2
    gradients['dW2'] = dW2
    gradients['dB2'] = dB2

    # Hidden layer
    dZ1 = np.dot(params['W2'].T, dZ2) * deriv_relu(Z1)
    dW1 = (1 / m) * np.dot(dZ1, A0.T)
    dB1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    # Store values
    gradients['dZ1'] = dZ1
    gradients['dW1'] = dW1
    gradients['dB1'] = dB1
    return gradients

# Update weights and biases
def update_params(params, gradients, alpha):
    """Updates the weights and biases of the network using gradients and the learning rate.
        Args:
          params: weights and biases of the network
          gradients: gradients as determined by backwards propagation
          alpha: (float) learning rate
        Returns:
          params: updated weights and biases of the network
        """

    params['W1'] = params['W1'] - (alpha * gradients['dW1'])
    params['B1'] = params['B1'] - (alpha * gradients['dB1'])
    params['W2'] = params['W2'] - (alpha * gradients['dW2'])
    params['B2'] = params['B2'] - (alpha * gradients['dB2'])
    return params