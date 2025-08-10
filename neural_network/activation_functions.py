import numpy as np

def sigmoid(Z):
    """Sigmoid activation function. Maps input values to a range between 0 and 1.

    Args:
        Z (np.ndarray): Input array or scalar.

    Returns:
        np.ndarray: Output after applying the sigmoid function element-wise.
        """
    return 1/(1 + np.exp(-Z))

def relu(Z):
    """ReLU (Rectified Linear Unit) activation function. Applies the function f(x) = max(0, x) element-wise.

    Args:
        Z (np.ndarray): Input array or scalar.

    Returns:
        np.ndarray: Output after applying ReLU element-wise.
    """
    return np.maximum(0,Z)

def softmax(Z):
    """Numerically stable softmax activation function. Converts logits to probabilities by exponentiating input values
        after subtracting the maximum to improve numerical stability.

        Args:
            Z (np.ndarray): Input array of shape (batch_size, num_classes)
                            or any shape where softmax is applied along the last axis.

        Returns:
            np.ndarray: Probability distribution with the same shape as input,
                        where values sum to 1 along the softmax axis.
        """
    exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)