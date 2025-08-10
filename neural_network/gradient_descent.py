def gradient_descent(X, Y, params, epochs, alpha):
  """ Undertakes the full training loop
      args:
        X: (np.ndarray) 784x1 arrays representing the image data
        Y: (np.ndarray) image labels
        params: (dict) weights and biases of the network
        epochs: (int) number of training epochs
        alpha: (float) learning rate
      returns:
        output: (np.ndarray) output ['A2'] of the network
        params: (np.ndarray)weights and biases of the trained network
        losses: (list) loss of the network per training epoch
        accuracies: (list) accuracy of the network per training epoch"""

  # Lists to store loss and accuracy per epoch
  losses = []
  accuracies = []

  for i in range(epochs):
      cache_values, output = forward_prop(params, X)
      network_error = cross_entropy_loss_grad(cache_values, Y)
      gradients = backward_prop(Y, network_error, cache_values, params)
      params = update_params(params, gradients, alpha)

      predictions = np.argmax(output, axis=0)
      true_labels = np.argmax(Y, axis=1)
      accuracy = np.mean(predictions == true_labels) * 100
      accuracies.append(accuracy)
      loss = cross_entropy_loss(cache_values,Y)
      losses.append(loss)

      if i % 10 == 0 or i==149:
          print(f'After {i+1} epochs - Accuracy: {accuracy:.2f}% | Loss: {loss:.2f}')
  return output, params, losses, accuracies