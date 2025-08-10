# Why build a neural network from scratch?
Modern machine learning frameworks like PyTorch and Tensorflow are incredibly powerful - but they abstract away almost all of the core mathematical operations. While this is ideal for productivity, since neural networks can be built in a few lines of code, it can lead to a shallow understanding of what's actually happening under the hood. 

I find I need to take complex ideas apart into their smallest building blocks and rebuild them myself before I truly understand why they work, and can confidently apply that knowledge to new problems. This project is that process: my attempt to lift the hood on neural networks by implementing one from scratch using only NumPy. The goal is simple: if I can't build it myself, then I don't fully understand it. 

In this work, I specifically want to:
- Reinforce my understanding of forward and backward propagation by explicitly coding each step
- Build and debug a training loop without relying on automatic differentiation (autograd) or other black-box tools
- Compare performance and results with an identical neural network implemented in PyTorch 

By doing this, I aim to bridge the gap between _using_ neural neotworks and _understanding_ them - turning magic into mechanics. 

# Philosopy of the project
This project isn't just about building a neural network. It's about building understanding - line by line, layer by layer. Rather than rushing to a .py script with abstracted logic, I'm intentionally using a literate programming approach to emphasise clarity and reasoning over brevity. 

To help with this, within the notebook there are a number of blocks with the heading: "📚 Side Quest:" These are deliberate deep dives. Sometimes they ask "Why does this work?", other times they explore alternatives, visualisations, or detours that help cement understanding. They’re not required to follow the code, but they document the thought process, debugging, and explorations that went into building the model.

If you want the just the code version, that’s in the neural_network folder. But if you want to understand how a neural network really works, follow the notebook — side quests and all.

## Why literate programming? 
Literate programming, coined by Donld Knuth, is the idea that "code should be written for humans first, and computers second". By interweaving explanations and mathematics with implementation, I'm treating this neural network not just as code, but as a form of interactive scientific reasoning. 

This method:
- Forces clarity of thought
- Encourages a narrative structure, and therefore makes the project transparent, reproducible and educational for my future self (and possibly others too)
- Allows me to debug concepts before debugging code

I work best when I'm able to deconstruct and reconstruct systems and ideas, and literate programming mirrors that. It lets me pull apart ideas, walk through them and then put them back together. It also slows me down enough to notice what I don't understand, instead of skipping to working code that I can’t explain.

### Why not go straight to .py files?
Uing .py scripts from the beginning encourages premature abstraction. It's too easy to bury complexity inside polished functions, without truly understanding what's going on. In contrast, notebooks allow me to easily see and inspect intermediate steps; support experimentation and introspection as I go, and align more closely with the spirit of mathematical exploration than pure software engineering. 

Towards the end of this project, core components will be modularised into .py files in order to build the final neural network, but only *after* I've made sense of each part in isolation. The .py files are then clean implementations of the code, but the notebook outlines how and why I arrived there.

# Architecture
In this project, we implement the following architecture manually in numpy. The network has an input layer of 784 units - a node for each pixel in the 28x28 image, a single hidden layer of 32 units (because 32 is a nice number that we can easily display if we want to visualise the activations of this layer) using ReLu as the activation function, and an output layer of 10 units with (safe) softmax to turn the logits into probabilities of how certain the network is about the predicted label for the image.

## Weight Matrices and Their Dimension
The network has 784 input nodes (corresponding to the 28x28 rray of input pixels), 32 hidden nodes (because 32 was a nice number to plot) and 10 output nodes (to match the number of output classes).

- W1: shape (32,784) - each hidden node has a weight for each input pixel
- B1: shape (32,1) - one bias per hidden node
- W2: shape (10,32) - each output node has a weight for each hidden node
- B2: shape (10,1) - one bias per output node. 

The (rows, cols) convention here is (number of units in the current layer, number of inputs from the previous layer). This shape makes the matrix multiplication Z = W·X + b work cleanly.

## Activation Functions from Scratch
Implemented three activation functions manually in NumPy. These are ReLU (Rectified Linear Unit) np.maximum(0,x) - if value is negative, return zero, else return the input value. 
Sigmoid: 1/(1+np.exp(-x)) - squashes values to between 0 and 1, whic is useful or probabilities, but prone to overstauration.

Safe softmax - uses the subtrax max trick to prevent exponents from exploding. 

## Manual Forward and Backward Pass
**Forward Pass:**
- Compute Z1 = W1·X + b1
- Apply activation (A1 = ReLU(Z1))
- Compute Z2 = W2·A1 + b2
- Apply softmax (Z2) to get class probabilities

**Backward Pass:**
- Compute output layer error: dZ2 = A2 - Y
- Calculate dW2, db2 via matrix multiplication and averaging
- Propagate error back through ReLU (dA1, dZ1)
- Calculate dW1, db1 for the first layer

It’s vanilla stochastic gradient descent (SGD), no fancy optimisers here. The NumPy version is basically a gokart engine: it runs, but it’s me who’s pedalling.

## Improvements and Optional Features
**Cross-Entropy Loss:** Pairs perfectly with softmax for multi-class problems and makes the maths cleaner.

**Safe Softmax:** Added for numerical stability. No exploding exponents here, thank you very much.

**Full-batch Gradient Descent:** No mini-batching in NumPy version (to keep it simple), but batching could easily be added.

**Optuna Hyperparameter Tuning:** Finds the best value for the learning rate without me just guessing.

## Design Choices and Learnings
### Why I used ReLU instead of Sigmoid
ReLU is known to be much easier to train, less prone to overstaturation and also trains much faster than Sigmoid. 

### Softmax and Cross-Entropy Loss over MSE
Softmax outputs a probability distribution, and cross entropy loss measures how far our predicted distribution is from the true answer. These two ideas are mathematically linked and therefore work well together. 

### What surprised me when implementing backprop manually
Mathematically, backpropagation is not that hard, it's just the chain rule on repeat, and I've survived harder (think statistical thermodynamics - there's a reason it's a compulsory module at the University of Sheffield). What's difficult is keeping track of all of the shapes, one transpose where it shouldn't be and it's error messages all the way down. 

For a small two-layer network like this, it's easy enough the debug (eventually) and fix this, but deeper architectures? I'd rather jump headfirst into a vat of piranahas than debug backprop manually. 

# Results
Hyperparameter tuning using optima improved accuracy on the training set from 10.34% to 89.4% in 100 epochs. Following training of 150 epochs, we reach an accuracy of 89.51% on the training set and 89.23% on the test set. 

Training curves:
- Loss vs Epoch: Clear downward trend, dropping rapidly at first and loss eventually slowing
- Accuracy vs Epoch: Rapid improvement in the first 20 epochs. Beyond epoch 80, limited improvement in loss.
<img width="541" height="500" alt="download" src="https://github.com/user-attachments/assets/b3666d30-d684-490e-a493-07bf7256a373" />

Confusion Matrix: 
- Model performs better on some digits than others, performs strongly on the digits that are visually distinct e.g., 1. 
- Is more prone to misclassification where two or more digits look similar - 8 and 3.


<img width="541" height="556" alt="download" src="https://github.com/user-attachments/assets/0819e34d-0a28-4929-bc3c-7b9634eb6821" />

**Limitations of the model**

This implementation does not use GPU acceleration or advanced optimization techniques like Adam. It’s designed for interpretability, not performance.

## Comparison with PyTorch
**Accuracy:**
Our NumPy implementation can sometimes achieve slightly higher accuracy, but this requires carefully aligning PyTorch’s settings—such as disabling mini-batching and shuffling—to match the NumPy training loop exactly. When using PyTorch or Keras with their default configurations, however, these frameworks generally reach higher accuracy much faster with fewer training epochs. For example, using Keras on a PyTorch backend, we observed that mini-batching significantly improves model accuracy and reduces the number of epochs required to achieve reasonable performance.

**Training speed:**
Training with PyTorch is much faster than manual implementations. PyTorch is like a Formula 1 engine installed in a go-kart — extensively optimized and refined by many brilliant engineers. My NumPy implementation is more like the go-kart’s original engine: slower, but uniquely my own.

**Code size and abstraction:**
The abstraction level in Keras is such that the same neural network architecture can be implemented in roughly 50 lines of code, compared to approximately 200 lines in NumPy (not counting exploratory side quests and debugging). Additionally, PyTorch and Keras provide many features “for free,” such as advanced optimizers like Adam, automatic gradient calculation for backpropagation, and GPU support — all of which speed up the development process.

# Reflections
I will almost certainly never build a neural network from scratch like this again. But having done it once, I now understand why PyTorch and TensorFlow are designed the way they are: why specific functions exist, and why we call them in the order we do. These high-level APIs no longer feel like magic; they’re more like LEGO bricks — I now know their shape, purpose, and the math behind them. This project was never about building the fastest neural network from scratch, but about truly understanding why and how neural networks work. With this foundation, I can confidently use the powerful tools available and explore more advanced developments in neural networks.
