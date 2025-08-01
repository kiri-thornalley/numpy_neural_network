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

## Why literate programming? 
Literate programming, coined by Donld Knuth, is the idea that "code should be written for humans first, and computers second". By interweaving explanations and mathematics with implementation, I'm treating this neural network not just as code, but as a form of interactive scientific reasoning. 

This method:
- Forces clarity of thought
- Encourages a narrative structure, and therefore makes the project transparent, reproducible and educational for my future self (and possibly others too)
- Allows me to debug concepts before debugging code

> Will you stop taking everything to pieces!

I work best when I'm able to deconstruct and reconstruct systems and ideas, and literate programming mirrors that. It lets me pull apart ideas, walk through them and then put them back together. It also slows me down enough to notice what I don't understand, instead of skipping to working code that I can’t explain.

### Why not go straight to .py files?
Uing .py scripts from the beginning encourages premature abstraction. It's too easy to bury complexity inside polished functions, without truly understanding what's going on. In contrast, notebooks allow me to easily see and inspect intermediate steps; support experimentation and introspection as I go, and align more closely with the spirit of mathematical exploration than pure software engineering. 

Towards the end of this project, core components will be modularised into .py files in order to build the final neural network, but only *after* I've made sense of each part in isolation. The .py files are then clean implementations of the code, but the notebook outlines how and why I arrived there.

# Architecture
Outline the architecture implemented manually. For MNIST, a common baseline is:
Input layer (784 units) - one for each pixel in the 28x28 image
→ Hidden layer (32 units) with ReLU
→ Output layer (10 units) with softmax

In this section, explain:
- The dimensions of your weight matrices and why they’re shaped that way
- How the activation functions are implemented from scratch (e.g. ReLU, softmax)
- Your manual forward and backward passes, gradient calculations, and updates (SGD)
- Any improvements or optional features you added (e.g., cross-entropy loss, momentum, batching, etc.)

## 🔍 Design Choices and Learnings
- Why I used ReLU instead of Sigmoid
What happens if we use Sigmoid? - Can we quantify exactly how much slower it is to train?

- Why I chose softmax + cross-entropy instead of MSE
- What surprised me when implementing backprop manually
- What I learned from comparing SGD and Adam

# Results
This section is about transparency. Present:
- The final test accuracy (e.g. ~92% on MNIST is a strong result for a 1-hidden-layer net from scratch)
- How many epochs it took
- Learning curves (loss vs epoch, accuracy vs epoch)
- Where the model struggled (e.g. confusion matrix showing misclassified digits)
- Any visualizations of weights or hidden activations you found interesting

- Also include limitations:
This implementation does not use GPU acceleration -or advanced optimization techniques like Adam-. It’s designed for interpretability, not performance.

# Comparison with Pytorch
Now replicate the same architecture using PyTorch (or Keras, optionally), and compare:

- Accuracy: Should be roughly the same.
- Training speed: PyTorch will obviously be faster — say why. (PyTorch is like a f1 engine installed into a gokart. The engineering has been refined until it cannot be refined any more. I have built a gokart in numpy. It is slower, but it is _my_ gokart)
- Code size: Show how few lines it takes in PyTorch and reflect on the abstraction layers.
- Abstractions: Highlight what PyTorch gives you “for free” (autograd, module structure, optimizers, GPU support)

I'll almost certainly never build a neural network from scratch like this again. But having done it once, I now understand _why_ PyTorch/Tensorflow are built the way they are, why certain functiond are called and why we call them in this order. The high-level APIs no longer feel like magic; they're like lego bricks I now know the shape of, purpose of, and the maths underneath. The purpose of this project was never about building a lightning fast neural network from scratch, but about understanding the why, the structure of a neural network. Once I have that, I can build more confidently with the real tools. 
