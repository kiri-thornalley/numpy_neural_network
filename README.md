# Why build a neural network from scratch?
Modern machine learning frameworks like PyTorch and Tensorflow are incredibly powerful - but they abstract away almost all of the core mathematical operations. While this is ideal for productivity, since neural networks can be built in a few lines of code, it can lead to a shallow understanding of what's actually happening under the hood. 

I find I need to take complex ideas apart into their smallest building blocks and rebuild them myself before I truly understand why they work, and can confidently apply that knowledge to new problems. This project is that process: my attempt to lift the hood on neural networks by implementing one from scratch using only NumPy. The goal is simple: if I can't build it myself, then I don't fully understand it. 

In this work, I specifically want to:
- Reinforce my understanding of forward and backward propagation by explicitly coding each step
- Build and debug a training loop without relying on automatic differentiation (autograd) or other black-box tools
- Compare performance and results with an identical neural network implemented in PyTorch 

By doing this, I aim to bridge the gap between _using_ neural neotworks and _understanding_ them - turning magic into mechanics. 

# Philosopy of the project
This project isn't just about building a neural network. It's about building understanding - line by line, layer by layer. Rather than rushing to a .py script with abstracted logic, I'm intentionally using a literate programming approach to emphasise clarity and reasoning over brevity. To help with this, within the notebook there are a number of blocks with the heading: "📚 Side Quest:" These are deliberate deep dives. Sometimes they ask "Why does this work?", other times they explore alternatives, visualisations, or detours that help cement understanding. They’re not required to follow the code, but they document the thought process, debugging, and explorations that went into building the model.

If you want the just the code version, that’s in the src/ folder. But if you want to understand how a neural network really works, follow the notebook(s) — side quests and all.

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


