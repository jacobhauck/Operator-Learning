# Operator Learning with PyTorch

This repository contains simple, unified implementations of popular
operator learning models, such as

- DeepONet
- Fourier Neural Operator (FNO)
- More in the future!

## What is Operator Learning?

## Important Implementation Concepts

Here we outline the important concepts underlying our implementations
of operator learning models, which are crucial for increasing
simplicity, unity and flexibility.

### Functions

In operator learning the primary data objects are *functions*; this
is in contrast to machine learning, where the data consists of tensors.
As such, the input to an operator learning model should, in general, be
a set of functions rather than tensors. In practice, however, functions
are handled using discrete, computational representations. For the
purposes of this project, we will always assume that a function is
represented by its values sampled on a collection of points. Thus, a
function object consists of two tensors