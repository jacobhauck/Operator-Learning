# Operator Learning with PyTorch

This repository contains simple, unified implementations of popular
operator learning models, such as

- DeepONet
- Shift-DeepONet
- HyperDeepONet
- Fourier Neural Operator (FNO)
- More in the future!

## Installing
Clone this repository and install with
```commandline
pip install .
```
This repository also requires PyTorch and `mlx`, my machine learning utility
library, which can be found [here](https://github.com/jacobhauck/ML-Template).

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
function object consists of two tensors: the sample values (which we will
denote by $\texttt{y}$) and the sampling points (which we will denote by
$\texttt{x}$). Generally we will allow $\texttt{x}$ to have the shape 
$(\dots, d)$, where $d$ is the dimension of the domain, and $\texttt{y}$ 
to have the shape $(\dots, c)$, where $c$ is the number of output values (or "channels") of the function.
Mathematically, we would say the function
$f : \Omega \to \mathbb{R}^\texttt{c}$, where 
$\Omega \subseteq \mathbb{R}^\texttt{d}$. Assuming that $f$ is applied
pointwise along the last dimension, we could write
$\texttt{y} = f(\texttt{x})$ to indicate that $\texttt{y}$ consists
of the values of $f$ sampled at $\texttt{x}$.

In order to facilitate the use of multiple representations of a function,
we will require that functions support the use of an **interpolator**,
an object that estimates the function value at points other than the
sampling points. If the function is known by an exact formula, then that
formula may be used as an (exact) interpolator. On the other hand, 
nearest-neighbor or linear interpolation (among other possibilities) can
be used to approximate the function value away from the sampling points.

In addition to being the inputs and outputs of operator models, functions
are also the basic elements of datasets (in addition to tensor data, as
usual). Hence, some or all of the variables in an operator learning
dataset will be function-valued. This complicates the usual parallelization
strategy of stacking tensor-valued variables along an extra "batch"
dimension, as the underlying tensor representation of a function, 
$\texttt{x}$ and $\texttt{y}$, may not be the same shape for all
observations in a dataset. We take the following approach to handle this
inconvenience: operator learning models should be implemented for a 
_specific_ representation (without access to the underlying function)
so that they can be applied without modification to batches of 
same-representation functions. The choice of sampling-based representations
(as opposed to, say, Fourier coefficients) enables
this simple approach to allow for batch parallelization.

In a similar way, datasets should keep track of which observations use
the same discrete representation in a way that facilitates sampling
batches with the same representation.
