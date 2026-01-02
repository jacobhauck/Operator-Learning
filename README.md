# Operator Learning with PyTorch

This repository contains simple, unified implementations of popular
operator learning models, such as

- DeepONet
- PCA-Net
- Mulftifidelity Encode-Approximate-Reconstruct (MFEAR) 
- Shift-DeepONet
- HyperDeepONet
- Two-Step DeepONet
- Fourier Neural Operator (FNO)
- General Neural Operator Transformer (GNOT)
- More in the future!

## Installing
Clone this repository and install with
```commandline
pip install .
```
This repository also requires PyTorch and `mlx`, my machine learning utility
library, which can be found [here](https://github.com/jacobhauck/ML-Template).

## What is Operator Learning?

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

## Demo Experiments

Operator learning demos are provided for all the models implemented. We
use the same problem for each demo, which is a simple 2D Poisson problem.
Let $v(x,7)$ satisfy\
$$\Delta v = u(x,y) \qquad (x,y) \in [0,10]^2,$$\
$$v(x,y) = 0, \qquad (x,y)\in \partial([0,10]^2),$$\
so that $v$ is a solution of the Poisson equation with source $u$ and homogeneous
Dirichlet boundary conditions. Let $u$ be a random function defined by\
$$u(x) = \sum_{k=-6}^6\sum_{\ell=-6}^6 \frac{3a_{k\ell}}{1+k^2+\ell^2}\sin\left(\frac{2\pi}{10} (kx + \ell y)\right),$$\
where $a_{k\ell} \sim N(0,1)$ are i.i.d. standard normal random variables.

We generate a dataset $\{(u_i, v_i)\}_{i=1}^{2000}$ of 2000 source--solution pairs
$(u_i, v_i)$, with $u_i$ drawn i.i.d. from the distribution of $u$ above. The
goal in each of our operator learning demos is to approximate the operator mapping
$u \mapsto v$ taking the source function $u$ to the solution $v$. We do this
by minimizing the relative $L^2$ loss, defined by\
$$\mathcal{L}(\theta) = \frac{1}{2000}\sum_{i=1}^{2000} \frac{\|G_\theta(u_i) - v_i\|_{L^2}^2}{\|v_i\|_{L^2}^2},$$\
where the $L^2$ norm $\|f\|_{L^2}$ is defined by\
$$\|f\|_{L^2}^2 = \int_0^{10}\int_0^{10} |f(x,y)|^2\;\text{d}x\;\text{d}y.$$\
All demo experiments report results using relative $L^2$ loss.

### Running the demos

The demo experiments are implemented using my machine learning experiment library,
[mlx](https://github.com/jacobhauck/ML-Template). Thus, to run the main demo for,
say, DeepONet, you simply use the command
```commandline
python -m mlx.run deeponet poisson
```
from the project directory. This runs the `deeponet` experiment found in the
`experiments` directory using the global configuration options defined in
`experiments/poisson.yaml`. See [mlx](https://github.com/jacobhauck/ML-Template)
for more information on how to create and run experiments.

### Other demos

In addition to the operator learning demos, there are also some debug demos
showing how some of the helper functions, like dataset generators, work. These
include:
- `fourier_feature_test` tests Fourier feature expansion modules used by several
  models
- `poisson_generate` tests Poisson dataset generation
- `mfear/test_integrator` tests the MFEAR numerical integration
- `pcanet/visualize_bases` tests the PCA implementation in PCA-Net
