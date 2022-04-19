# Numerical Differentiation Leveraging Convolution (ndc)

<img alt="logo" align="left" height="170px" src="logo.png" style="padding-right: 20px">

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/famura/ndc/branch/main/graph/badge.svg?token=ESUTNFwtYY)](https://codecov.io/gh/famura/ndc)
[![isort](https://img.shields.io/badge/imports-isort-green)](https://pycqa.github.io/isort/)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**What for?**

Differentiate signals stored as PyTorch tensors, e.g. measurements obtained from a device or simulation, where automatic differentiation can not be applied.

**Features**

* Theoretically **any order, any stencils, and any step size** (see [this Wiki page](https://en.wikipedia.org/wiki/Finite_difference_coefficient) for information). Be aware that there are numerical limits when computing the filter kernel's coefficients, e.g. small step sized and high orders lead to numerical issues.
* Works for **multidimensional signals**, assuming that all dimensions share the same step size.
* Computations can be executed on **CUDA**. However, this has not been tested extensively.
* Straightforward implementation which you can easily adapt to your needs.

**How?**

The idea of this small repository is to use the duality between convolution, i.e., filtering, and [numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation) to leverage the existing functions for 1-dimensional convolution in order to compute the (time) derivatives.

**Why PyTorch?**

More often then not I received (recorded) simulation data as PyTorch tensors rather than numpy arrays.
Thus, I think it is nice to have a function to differentiate measurement signals without switching the data type or computation device.
Moreover, the `torch.conv1d` function fits perfectly for this purpose.


## Citing

If you use code or ideas from this repository for your projects or research, please cite it.
```
@misc{Muratore_ncd,
  author = {Fabio Muratore},
  title = {ndc - Numerical differentiation leveraging convolutions},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/famura/ndc}}
}
```

## Installation

To install the core part of the package run
```
pip install ndc
```

For (local) development install the dependencies with
```
pip install -e .[dev]
```

## Usage

Consider a signal `x`, e.g. a measurement you obtained form a device. This package assumes that the signal to differentiate is of shape `(num_steps, dim_data)`

```python
import torch
import ndc

# Assuming you got x(t) from somewhere.
assert isinstance(x, torch.Tensor)
num_steps, dim_data = x.shape 

# Specify the derivative. Here, the first order central derivative.
stencils = [-1, 0, 1]
order = 1
step_size = dt # should be known from your signal x(t), else use 1
padding = True # if true, the initial and final values are repeated as often as necessary to match the  length of x 

dx_dt_num = ndc.differentiate_numerically(x, stencils, order, step_size, padding)
assert dx_dt_num.device == x.device
if padding:
    assert dx_dt_num.shape == (num_steps, dim_data)
else:
    assert dx_dt_num.shape == (num_steps - sum(s != 0 for s in stencils), dim_data)
```


## Contributions

Maybe you want another padding mode, or you found a way to improve the CUDA support. Please feel free to leave a pull request or issue.
