```@meta
CollapsedDocStrings = false
ShareDefaultModule = true
```



MLPGradientFlow.jl allows to investigate the loss landscape and training dynamics of multi-layer perceptrons.

## Features

- Train multi-layer perceptrons on the CPU to convergence, using first and second order optimization methods.
- Fast implementations of gradients and hessians.
- Follow gradient flow (using differential equation solvers) or popular (stochastic) gradient descent dynamics (Adam etc.).
- Accurate approximations of loss function and its derivatives for infinite normally distributed input data, using Gauss-Hermite quadrature or symbolic integrals.
- Utility functions to investigate teacher-student setups and loss landscape visualization.
