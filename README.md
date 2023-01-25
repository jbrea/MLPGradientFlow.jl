# MLPGradientFlow.jl

This package allows to integrate the gradient flow of the loss
function of multi-layer perceptrons,
$$\dot \theta = -\nabla_\theta \big(L(\theta) + R(\theta)\big)$$
with barrier function
$$R(\theta) =  \big(\frac12\|\theta\|_2^2 - c\big)^2 \mbox{if } \frac12\|\theta\|_2^2 > c \mbox{ and 0 otherwise}\, .$$

Activation functions can be e.g. `relu`, `sigmoid` ( $1/(1 + \exp(-x))$ ), `sigmoid2` ( $erf(x/\sqrt{2})$ ), `tanh`, `softplus`, `gelu`, `g` (`g(x) = sigmoid(4x) + softplus(x)`), or `softmax` (in the output layer).

## Installation

### From Julia
```julia
using Pkg; Pkg.add(url = "https://github.com/jbrea/MLPGradientFlow.jl.git")
```

### From Python

Install `juliacall`, e.g. `pip install juliacall`.
```python
from juliacall import Main as jl
jl.seval('using Pkg; Pkg.add(url = "https://github.com/jbrea/MLPGradientFlow.jl.git")')
```

## Usage

### From Julia

```julia
using MLPGradientFlow

inp = randn(2, 10_000)
w_teach = randn(4, 2)
a_teach = randn(1, 4)
target = a_teach * sigmoid.(w_teach * inp)

n = Net(layers = ((5, sigmoid, false), (1, identity, false)),
        input = inp, target = target)

p = random_params(n)

n(p)                         # run network on parameters p
n(p, randn(2, 100))          # run on parameters p and a random dataset

res = train(n, p,
            maxtime_ode = 2, maxtime_optim = 2,          # maxtime in seconds
            maxiterations_optim = 10^3, verbosity = 1)

res["x"]                     # final solution
res["ode_x"]                 # solution after ODE solver

res["loss"]                  # final loss
x = params(res["x"])
loss(n, x)                   # recompute final loss
gradient(n, x)               # compute gradient at solution
hessian(n, x)                # compute hessian at solution
hessian_spectrum(n, x)   # compute spectrum of the hessian at solution


?MLPGradientFlow.train # look at the doc string of train.

# to run multiple initial seeds in parallel
ps = ntuple(_ -> random_params(n), 4)
res = MLPGradientFlow.train(n, ps,
                            maxtime_ode = 2, maxtime_optim = 2,
                            maxiterations_optim = 10^3, verbosity = 1)


# Neural networks without biases and with only a single hidden layer, can also train under the assumption of normally distributed input. For relu and sigmoid2 the implementation uses analytical values for the gaussian integrals (use `f = Val(relu)` for the analytical integration and `f = relu` for the numerical integration). For other activation functions, numerical integration of approximations thereof have to be used.
using LinearAlgebra, ComponentArrays
d = 9
inp = randn(d, 10^5)
w_teach = I(d)[1:d-1,:]
a_teach = ones(d-1)'
target = a_teach * relu.(w_teach * inp)

n = Net(layers = ((2, relu, false), (1, identity, false)),
        input = inp, target = target)
p = random_params(n)

# Train under the assumption of infinite, normally distributed input data
xt = ComponentVector(w1 = w_teach, w2 = a_teach)
ni = NetI(p, xt, Val(relu))
res = train(ni, p, maxiterations_ode = 10^3, maxiterations_optim = 0)

# Recommendations for different activation functions:
# Using analytical integrals:
# - NetI(p, xt, Val(relu))
# - NetI(p, xt, Val(sigmoid2))
# Using approximations:
# - NetI(p, xt, load_potential_approximator(softplus))
# - NetI(p, xt, load_potential_approximator(gelu))
# - NetI(p, xt, load_potential_approximator(g))
# - NetI(p, xt, load_potential_approximator(tanh))
# - NetI(p, xt, load_potential_approximator(sigmoid))

# compare the loss computed with finite data to the loss computed with infinite data
loss(n, params(res["init"])) # finite data
loss(ni, params(res["init"])) # infinite data
loss(n, params(res["x"])) # finite data
loss(ni, params(res["x"])) # infinite data

# For softplus with approximations
approx = load_potential_approximator(softplus)

ni = NetI(p, xt, approx)
res = train(ni, p, maxiterations_ode = 10^3, maxiterations_optim = 0)

# This result can be fine tuned with numerical integration (WARNING: this is slow!!)

ni2 = NetI(p, xt, softplus)
res2 = train(ni2, params(res["x"]), maxtime_ode = 60, maxtime_optim = 60) 

```

### From Python

```python
import numpy as np
import juliacall as jc
from juliacall import Main as jl

jl.seval('using MLPGradientFlow')

mg = jl.MLPGradientFlow

w = np.random.normal(size = (5, 2))/10
b1 = np.zeros(5)
a = np.random.normal(size = (1, 5))/5
b2 = np.zeros(1)
inp = np.random.normal(size = (2, 10_000))

w_teach = np.random.normal(size = (4, 2))
a_teach = np.random.normal(size = (1, 4))
target = np.matmul(a_teach, jl.map(mg.sigmoid, np.matmul(w_teach, inp)))

mg.train._jl_help() # look at the docstring

# continue as in julia (see above), e.g.
p = mg.params((w, b1), (a, b2))

n = mg.Net(layers = ((5, mg.sigmoid, True), (1, jl.identity, True)),
           input = inp, target = target)

res = mg.train(n, p,
               maxtime_ode = 2, maxtime_optim = 2,
               maxiterations_optim = 10**3, verbosity = 1)

# convert the result to a python dictionary with numpy arrays
def convert2py(jldict):
     d = dict(jldict)
     for k, v in jldict.items():
         if isinstance(v, jc.DictValue):
             d[k] = convert2py(v)
         if isinstance(v, jc.ArrayValue):
             d[k] = v.to_numpy()
     return d

py_res = convert2py(res)

# convert parameters in python format back to julia parameters
p = mg.params(jc.convert(jl.Dict, py_res['x']))

# save results in torch.pickle format
mg.pickle("myfilename.pt", res)

mg.hessian_spectrum(n, p)    # look at hessian spectrum

# an MLP with 2 hidden layers with biases in the second hidden layer
n2 = mg.Net(layers = ((5, mg.sigmoid, False), (4, mg.g, True), (2, mg.identity, False)),
            input = inp, target = np.random.normal(size = (2, 10_000)))

p2 = mg.params((w, None),
               (np.random.normal(size = (4, 5)), np.zeros(4)),
               (np.random.normal(size = (2, 4)), None))

mg.loss(n2, p2)
mg.gradient(n2, p2)

# for more examples have a look at the julia code above
```
