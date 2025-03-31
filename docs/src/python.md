# Usage from Python

## Installation

Install `juliacall`, e.g. `pip install juliacall`.
```python
from juliacall import Main as jl
jl.seval('using Pkg; Pkg.add(url = "https://github.com/jbrea/MLPGradientFlow.jl.git")')
```

## Examples
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

```

