import numpy as np
import torch
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev
import torch.nn as nn
from juliacall import Main as jl
import timeit
import pandas as pd
torch.set_num_threads(1)

def get_mlp(Din, layers, f = nn.Softplus(), dtype = torch.float64):
    layers = list(layers)
    layers.insert(0, Din)
    mlp = nn.Sequential()
    for i in range(len(layers)-2):
        mlp.append(nn.Linear(layers[i], layers[i+1], dtype = dtype))
        mlp.append(f)
    mlp.append(nn.Linear(layers[-2], layers[-1], dtype = dtype))
    return mlp

def random_params(Din, layers, Dout = 1, dtype = torch.float64):
    p = []
    layers = list(layers)
    layers.insert(0, Din)
    for i in range(len(layers)-1):
        p.append((torch.randn(layers[i], layers[i+1], dtype = dtype),
                  torch.randn(Dout, layers[i+1], dtype = dtype)))
    return p

def setup(Din = 8, Dout = 1, Nsamples = 10**3, k = 4,
          ft = nn.Softplus(), dtype = torch.float64):
    inp = torch.randn(Nsamples, Din, dtype = dtype)
    ((w1, b1), (w2, b2)) = random_params(Din, (k, Dout), dtype = dtype)
    targ = b2 + torch.matmul(ft(b1 + torch.matmul(inp, w1)), w2)
    return inp, targ

def loss(params, inp, targ, f):
    x = inp
    for (w, b) in params:
        x = f(b + torch.matmul(x, w))
    return nn.MSELoss()(x, targ)

def test_grad(mlp, inp, targ, N = 10**3, lossfunc = nn.MSELoss()):
    for _ in range(N):
        mlp.zero_grad()
        y = mlp(inp)
        l = lossfunc(y, targ)
        l.backward()

def test_hess(params, inp, targ, f = nn.Softplus(), N = 10**2):
    if len(params) == 2:
        params = (params[0][0], params[0][1], params[1][0], params[1][1])
        l = lambda w1, b1, w2, b2: loss(((w1, b1), (w2, b2)), inp, targ, f)
    elif len(params) == 3:
        params = (params[0][0], params[0][1],
                  params[1][0], params[1][1],
                  params[2][0], params[2][1])
        l = lambda w1, b1, w2, b2, w3, b3: loss(((w1, b1),
                                                 (w2, b2),
                                                 (w3, b3)), inp, targ, f)
    for _ in range(N):
        H = torch.autograd.functional.hessian(l, params)
    return H

jl.seval('using Pkg; Pkg.activate("."); using MLPGradientFlow')
jl.seval('''
function test_grad(net, x; N = 10^3)
    dx = zero(x)
    for _ in 1:N
        MLPGradientFlow.gradient!(dx, net, x)
    end
end
''')
jl.seval('''
function test_hess(net, x; N = 10^2)
    H = MLPGradientFlow.Hessian(x)
    for _ in 1:N
        MLPGradientFlow.hessian!(H, net, x)
    end
end
''')
mg = jl.MLPGradientFlow

def t2jl(x):
    return np.array(x.t())

def jlfunc(f):
    if isinstance(f, nn.Softplus):
        return jl.softplus
    elif isinstance(f, nn.Sigmoid):
        return jl.sigmoid
    elif isinstance(f, nn.Tanh):
        return jl.tanh
    elif isinstance(f, nn.ReLU):
        return jl.relu

def compare(Din, N, layers, f, repeat = 10, number = 1, Ngrad = 10**3, Nhess = 10**2):
    print(N, layers, Ngrad, Nhess)
    inp, targ = setup(Din = Din, Nsamples = N)
    mlp = get_mlp(Din, layers, f = f)
    p = random_params(Din, layers)
    def test_grad_t():
        return test_grad(mlp, inp, targ, N = Ngrad)
    def test_hess_t():
        return test_hess(p, inp, targ, f = f, N = Nhess)
    tgt = timeit.repeat(test_grad_t, number = number, repeat = repeat)
    tht = timeit.repeat(test_hess_t, number = number, repeat = repeat)
    layersj = []
    for l in layers:
        layersj.append((l, jlfunc(f), True))
    net = mg.Net(layers = tuple(layersj),
                 input = t2jl(inp), target = t2jl(targ), derivs = 2, verbosity = 0)
    x = mg.random_params(net)
    def test_grad_j():
        return jl.test_grad(net, x, N = Ngrad)
    def test_hess_j():
        return jl.test_hess(net, x, N = Nhess)
    # warmup (compilation of julia code)
    test_grad_j()
    test_hess_j()
    tgj = timeit.repeat(test_grad_j, number = number, repeat = repeat)
    thj = timeit.repeat(test_hess_j, number = number, repeat = repeat)
    return pd.DataFrame({'grad_torch' : tgt,
                         'hess_torch' : tht,
                         'grad_jl' : tgj,
                         'hess_jl' : thj,
                         'Din' : Din,
                         'N' : N,
                         'Nparams': len(x),
                         'layers' : str(layers)}
                        )

Ns = [10, 10**2, 10**3, 10**4, 10**5]
ks = [10, 50, 100]
tmp = pd.concat([compare(2, n, (k, 1), nn.Softplus(),
                          repeat = 10,
                          Ngrad = int(np.sqrt(10**10/(n*k)/3)),
                          Nhess = int(np.sqrt(10**8/(n*k))/3))
                 for k in ks for n in Ns])

tmp.to_csv("efficiency.csv")
