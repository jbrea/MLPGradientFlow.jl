using Pkg
Pkg.activate(joinpath(@__DIR__))
using MLPGradientFlow, ComponentArrays, Random, OrdinaryDiffEq, Statistics, DataFrames
using PGFPlotsX, ColorSchemes, Serialization
import MLPGradientFlow: params
import PGFPlotsX: Axis
colors = ColorSchemes.Johnson

function setup(; Din = 2, Nsamples = 10^4,
                 seed = 123, rng = Xoshiro(seed),
                 parameter_rng = rng,
                 k = 4, r = 8, f = softplus, biases = true)
    inp = randn(rng, Din, Nsamples)
    xt = ComponentArray(w1 = randn(parameter_rng, k, Din + biases),
                        w2 = randn(parameter_rng, 1, k + biases))
    h = xt.w1[:, 1:Din] * inp
    if biases
        h .+= xt.w1[:, end]
    end
    targ = xt.w2[:, 1:k] * f.(h)
    if biases
        targ .+= xt.w2[:, end]
    end
    net = Net(layers = ((r, f, biases), (1, identity, biases)),
              input = inp, target = targ, derivs = 2)
    x = random_params(parameter_rng, net)
    return net, x, xt
end
