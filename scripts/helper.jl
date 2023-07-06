using Pkg
Pkg.activate(joinpath(@__DIR__))
using MLPGradientFlow, ComponentArrays, Random, OrdinaryDiffEq, Statistics, DataFrames
using PGFPlotsX, ColorSchemes, Serialization
import MLPGradientFlow: params
import PGFPlotsX: Axis
colors = ColorSchemes.Johnson

function random_teacher(; input, Nsamples, rng, parameter_rng, k, Din, biases, f, kwargs...)
    xt = ComponentArray(w1 = randn(parameter_rng, k, Din + biases),
                        w2 = randn(parameter_rng, 1, k + biases))
    inp = input(; rng, Din, Nsamples)
    h = xt.w1[:, 1:Din] * inp
    if biases
        h .+= xt.w1[:, end]
    end
    targ = xt.w2[:, 1:k] * f.(h)
    if biases
        targ .+= xt.w2[:, end]
    end
    inp, targ, xt
end
function aifeynman_11(; Nsamples, kwargs...)
    inp = rand(3, Nsamples)*4 .+ 1
    f = x -> begin
        q1, epsilon, r = x
        q1*r/(4*pi*epsilon*r^3)
    end
    inp, f.(eachcol(inp)), missing
end
standard_normal_input(; rng, Din, Nsamples, kwargs...) = randn(rng, Din, Nsamples)
function setup(; Din = 2, Nsamples = 10^4,
                 seed = 123, rng = Xoshiro(seed),
                 parameter_rng = rng,
                 teacher = random_teacher,
                 input = standard_normal_input,
                 k = 4, r = 8, f = softplus, biases = true)
    inp, targ, xt = teacher(; input, Din, Nsamples, rng, parameter_rng, k, r, f, biases)
    net = Net(layers = ((r, f, biases), (1, identity, biases)),
              input = inp, target = targ, derivs = 2)
    x = random_params(parameter_rng, net)
    return net, x, xt
end
