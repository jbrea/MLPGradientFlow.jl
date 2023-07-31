using Pkg
Pkg.activate(joinpath(@__DIR__))
using MLPGradientFlow, ComponentArrays, Random, OrdinaryDiffEq, Statistics, DataFrames, LinearAlgebra
using PGFPlotsX, ColorSchemes, Serialization
import MLPGradientFlow: params
import PGFPlotsX: Axis
colors = ColorSchemes.Johnson

###
### Utils
###

standard_normal_input(; rng, Din, Nsamples, kwargs...) = randn(rng, Din, Nsamples)
function split(p::ComponentArray, i, μ)
    w2 = hcat(p.w2, p.w2[i] * (1-μ))
    w2[i] *= μ
    ComponentArray(w1 = vcat(p.w1, p.w1[i, :]'), w2 = w2)
end

###
### Teachers
###

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
    inp, f.(eachcol(inp))', missing
end


###
### Setup
###

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

function perturbed_saddle(smallnet, localmin; i, μ, σ = 1e-2, kwargs...)
    dir = split(localmin, i, 0) - split(localmin, i, 1)
    p = split(localmin, i, μ)
    l = smallnet.layerspec
    net = Net(layers = ((l[1][1]+1, l[1][2], l[1][3]), l[2]),
              bias_adapt_input = false,
              input = smallnet.input, target = smallnet.target, derivs = 2)
    H = hessian(net, p)
    eig = eigen(Symmetric(H))
    eps = (dir .== 0) .* randn(length(dir))*σ
    x = p .+ eps
    alignment = eig.vectors[:, 1]' * eps
    return net, x, alignment
end

function netfromres(res)
    lspec = map(x -> (x[1], eval(Meta.parse(x[2])), x[3]), res["layerspec"])
    Net(layers = lspec,
        bias_adapt_input = false,
        input = res["input"],
        target = res["target"],
        derivs = 2),
    params(res["init"])
end
