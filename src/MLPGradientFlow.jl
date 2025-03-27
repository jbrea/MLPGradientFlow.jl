module MLPGradientFlow

using ComponentArrays, LoopVectorization, OrdinaryDiffEq, ArrayInterface, Static,
      Optim, IfElse, LinearAlgebra, Distributed, Dates, Printf, Pkg,
      OrderedCollections, SLEEFPirates, Random, SpecialFunctions, Pickle, StrideArraysCore
using FastGaussQuadrature
using NLopt, Sundials

export Net, NetI, TeacherNet, Adam, Descent, FullBatch, MiniBatch
export loss, gradient, hessian, hessian_spectrum, train, random_params, params, params2dict, gauss_hermite_net
export sigmoid, softplus, g, gelu, square, relu, softmax, sigmoid2, cube, Poly, selu, silu, tanh_fast
export pickle, unpickle

###
### Activation Functions
###

softmax(x::AbstractMatrix) = (a -> a ./ sum(a, dims = 1))(exp.(x))
softmax(x::AbstractVector) = (a -> a ./ sum(a))(exp.(x))

struct Poly{N,T}
    coeff::NTuple{N,T}
end
Base.broadcastable(x::Poly) = Ref(x)
Poly(coeff...; T = Float64) = Poly{length(coeff),T}(coeff)
function _poly_inner(coeff, arg, x)
    first(coeff) * arg + _poly_inner(Base.tail(coeff), arg*x, x)
end
_poly_inner(coeff::Tuple{T}, arg, ::Any) where T = first(coeff) * arg
(p::Poly)(x::T, unused...) where T = _poly_inner(p.coeff, one(T), x)
function deriv(p::Poly)
    Poly(ntuple(i -> p.coeff[i+1]*i, length(p.coeff)-1))
end
second_deriv(p::Poly) = deriv(deriv(p))

silu(x) = x*sigmoid(x)
deriv(::typeof(silu)) = silu′
second_deriv(::typeof(silu)) = silu′′
function silu′(x::T, y) where T
    s = IfElse.ifelse(x == 0, T(0.5), y/x)
    x*sigmoid′(x, s) + s
end
function silu′(x)
    s = sigmoid(x)
    x*sigmoid′(x, s) + s
end
function silu′′(x::T, y, y′) where T
    s = IfElse.ifelse(x == 0, T(.5), y/x)
    s′ = sigmoid′(x, s)
    x*sigmoid′′(x, s, s′) + 2*s′
end
function silu′′(x)
    s = sigmoid(x)
    s′ = sigmoid′(x, s)
    x*sigmoid′′(x, s, s′) + 2*s′
end

selu(x::T) where T = T(1.05070098)*IfElse.ifelse(x > 0, x, T(1.67326324) * (exp(x) - 1))
deriv(::typeof(selu)) = selu′
second_deriv(::typeof(selu)) = selu′′
selu′(x, y) = selu′(x)
selu′(x::T) where T = T(1.05070098)*IfElse.ifelse(x > 0, one(T), T(1.67326324) * exp(x))
selu′′(x, y, y′) = selu′′(x)
selu′′(x::T) where T = IfElse.ifelse(x > 0, zero(T), T(1.05070098*1.67326324) * exp(x))

const sigmoid = sigmoid_fast
deriv(::typeof(sigmoid_fast)) = sigmoid′
second_deriv(::typeof(sigmoid_fast)) = sigmoid′′
sigmoid′(x) = sigmoid′(x, sigmoid(x))
sigmoid′(x, y) = y * (1 - y)
function sigmoid′′(x)
    y = sigmoid(x)
    sigmoid′′(x, y, sigmoid′(x, y))
end
sigmoid′′(x, y, y′) = y′ * (1 - 2y)

square(x) = x^2/2
deriv(::typeof(square)) = square′
second_deriv(::typeof(square)) = square′′
square′(x, y) = x
square′(x) = x
square′′(x::T, y, y′) where T = one(T)
square′′(::T) where T = one(T)

cube(x) = x^3
deriv(::typeof(cube)) = cube′
second_deriv(::typeof(cube)) = cube′′
cube′(x, y) = 3x^2
cube′(x) = 3x^2
cube′′(x, y, y′) = 6x
cube′′(x) = 6x

softplus(x) = IfElse.ifelse(x < 34, Base.log(Base.exp(x) + 1), x)
deriv(::typeof(softplus)) = softplus′
second_deriv(::typeof(softplus)) = softplus′′
softplus′(x, y) = sigmoid(x)
softplus′(x) = softplus′(x, nothing)
softplus′′(x, y, y′) = sigmoid′(x, y′)
softplus′′(x) = sigmoid′(x)

g(x) = sigmoid(4x) + softplus(x)
deriv(::typeof(g)) = g′
second_deriv(::typeof(g)) = g′′
g′(x)  = 4*sigmoid′(4x) + sigmoid(x)
g′′(x) = 16*sigmoid′′(4x) + sigmoid′(x)

const invsqrt2 = 1/sqrt(2)
const invsqrtπ = 1/sqrt(π)
const gelu = SLEEFPirates.gelu
gelu′(x, y) = IfElse.ifelse(x == 0, 0.5, y/x) + x*invsqrtπ*invsqrt2*Base.exp(-x^2/2)
gelu′(x) = gelu′(x, gelu(x))
gelu′′(x) = invsqrtπ*invsqrt2*(2 - x^2)*Base.exp(-x^2/2)
gelu′′(x, y, y′) = gelu′′(x)
deriv(::typeof(gelu)) = gelu′
second_deriv(::typeof(gelu)) = gelu′′

relu(x::T) where T = IfElse.ifelse(x > 0, x, zero(T))
deriv(::typeof(relu)) = relu′
second_deriv(::typeof(relu)) = relu′′
relu′(x, y) = x > 0
relu′(x) = x > 0
relu′′(::T) where T = zero(T)

deriv(::typeof(tanh_fast)) = tanh_fast′
second_deriv(::typeof(tanh_fast)) = tanh_fast′′
tanh_fast′(x, y) = 1 - y^2
tanh_fast′(x) = tanh_fast′(x, tanh_fast(x))
tanh_fast′′(x, y, y′) = -2y*y′
function tanh_fast′′(x)
    y = tanh_fast(x)
    tanh_fast′′(x, y, tanh_fast′(x, y))
end

deriv(::typeof(tanh)) = tanh′
second_deriv(::typeof(tanh)) = tanh′′
tanh′(x, y) = 1 - y^2
tanh′(x) = tanh′(x, tanh(x))
tanh′′(x, y, y′) = -2y*y′
function tanh′′(x)
    y = tanh(x)
    tanh′′(x, y, tanh′(x, y))
end

sigmoid2(x::T) where T = erf(x * T(0.7071067811865476))
deriv(::typeof(sigmoid2)) = sigmoid2′
second_deriv(::typeof(sigmoid2)) = sigmoid2′′
sigmoid2′(x::T) where T = T(0.7978845608028654) * Base.exp(-x^2/2)
sigmoid2′(x, y) = sigmoid2′(x)
sigmoid2′′(x, y, y′) = -x*y′
sigmoid2′′(x::T) where T = sigmoid2′′(x, zero(T), sigmoid2′(x))

deriv(::typeof(identity)) = identity′
second_deriv(::typeof(identity)) = identity′′
identity′(x::T) where T = one(T)
identity′(x::T, y) where T = one(T)
identity′′(x::T) where T = zero(T)
identity′′(x::T, y, y′) where T = zero(T)

deriv(::typeof(exp)) = exp
second_deriv(::typeof(exp)) = exp
Base.exp(x, _) = exp(x)
Base.exp(x, _, _) = exp(x)

@inline function A_mul_B!(a::AbstractMatrix{T}, f, w, input) where T
    @tturbo for m ∈ indices(w, 1), n ∈ indices(input, 2)
        amn = zero(T)
        for k ∈ indices(input, 1)
            amn += w[m,k] * input[k,n]
        end
        a[m, n] = f(amn)
    end
end
@inline function A_mul_B!(a::AbstractMatrix{T}, a′, f, w, input) where T
    f′ = deriv(f)
    @tturbo for m ∈ indices(w, 1), n ∈ indices(input, 2)
        amn = zero(T)
        for k ∈ indices(input, 1)
            amn += w[m,k] * input[k,n]
        end
        y = f(amn)
        a′[m, n] = f′(amn, y)
        a[m, n] = y
    end
end
@inline function A_mul_B!(a::AbstractMatrix{T}, a′, a′′, f, w, input) where T
    f′ = deriv(f)
    f′′ = second_deriv(f)
    @tturbo for m ∈ indices(w, 1), n ∈ indices(input, 2)
        amn = zero(T)
        for k ∈ indices(input, 1)
            amn += w[m,k] * input[k,n]
        end
        y = f(amn)
        y′ = f′(amn, y)
        a′′[m, n] = f′′(amn, y, y′)
        a′[m, n] = y′
        a[m, n] = y
    end
end
@inline A_mul_B!(a::AbstractMatrix{T}, a′, a′′, ::typeof(relu), w, input) where T =
    A_mul_B!(a, a′, relu, w, input)
@inline A_mul_B!(a::AbstractMatrix{T}, a′, ::typeof(identity), w, input) where T =
    A_mul_B!(a, identity, w, input)
@inline A_mul_B!(a::AbstractMatrix{T}, a′, a′′, ::typeof(identity), w, input) where T =
    A_mul_B!(a, identity, w, input)
@inline A_mul_B!(a::AbstractMatrix{T}, a′, a′′, ::typeof(square), w, input) where T =
    A_mul_B!(a, a′, square, w, input)
@inline function A_mul_B!(a::AbstractMatrix{T}, a′, ::typeof(g), w, input) where T
    @tturbo for m ∈ indices(w, 1), n ∈ indices(input, 2)
        amn = zero(T)
        for k ∈ indices(input, 1)
            amn += w[m,k] * input[k,n]
        end
        y1 = sigmoid(4*amn)
        a′[m, n] = 4*sigmoid′(amn, y1) + sigmoid(amn)
        a[m, n] = y1 + softplus(amn)
    end
end
@inline function A_mul_B!(a::AbstractMatrix{T}, a′, a′′, ::typeof(g), w, input) where T
    @tturbo for m ∈ indices(w, 1), n ∈ indices(input, 2)
        amn = zero(T)
        for k ∈ indices(input, 1)
            amn += w[m,k] * input[k,n]
        end
        y1 = sigmoid(4*amn)
        y1′ = sigmoid′(amn, y1)
        y2′ = sigmoid(amn)
        a′′[m, n] = 16*sigmoid′′(amn, y1, y1′) + sigmoid′(amn, y2′)
        a′[m, n] = 4y1′ + y2′
        a[m, n] = y1 + softplus(amn)
    end
end
A_mul_B!(a::AbstractMatrix{T}, ::typeof(softmax), w, input) where T =
    A_mul_B!(a, nothing, nothing, softmax, w, input)
A_mul_B!(a::AbstractMatrix{T}, a′, ::typeof(softmax), w, input) where T =
    A_mul_B!(a, a′, nothing, softmax, w, input)
@inline function A_mul_B!(a::AbstractMatrix{T}, a′, a′′, ::typeof(softmax), w, input) where T
    lastidx = size(a, 1)
    @tturbo for m ∈ indices(w, 1), n ∈ indices(input, 2)
        amn = zero(T)
        for k ∈ indices(input, 1)
            amn += w[m,k] * input[k,n]
        end
        a[m, n] = amn
    end
    @tturbo for n ∈ indices(input, 2)
        amax = typemin(T)
        for m ∈ indices(w, 1)
            amax = max(amax, a[m, n])
        end
        a[lastidx, n] = amax
    end
    @tturbo for m ∈ indices(w, 1), n ∈ indices(input, 2)
        a[m, n] = exp(a[m, n] - a[lastidx, n])
    end
    @tturbo for n ∈ indices(input, 2)
        an = zero(T)
        for m ∈ indices(w, 1)
            an += a[m, n]
        end
        a[lastidx, n] = an
    end
    @tturbo for m ∈ indices(w, 1), n ∈ indices(input, 2)
        a[m, n] /= a[lastidx, n]
        a[lastidx, n] = 0
    end
end

###
### Net
###

struct Dense{W, S1, T1, T2, T3, T4, T5, T6, T7, F}
    k::S1
    i0::Int
    f::F
    bias::Bool
    nup::Int
    nparams::Int
    w::W
    a::T1
    a′::T2
    a′′::T3
    delta::T4
    delta′::T5
    g::T6
    b::T7
end
StaticStrideArray(x) = StrideArray(x, StaticInt.(size(x)))
alloc_a′(::Any, T, k, N, nextbias) = StaticStrideArray(zeros(T, k + nextbias, N))
alloc_a′′(::Any, T, k, N) = StaticStrideArray(zeros(T, k, N))
function alloc_a′(::Union{typeof(identity), typeof(relu), typeof(softmax)}, T, k, N, nextbias)
    a′ = StaticStrideArray(ones(k + nextbias, N))
    if nextbias
        a′[end, :] .= false
    end
    a′
end
alloc_a′′(::Union{typeof(identity), typeof(relu), typeof(softmax)}, T, k, N) = StaticStrideArray(zeros(k, N))
alloc_a′′(::typeof(square), T, k, N) = StaticStrideArray(ones(k, N))
alloc_delta(::Any, T, k, N) = StaticStrideArray(zeros(T, k, N))
alloc_delta′(::Any, T, k, N) = StaticStrideArray(zeros(T, k, N))
alloc_g(::Any, T, k, nup, N) = StaticStrideArray(zeros(T, nup, k, N))
alloc_b(::Any, T, k, nup, N) = StaticStrideArray(zeros(T, k + nup, k, N))
function Dense(T, k, N; nin, nup, f, w, bias, nextbias, i0 = 1, derivs = 1)
    Dense(StaticInt(k),
          i0,
          f,
          bias,
          nup,
          k*(nin + bias),
          w,
          StaticStrideArray(ones(T, k + nextbias, N)),
          derivs > 0 ? alloc_a′(f, T, k, N, nextbias) : nothing, # nextbias is needed in the last loop of _hessian! (this could be improved, but without significant performance gain)
          derivs > 1 ? alloc_a′′(f, T, k, N) : nothing,
          derivs > 0 ? alloc_delta(f, T, k, N) : nothing,
          derivs > 1 ? alloc_delta′(f, T, k, N) : nothing,
          derivs > 1 ? alloc_g(f, T, k, nup, N) : nothing,
          derivs > 1 ? alloc_b(f, T, k, nup, N) : nothing,
         )
end
max_derivs_allocated(::Dense{W, S1, T1, Nothing, Nothing}) where {W, S1, T1} = 0
max_derivs_allocated(::Dense{W, S1, T1, T2, Nothing}) where {W, S1, T1, T2} = 1
max_derivs_allocated(::Dense) = 2
function Dense(l::Dense; N = size(l.a, 2), derivs = 1)
    Dense(eltype(l.a), Int(l.k), N;
          nup = l.nup, f = l.f, w = l.w, i0 = l.i0,
          bias = l.bias, nin = l.nparams÷l.k - l.bias,
          nextbias = size(l.a, 1) > l.k, derivs)
end
function Base.show(io::IO, d::Dense)
    println(io, "dense layer: $(Int(d.k)) neurons, activation $(d.f), $(d.bias ? "with" : "without") biases    # $(Int(d.nparams)) parameters")
end
struct Net{LS, L, I, T, W}
    nparams::Int
    Din::Int
    layerspec::LS
    layers::L
    input::I
    target::T
    weights::W
end
(n::Net)(x) = forward!(n, x)
(n::Net)(x, input) = forward!(Net(n; input, derivs = 0), x)
max_derivs_allocated(net::Net) = max_derivs_allocated(first(net.layers))
function prepare_input(input, d, firstlayerbias; copy_input = true, verbosity = 1)
    if size(input, 1) == d && firstlayerbias
        verbosity > 0 && @info("Appending 1 row to input for biases.")
        StaticStrideArray(vcat(input, ones(eltype(input), size(input, 2))'))
    elseif copy_input
        verbosity > 0 && @info("Copying input.")
        StaticStrideArray(copy(input))
    else
        StaticStrideArray(input)
    end
end
"""
    Net(; layers, input, target, weights = nothing,
          bias_adapt_input = true, derivs = 2, copy_input = true, verbosity = 1,
          Din = size(input, 1) - last(first(layers))*(1-bias_adapt_input))


    layers # ((num_neurons_layer1, activation_function_layer1, has_bias_layer1),
              (num_neurons_layer2, activation_function_layer2, has_bias_layer2),
              ...)
    input  # Dᵢₙ × N matrix
    target # Dₒᵤₜ × N matrix
    weights # nothing or N array
    bias_adapt_input = true # adds a row of 1s to the input
    derivs = 2              # allocate memory for derivs derivatives (0, 1, 2)
    copy_input = true       # copy the input when creating the net

### Example
```
input = randn(2, 100)
target = randn(1, 100)
net = Net(layers = ((10, softplus, true), (1, identity, true)),
          input = inp, target = targ)
"""
function Net(; layers, input::AbstractArray{T}, target::AbstractArray{S},
               weights = nothing,
               bias_adapt_input = true, derivs = 2, copy_input = true, verbosity = 1,
               Din = size(input, 1) - last(first(layers))*(1-bias_adapt_input)) where {T,S}
    if T != S && !(S <: Integer)
        @warn "`input` ($T) and `target` ($S) have not the same type."
    end
    if layers[end][2] ≠ softmax && !ismissing(layers[end][1]) && size(target, 1) ≠ layers[end][1]
        error("Output size of the network ($(layers[end][1])) does not match dimensionalty of the target ($(size(target, 1))).")
    end
    if isa(target, AbstractVector{<:Integer})
        if minimum(target) < 1 || maximum(target) > layers[end][1]
            error("Target labels have to be between 1 and $(layers[end][1]) (the number of units in the output layer). Got range $(minimum(target)),...,$(maximum(target))")
        end
    end
    layerspec = layers
    N = size(input, 2)
    input = prepare_input(input, Din, last(first(layerspec)); copy_input, verbosity)
    dims = (Din, first.(layerspec)...)
    if ismissing(last(layerspec)[1])
        layerspec = (layerspec[1:end-1]..., (size(target, 1), last(layerspec)[2:end]...))
    end
    i0 = 1
    layers = tuple([begin
                    l = Dense(T, k, N;
                     derivs,
                     nin = dims[i],
                     nup = i == length(layerspec) ? 0 : sum(first.(layerspec[i+1:end])),
                     f, w = Val(Symbol(:w, i)),
                     bias, i0,
                     nextbias = i == length(layerspec) ? (layerspec[i][2] == softmax) : last(layerspec[i+1]))
                    i0 += l.nparams
                    l
                    end
               for (i, (k, f, bias)) in pairs(layerspec)]...)
    target = StaticStrideArray(copy(target))
    Net(Int(sum(getproperty.(layers, :nparams))), Din, layerspec, layers, input, target, weights)
end
Net(net::Net; input = net.input, kwargs...) = Net(; layers = net.layerspec, input, target = net.target, bias_adapt_input = (input !== net.input), copy_input = false, derivs = max_derivs_allocated(net), weights = net.weights, kwargs...)
function Base.show(io::IO, n::Net)
    println(io, "Multilayer Perceptron ($(n.nparams) parameters of type $(eltype(n.input)))\ninput dimensions: $(size(n.input, 1)-first(n.layers).bias)")
    for l in n.layers
        show(io, l)
    end
end

struct TeacherNet{N,P}
    net::N
    p::P
end
"""
    TeacherNet(; p = nothing, input, kwargs...)

Creates a network with parameters `p` attached. If `p == nothing`, [`random_params`](@href) are generated.
A `TeacherNet` is a callable object that returns the target given some input.

# Example
```
julia> input = randn(3, 10^4);

julia> teacher = TeacherNet(; layers = ((8, softplus, true), (1, identity, true)), input);

julia> target = teacher(input);

julia> new_input = randn(3, 10^3);

julia> new_target = teacher(new_input);
```
"""
function TeacherNet(; p = nothing, input, kwargs...)
    target = zeros(1, size(input, 2))
    net = Net(; input, target, kwargs..., derivs = 0)
    if p === nothing
        p = random_params(net)
    end
    net.target .= net(p)
    TeacherNet(net, p)
end
function Base.show(io::IO, teacher::TeacherNet)
    println(io, "TeacherNet")
    show(io, teacher.net)
end
(n::TeacherNet)(inp) = n.net(n.p, inp)

###
### gaussian input
###

include("normal_integrals.jl")
include("gaussian_input.jl")


###
### forward - backward
###

propagate!(::Tuple{}, ::Any, ::Any; kwargs...) = nothing
function propagate!(layers::Tuple, x, input; derivs = 0)
    l = first(layers)
    if derivs == 0
        A_mul_B!(l.a, l.f, getweights(l, x), input)
    elseif derivs == 1
        A_mul_B!(l.a, l.a′, l.f, getweights(l, x), input)
    else
        A_mul_B!(l.a, l.a′, l.a′′, l.f, getweights(l, x), input)
    end
    propagate!(Base.tail(layers), x, l.a; derivs)
end
getweights(l, x::ComponentArray) = getproperty(x, l.w)
getweights(l, x) = reshape(view(x, l.i0:l.i0+l.nparams-1), Int(l.k), :)
function forward!(net, x, input; derivs = 1)
    propagate!(net.layers, x, input; derivs)
    last(net.layers).a
end
nparams(net::Net) = net.nparams
nparams(net::NetI) = net.student.nparams
function checkparams(net, x)
    @assert length(x) == nparams(net) "Parameter vector has length $(length(x)) but network has $(nparams(net)) parameters."
end
function forward!(net, x; derivs = 0)
    checkparams(net, x)
    forward!(net, x, net.input; derivs)
end

struct NegativeLogLikelihood end
struct MSE end
function ℓ(::NegativeLogLikelihood, a, target::AbstractVector, ::Nothing)
    res = zero(eltype(a))
    @tturbo for j in indices(target)
        res -= log(a[target[j], j])
    end
    res
end
function ℓ(::NegativeLogLikelihood, a, target::AbstractMatrix{T}, ::Nothing) where T
    res = zero(T)
    @inbounds @fastmath for i in indices(target, 1), j in indices(target, 2)
        t = target[i, j]
        res -= IfElse.ifelse(isequal(t, zero(T)), zero(T), t * log(a[i, j]))
    end
    res
end
function ℓ(::NegativeLogLikelihood, a, target::AbstractVector, weights)
    res = zero(eltype(a))
    @tturbo for j in indices(target)
        res -= log(a[target[j], j]) * weights[j]
    end
    res
end
function ℓ(::NegativeLogLikelihood, a, target::AbstractMatrix{T}, weights) where T
    res = zero(T)
    @inbounds @fastmath for i in indices(target, 1), j in indices(target, 2)
        t = target[i, j]
        res -= IfElse.ifelse(isequal(t, zero(T)), zero(T), t * log(a[i, j])) * weights[j]
    end
    res
end
function ℓ(::MSE, a, target, ::Nothing)
    res = zero(eltype(a))
    @tturbo for i in indices(target, 1), j in indices(target, 2)
        res += (target[i, j] - a[i, j])^2
    end
    res
end
function ℓ(::MSE, a, target, weights)
    res = zero(eltype(a))
    @tturbo for i in indices(target, 1), j in indices(target, 2)
        res += (target[i, j] - a[i, j])^2 * weights[j]
    end
    res
end
function __loss(net::Net, x; forward = true, derivs = 0, weights = net.weights, losstype = MSE())
    input = net.input
    target = net.target
    forward && forward!(net, x, input; derivs)
    l = last(net.layers)
    ℓ(losstype, l.a, target, weights)
end
function _loss(net, x;
               forward = true, weights = isa(net, Net) ? net.weights : nothing,
               losstype = MSE(),
               derivs = 0,
               nx = weightnorm(x), maxnorm = Inf, merge = nothing)
    res = __loss(net, x; forward, weights, losstype, derivs)
    if nx > maxnorm
        res += (nx - maxnorm)^3/3 * n_samples(net)
    end
    if merge !== nothing
        w1 = getweights(net.layers[merge.layer], x)
        λ = merge.lambda/2 * n_samples(net)
        i, j = merge.pair
        for k in indices(w1, 2)
            res += λ * (w1[i, k] - w1[j, k])^2
        end
    end
    res
end
"""
    loss(net, x, input = net.input, target = net.target;
         verbosity = 1, losstype = MSE())

Compute the loss of `net` at parameter value `x`.
"""
function loss(net::Net, x; input = net.input, target = net.target, kwargs...)
    checkparams(net, x)
    _net = if input != net.input || target != net.target
        Net(net; input, target)
    else
        net
    end
    _loss(_net, x; kwargs...)/size(input, 2)
end
function ℓ′!(::MSE, l, target)
    @tturbo for m in indices(target, 1), n in indices(target, 2)
        l.delta[m, n] = 2 * l.a′[m, n] * (l.a[m, n] - target[m, n])
    end
end
function ℓ′′!(::MSE, l, target)
    @tturbo for m in indices(l.delta, 1), n in indices(target, 2)
        d = 2 * (l.a[m, n] - target[m, n])
        l.delta[m, n] = l.a′[m, n] * d
        l.delta′[m, n] = l.a′′[m, n] * d
    end
end
function ℓ′!(::NegativeLogLikelihood, l, target::AbstractVector)
    a = l.a
    T = eltype(l.a)
    @tturbo for i in 1:l.k, j in indices(target)
        l.delta[i, j] = a[i, j]
    end
    @tturbo for j in indices(target)
        l.delta[target[j], j] -= one(T)
    end
end
function ℓ′!(::NegativeLogLikelihood, l, target::AbstractMatrix)
    a = l.a
    @tturbo for i in indices(target, 1), j in indices(target, 2)
        l.delta[i, j] = a[i, j] - target[i, j]
    end
end
ℓ′′!(lt::NegativeLogLikelihood, l, target) = ℓ′!(lt, l, target)
function backprop!(layers, target, x::AbstractArray{T};
                   derivs = 1,
                   losstype = MSE()) where T
    l = last(layers)
    if derivs == 1
        ℓ′!(losstype, l, target)
    else
        ℓ′′!(losstype, l, target)
    end
    backprop!(layers, x; derivs)
end
function _backprop!(pdelta, pa′, delta, w)
    for k in indices(pdelta, 1), i in indices(pdelta, 2)
        deltaki = zero(eltype(pdelta))
        @tturbo for l in indices(delta, 1)
            deltaki += delta[l, i] * w[l, k]
        end
        pdelta[k, i] = deltaki * pa′[k, i]
    end
end
function _backprop!(pdelta, pa′, pdelta′, pa′′, delta, w)
    for k in indices(pdelta, 1), i in indices(pdelta, 2)
        deltaki = zero(eltype(pdelta))
        @tturbo for l in indices(delta, 1)
            deltaki += delta[l, i] * w[l, k]
        end
        pdelta[k, i] = deltaki * pa′[k, i]
        pdelta′[k, i] = deltaki * pa′′[k, i]
    end
end
function backprop!(layers, x; derivs = 1)
    length(layers) == 1 && return
    l = last(layers)
    front = Base.front(layers)
    w = getweights(l, x)
    prev = last(front)
    if derivs == 1
        _backprop!(prev.delta, prev.a′, l.delta, w)
    else
        _backprop!(prev.delta, prev.a′, prev.delta′, prev.a′′, l.delta, w)
    end
    backprop!(front, x; derivs)
end
get_input(::Tuple{}, input) = input
get_input(layers, _) = last(layers).a
function _update_derivatives!(dw, delta, inp, ::Nothing)
    @tturbo for k in indices(delta, 1), l in indices(inp, 1)
        wkl = zero(eltype(dw))
        for i in indices(inp, 2)
            wkl += delta[k, i] * inp[l, i]
        end
        dw[k, l] = wkl
    end
end
function _update_derivatives!(dw, delta, inp, weights)
    @tturbo for k in indices(delta, 1), l in indices(inp, 1)
        wkl = zero(eltype(dw))
        for i in indices(inp, 2)
            wkl += delta[k, i] * inp[l, i] * weights[i]
        end
        dw[k, l] = wkl
    end
end
function update_derivatives!(dx, layers, input, x, weights)
    l = last(layers)
    dw = getweights(l, dx)
    delta = l.delta
    front = Base.front(layers)
    inp = get_input(front, input)
    _update_derivatives!(dw, delta, inp, weights)
    length(front) == 0 && return
    update_derivatives!(dx, front, input, x, weights)
end
function _gradient!(dx, net::Net, x; forward = true, derivs = 1, weights = net.weights, losstype = MSE())
    input = net.input
    target = net.target
    forward && forward!(net, x, input; derivs)
    layers = net.layers
    backprop!(layers, target, x; derivs, losstype)
    update_derivatives!(dx, layers, input, x, weights)
end
function gradient!(dx, net, x;
                   forward = true,
                   nx = weightnorm(x),
                   maxnorm = Inf,
                   merge = nothing,
                   weights = net.weights,
                   losstype = MSE(),
                   derivs = 1)
    _gradient!(dx, net, x; forward, weights, derivs, losstype)
    if nx > maxnorm
        dx .+= (nx - maxnorm)^2*x/length(x) * n_samples(net)
    end
    if merge !== nothing
        dw = getweights(net.layers[merge.layer], dx)
        w = getweights(net.layers[merge.layer], x)
        λ = merge.lambda * n_samples(net)
        i, j = merge.pair
        for k in indices(w, 2)
            dw[i, k] -= λ * (w[j, k] - w[i, k])
            dw[j, k] -= λ * (w[i, k] - w[j, k])
        end
    end
    dx
end
"""
    gradient(net, x)

Compute the gradient of `net` at parameter value `x`.
"""
function gradient(net::Net, x; kwargs...)
    checkparams(net, x)
    dx = zero(x)
    gradient!(dx, net, x; kwargs...)
    dx ./= size(net.input, 2)
end

function forward_g!(g, prev, layers, x, offset = StaticInt(0))
    l = first(layers)
    w = getweights(l, x)
    a′ = prev.a′
    if offset == 0
        @tturbo for l in 1:l.k, k in indices(g, 2), i in indices(a′, 2)
            g[l, k, i] = a′[k, i] * w[l, k]
        end
        offset = l.k
    else
        oldoffset = offset - prev.k
        @tturbo for l in 1:l.k, k in indices(g, 2), i in indices(a′, 2)
            glk = zero(eltype(g))
            for r in 1:prev.k
                glk += a′[r, i] * w[l, r] * g[r + oldoffset, k, i]
            end
            g[l + offset, k, i] = glk
        end
        offset += l.k
    end
    forward_g!(g, l, Base.tail(layers), x, offset)
end
forward_g!(::Any, ::Any, ::Tuple{}, ::Any, ::Any; kwargs...) = nothing
function compute_g!(layers, x)
    tail = Base.tail(layers)
    l = first(layers)
    forward_g!(l.g, l, tail, x)
    compute_g!(tail, x)
end
compute_g!(::Tuple{}, ::Any; kwargs...) = nothing
function ℓ_backprop_b!(::MSE, b, g, l, off, goff)
    a′ = l.a′
    delta′ = l.delta′
    T = eltype(b)
    if l.b === b # output layer
        @tturbo for k in indices(b, 2), i in indices(a′, 2)
            ap = T(a′[k, i])
            b[k, k, i] = delta′[k, i] + 2*ap*ap
        end
    else
        @tturbo for n in indices(l.b, 2), i in indices(a′, 2)
            ap = a′[n, i]
            tmp = 2*ap*ap + delta′[n, i]
            for k in indices(b, 2)
                b[n + off, k, i] = g[n + goff, k, i] * tmp
            end
        end
    end
end
function ℓ_backprop_b!(::NegativeLogLikelihood, b, g, l, off, goff)
    a = l.a
    if l.b === b # output layer
        @tturbo for k in indices(b, 2), n in indices(b, 2), i in indices(a, 2)
            b[k, n, i] = - a[k, i] * a[n, i]
        end
        @tturbo for k in indices(b, 2), i in indices(a, 2)
            b[k, k, i] += a[k, i]
        end
    else
        @tturbo for k in indices(b, 2), n in indices(l.b, 2), i in indices(a, 2)
            bnki = zero(eltype(b))
            for m in indices(l.b, 2)
                bnki -= g[m + goff, k, i] * a[m, i] * a[n, i]
            end
            b[n + off, k, i] = g[n + goff, k, i] * a[n, i] + bnki
        end
    end
end
function backprop_b!(b, g, layers, prev, x::AbstractArray{T}, offset = StaticInt(0);
                     losstype = MSE()) where T
    l = last(layers)
    delta′ = l.delta′
    a′ = l.a′
    if prev === nothing
        offset = l.k
        off = size(b, 1) - offset
        goff = off - size(b, 2)
        ℓ_backprop_b!(losstype, b, g, l, off, goff)
    else
        w = getweights(prev, x)
        oldoff = size(b, 1) - offset
        offset = offset + l.k
        off = size(b, 1) - offset
        goff = off - size(b, 2)
        if goff < 0
            @tturbo for n in 1:l.k, k in indices(b, 2), i in indices(a′, 2)
                bki = zero(T)
                for r in 1:prev.k
                    bki += w[r, n] * b[r + oldoff, k, i]
                end
                b[n + off, k, i] = T(a′[n, i]) * bki
            end
            @tturbo for k in 1:l.k, i in indices(b, 3)
                b[k + off, k, i] += delta′[k, i]
            end
        else
            @tturbo for n in 1:l.k, k in indices(b, 2), i in indices(a′, 2)
                bki = zero(T)
                for r in 1:prev.k
                    bki += w[r, n] * b[r + oldoff, k, i]
                end
                b[n + off, k, i] = g[n + goff, k, i] * delta′[n, i]
                b[n + off, k, i] += T(a′[n, i]) * bki
            end
        end
    end
    l.g === g && return
    backprop_b!(b, g, Base.front(layers), l, x, offset; losstype)
end
function compute_b!(front, layers, x; losstype = MSE())
    l = last(front)
    backprop_b!(l.b, l.g, layers, nothing, x; losstype)
    compute_b!(Base.front(front), layers, x; losstype)
end
compute_b!(::Tuple{}, ::Any, ::Any; kwargs...) = nothing
# This function is very costly; don't know how to optimize further
function _update_hessian!(hh, w1, w2, inp1, inp2, b, off, ::Nothing)
    @tturbo for k in indices(w1, 1), j in indices(w1, 2),
               n in indices(w2, 1), l in indices(w2, 2)
        hkjnl = zero(eltype(w1))
        for i in indices(inp1, 2)
            hkjnl += inp1[j, i] * inp2[l, i] * b[n + off, k, i]
        end
        hh[k, j, n, l] = hkjnl
    end
end
function _update_hessian2!(hh, w1, w2, inp1, delta, a′, ::Nothing)
    @tturbo for k in indices(w1, 1), j in indices(w1, 2), n in indices(w2, 1)
        hkjnk = zero(eltype(w1))
        for i in indices(inp1, 2)
            hkjnk += inp1[j, i] * delta[n, i] * a′[k, i]
        end
        hh[k, j, n, k] += hkjnk
    end
end
function _update_hessian3!(hh, w1, w2, inp1, inp2, delta, b, a′, g, off, goff, ::Nothing)
    @tturbo for k in indices(w1, 1), j in indices(w1, 2),
               n in indices(w2, 1), l in indices(w2, 2)
        hkjnl = zero(eltype(w1))
        for i in indices(inp1, 2)
            hkjnl += inp1[j, i] * inp2[l, i] * b[n + off, k, i]
            hkjnl += inp1[j, i] * delta[n, i] * a′[l, i] * g[l + goff, k, i]
        end
        hh[k, j, n, l] = hkjnl
    end
end
function _update_hessian!(hh, w1, w2, inp1, inp2, b, off, weights)
    @tturbo for k in indices(w1, 1), j in indices(w1, 2),
               n in indices(w2, 1), l in indices(w2, 2)
        hkjnl = zero(eltype(w1))
        for i in indices(inp1, 2)
            hkjnl += inp1[j, i] * inp2[l, i] * b[n + off, k, i] * weights[i]
        end
        hh[k, j, n, l] = hkjnl
    end
end
function _update_hessian2!(hh, w1, w2, inp1, delta, a′, weights)
    @tturbo for k in indices(w1, 1), j in indices(w1, 2), n in indices(w2, 1)
        hkjnk = zero(eltype(w1))
        for i in indices(inp1, 2)
            hkjnk += inp1[j, i] * delta[n, i] * a′[k, i] * weights[i]
        end
        hh[k, j, n, k] += hkjnk
    end
end
function _update_hessian3!(hh, w1, w2, inp1, inp2, delta, b, a′, g, off, goff, weights)
    @tturbo for k in indices(w1, 1), j in indices(w1, 2),
               n in indices(w2, 1), l in indices(w2, 2)
        hkjnl = zero(eltype(w1))
        for i in indices(inp1, 2)
            hkjnl += inp1[j, i] * inp2[l, i] * b[n + off, k, i] * weights[i]
            hkjnl += inp1[j, i] * delta[n, i] * a′[l, i] * g[l + goff, k, i] * weights[i]
        end
        hh[k, j, n, l] = hkjnl
    end
end
function _hessian!(h, l1, layers, inp1, inp2, prev, x,
                   off1 = StaticInt(1), off = StaticInt(0), off2 = off1, weights = nothing)
    l2 = first(layers)
    if l1 === l2
        w = getweights(l1, x)
        b = l2.b
        hh = h.blocks[off1]
        _update_hessian!(hh, w, w, inp1, inp1, b, 0, weights)
        off += l1.k
    else
        a′ = prev.a′
        w1 = getweights(l1, x)
        w2 = getweights(l2, x)
        b = l2.b
        hh = h.blocks[off1]
        if l1.a′ === a′
            _update_hessian!(hh, w1, w2, inp1, inp2, l1.b, off, weights)
            _update_hessian2!(hh, w1, w2, inp1, l2.delta, a′, weights)
        else
            goff = off - l1.k - prev.k
            _update_hessian3!(hh, w1, w2, inp1, inp2, l2.delta, l1.b, a′, l1.g, off, goff, weights)
        end
        off += l2.k
    end
    off1 += StaticInt(1)
    length(layers) == 1 && return off1
    _hessian!(h, l1, Base.tail(layers), inp1, l2.a, l2, x, off1, off, off2, weights)
end
function compute_h!(h, layers, input, x, off1 = StaticInt(1), weights = nothing)
    l = first(layers)
    off1 = _hessian!(h, l, layers, input, input, nothing, x, off1, StaticInt(0), off1, weights)
    compute_h!(h, Base.tail(layers), l.a, x, off1, weights)
end
compute_h!(::Any, ::Tuple{}, ::Any, ::Any, ::Any, ::Any) = nothing
function copy_h!(flat, offsets, blocks)
    for idx in eachindex(offsets)
        off1, off2 = offsets[idx]
        b = blocks[idx]
        N1 = size(b, 1)
        N2 = size(b, 3)
        @tturbo for i in indices(b, 1), j in indices(b, 2), k in indices(b, 3), l in indices(b, 4)
            tmp = b[i, j, k, l]
            flat[off1 + (j-1)*N1 + i, off2 + (l-1)*N2 + k] = tmp
            flat[off2 + (l-1)*N2 + k, off1 + (j-1)*N1 + i] = tmp
        end
    end
end
function _hessian!(h, net::Net, x;
        forward = true, backprop = true, derivs = 2, losstype = MSE(), weights = net.weights)
    input = net.input
    target = net.target
    forward && forward!(net, x, input; derivs)
    layers = net.layers
    backprop && backprop!(layers, target, x; derivs, losstype)
    compute_g!(layers, x)
    compute_b!(layers, layers, x; losstype)
    compute_h!(h, layers, input, x, StaticInt(1), weights)
    copy_h!(h.flat, h.offsets, h.blocks)
    h.flat
end
function hessian!(h, net, x;
                  forward = true, backprop = true,
                  losstype = MSE(), derivs = 2,
                  nx = weightnorm(x), maxnorm = Inf, merge = nothing, weights = net.weights)
    hflat = _hessian!(h, net, x; forward, backprop, losstype, weights, derivs)
    if merge !== nothing # doesn't work for NetI
        λ = merge.lambda * n_samples(net)
        i, j = merge.pair
        hw1 = h.blocks[merge.layer]
        for k in indices(hw1, 2)
            hw1[i, k, i, k] += λ
            hw1[i, k, j, k] -= λ
            hw1[j, k, j, k] += λ
            hw1[j, k, i, k] -= λ
        end
        copy_h!(hflat, h.offsets, h.blocks)
    end
    if nx > maxnorm
        hflat .+= (2*(nx - maxnorm) * x * x'/length(x)^2 + I*(nx - maxnorm)^2/length(x)) * n_samples(net)
    end
    hflat
end
struct Hessian{T}
    blocks::Vector{Array{T, 4}}
    offsets::Vector{Tuple{Int,Int}}
    flat
end
function Hessian(x, flat = zeros(eltype(x), length(x), length(x)))
    k = keys(x)
    offsets = [(j == 1 ? 0 : sum(length(getproperty(x, k[l])) for l in 1:j-1),
                i == 1 ? 0 : sum(length(getproperty(x, k[l])) for l  in 1:i-1))
               for i in eachindex(k), j in eachindex(k) if j ≤ i]
    blocks = [zeros(eltype(x),
                    size(getproperty(x, k[j]))...,
                    size(getproperty(x, k[i]))...)
              for i in eachindex(k), j in eachindex(k) if j ≤ i]
    Hessian(blocks, offsets, flat)
end
Hessian(template::Hessian, H) = Hessian(template.blocks, template.offsets, H)
Hessian(::Nothing, H) = H
"""
    hessian(net, x)

Compute hessian of `net` at parameter value `x`.
"""
function hessian(net::Net, x; kwargs...)
    checkparams(net, x)
    H = Hessian(x)
    hessian!(H, net, x; kwargs...) ./ size(net.input, 2)
end
"""
    hessian_spectrum(net, x)

Compute the spectrum of the hessian of `net` at `x`.
"""
hessian_spectrum(net, x) = eigen(Symmetric(hessian(net, x)))

###
### Optimizers
###
### Copied and adapted from https://github.com/FluxML/Flux.jl/blob/ba48ad082a257fef9067c006cfc3f52d61e6da70/src/optimise/optimisers.jl#L167
abstract type AbstractOptimiser end
mutable struct Adam <: AbstractOptimiser
    eta::Float64
    beta::Tuple{Float64,Float64}
    epsilon::Float64
    state::IdDict{Any, Any}
end
Adam(η::Real = 0.001, β::Tuple = (0.9, 0.999), ϵ::Real = 1e-8) = Adam(η, β, ϵ, IdDict())
Adam(η::Real, β::Tuple, state::IdDict) = Adam(η, β, 1e-8, state)

function apply!(o::Adam, x, Δ)
    η, β = o.eta, o.beta

    mt, vt, βp = get!(o.state, x) do
      (zero(x), zero(x), Float64[β[1], β[2]])
    end :: Tuple{typeof(x),typeof(x),Vector{Float64}}

    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η
    βp .= βp .* β

    return Δ
end
mutable struct Descent <: AbstractOptimiser
    eta::Float64
end

Descent() = Descent(1e-3)

function apply!(o::Descent, x, Δ)
    Δ .*= o.eta
end

###
### Batch
###
struct FullBatch end
step!(::Any, ::FullBatch) = nothing
start_of_epoch(::FullBatch) = true
mutable struct MiniBatch{T1,T2}
    input::T1
    target::T2
    batchsize::Int
    start::Int
end
function step!(n, b::MiniBatch)
    start = b.start
    batchsize = b.batchsize
    @views n.input .= b.input[:, start:start+batchsize-1]
    @views n.target .= b.target[:, start:start+batchsize-1]
    b.start += batchsize
    if b.start > size(b.input, 2) - batchsize + 1
        b.start = 1
    end
end
start_of_epoch(b::MiniBatch) = b.start == 1


###
### ODE
###

needs_hessian(::Any) = true
needs_hessian(::Descent) = false
needs_hessian(::Adam) = false
needs_hessian(::BFGS) = false
needs_hessian(::LBFGS) = false
needs_hessian(::RK4) = false
needs_hessian(::Symbol) = false # NLopt
function needs_hessian(it1, alg1, it2, alg2)
    (it1 > 0 && needs_hessian(alg1)) || (it2 > 0 && needs_hessian(alg2))
end
function swapsign(f!)
    let f! = f!
        function(dx, x, _, _)
            f!(dx, x)
            dx .*= -1.
        end
    end
end
Base.@kwdef mutable struct ODETerminator{T}
    i::Int = 0
    t0::Float64 = 0.
    maxtime::Float64 = 180.
    maxiter::Int = typemax(Int)
    stopped_by::String = ""
    o::T
end
function (terminator::ODETerminator)(u, t, integrator)
    if terminator.i == 0
        terminator.t0 = time()
    end
    terminator.i += 1
    Δ = time() - terminator.t0
    if converged(terminator.o)
        terminator.stopped_by = "patience ($(terminator.o.patience))"
        true
    elseif Δ > terminator.maxtime
        terminator.stopped_by = "maxtime_ode > $(terminator.maxtime)s"
        true
    elseif t > 1e300
        terminator.stopped_by = "t > 1e300"
        true
    elseif terminator.i ≥ terminator.maxiter
        terminator.stopped_by = "maxiterations_ode ≥ $(terminator.maxiter)"
        true
    else
        false
    end
end
function terminator(o; maxtime = 20, maxiter = typemax(Int), losstype = MSE())
    DiscreteCallback(ODETerminator(; o, maxtime = float(maxtime), maxiter),
                     terminate!)
end
function optim_solver_default(x)
    length(x) ≤ 128 && return Newton(linesearch = Optim.LineSearches.HagerZhang(linesearchmax = 1000))
    length(x) ≤ 1024 && return :LD_SLSQP
    length(x) ≤ 10^6 && return BFGS()
    return LBFGS()
end
function alg_default(x)
    length(x) ≤ 1024 && return KenCarp58()
    return TRBDF2(autodiff = false)
end
function default_hessian_template(net, p, maxiterations_ode, alg,
                                  maxiterations_optim, optim_solver, verbosity)
    isa(net, NetI) && return nothing
    if needs_hessian(maxiterations_ode, alg, maxiterations_optim, optim_solver)
        if length(p) > 10^4 && verbosity > 0
            @warn "Computing Hessians for $(length(p)) parameters requires a lot of memory and time"
        end
        Hessian(p)
    else
        nothing
    end
end
weightnorm(x) = sum(abs2, x)/(2*length(x))
Base.@kwdef mutable struct OptimizationState{T,N,NE,LT,Tau,M,B,W}
    net::N
    net_eval::NE = net
    t0::Float64 = time()
    fk::Int = 0
    gk::Int = 0
    hk::Int = 0
    k_last_best::Int = 0
    bestl::Float64 = Inf
    bestx::T = random_params(net)
    maxnorm::Float64 = Inf
    verbosity::Int = 0
    show_progress::Bool = true
    hessian_template = nothing
    progress_interval::Float64 = 5.
    patience::Int = 2*10^4
    losstype::LT = isa(net, NetI) ? MSE() : net.layers[end].f == softmax ? NegativeLogLikelihood() : MSE()
    tauinv::Tau = nothing
    merge::M = nothing
    batcher::B = FullBatch()
    weights::W = nothing
end
scale!(x, ::Nothing) = x
scale!(x, tauinv) = x .*= tauinv
function OptimizationState(net; maxnorm = Inf, progress_interval = 5., net_eval = net, batchsize = nothing, kwargs...)
    if !isnothing(batchsize) && batchsize != n_samples(net)
        if n_samples(net) % batchsize != 0
            @warn "Number of samples $(n_samples(net)) is not a multiple of the batchsize $batchsize."
        end
        batcher = MiniBatch(net.input, net.target, batchsize, 1)
        _net = Net(net,
                   input = net.input[:, 1:batchsize],
                   target = net.target[:, 1:batchsize],
                   bias_adapt_input = false)
    else
        batcher = FullBatch()
        _net = net
    end
    OptimizationState(; net = _net, net_eval, maxnorm = float(maxnorm), progress_interval = float(progress_interval), batcher, kwargs...)
end
struct Loss{T,N}
    o::OptimizationState{T,N}
end
function (l::Loss)(x; nx = weightnorm(x), forward = true, derivs = 0)
    o = l.o
    loss = _loss(o.net_eval, x;
                 losstype = o.losstype, forward, nx, maxnorm = o.maxnorm,
                 merge = o.merge, weights = o.weights, derivs)
    o.fk += 1
    if loss < o.bestl
        o.k_last_best = o.fk
        o.bestl = loss
        o.bestx .= x
    end
    loss
end
struct Grad{T,N}
    l::Loss{T,N}
end
function (g::Grad)(dx, x; nx = weightnorm(x), forward = true, kwargs...)
    o = g.l.o
    step!(o.net, o.batcher)
    gradient!(dx, o.net, x; forward, nx, maxnorm = o.maxnorm, merge = o.merge, weights = o.weights, kwargs...)
    if forward == true && start_of_epoch(o.batcher)
        g.l(x; nx, forward = o.net !== o.net_eval)
    end
    o.gk += 1
    if o.show_progress
        if time() - o.t0 > o.progress_interval
            o.t0 = time()
            @printf "%s: evals: (ℓ: %6i, ∇: %6i, ∇²: %6i), loss: %g\n" now() o.fk o.gk o.hk o.bestl
        end
    end
    scale!(dx, g.l.o.tauinv)
end
struct Hess{T,N}
    g::Grad{T,N}
end
function (h::Hess)(H, x; nx = weightnorm(x), kwargs...)
    o = h.g.l.o
    hessian!(Hessian(o.hessian_template, H), o.net, x;
             nx, maxnorm = o.maxnorm,
             weights = o.weights,
             merge = o.merge, kwargs...)
    o.hk += 1
    scale!(H, h.g.l.o.tauinv)
end
struct NLOptObj{T,N} <: Function
    g::Grad{T,N}
end
converged(o) = o.fk - o.k_last_best > o.patience
function (nl::NLOptObj)(x, G)
    nx = weightnorm(x)
    loss = nl.g.l(x; nx, derivs = 1)
    if length(G) > 0
        nl.g(G, x; nx, forward = false)
    end
    if converged(nl.g.l.o)
        error()
    end
    loss
end
struct OptimObj{T,N}
    h::Hess{T,N}
end
function (op::OptimObj)(F, G, H, x)
    nx = weightnorm(x)
    derivs = min(2, (G !== nothing) + 2*(H !== nothing))
    loss = op.h.g.l(x; nx, derivs)
    backprop = true
    if G !== nothing
        op.h.g(G, x; nx, forward = false, derivs)
        backprop = false
    end
    H === nothing || op.h(H, x; nx, forward = false, backprop = backprop)
    F === nothing || return loss
    nothing
end

function get_functions(net, maxnorm; kwargs...)
    o = OptimizationState(net; maxnorm, kwargs...)
    l = Loss(o)
    g = Grad(l)
    h = Hess(g)
    (l, g, h, OptimObj(h), NLOptObj(g))
end
"""
    train(net, x0; kwargs...)

Train `net` from initial point `x0`.

Keyword arguments:

    maxnorm = Inf                  # constant c in loss formula

    batchsize = nothing,           # using the full data set in each step when `nothing`

    alg = alg_default(p)           # ODE solver: KenCarp58() for length(p) ≤ 64, RK4() otherwise
    maxT = 1e10                    # upper integration limit
    save_everystep = true          # return a trajectory and loss curve
    n_samples_trajectory = 100     # number of samples of the trajectory
    abstol = 1e-6                  # absolute tolerance of the ODE solver
    reltol = 1e-3                  # relative tolerance of the ODE solver
    maxtime_ode = 3*60             # maximum amount of time in seconds for the ODE solver
    maxiterations_ode = 10^6       # maximum iterations of ODE solver

    maxiterations_optim = 10^5     # maximum iterations of optimizer
    g_tol = 1e-12                  # stop if gradient norm is below g_tol
    patience = 10^4                # Number of steps without decrease of the loss function until converged
    tauinv = nothing               # nothing, a scalar or a ComponentArray of shape `x0` with inverse time scales
    minloss = 1e-30                # stop if MSE loss is below minloss
    maxtime_optim = 2*60           # maximum amount of time in seconds for the Newton optimization
    optim_solver = optim_solver_default(p) # optimizer: NewtonTrustRegion() for length(p) ≤ 32, :LD_SLSQP for length(p) ≤ 1000, BFGS() otherwise

    verbosity = 1                  # use verbosity = 1 for more outputs
    show_progress = true           # show progress
    progress_interval = 5          # show progress every x seconds
    result = :dict                 # change to result = :raw for more detailed results
    exclude = String[]             # dictionary keys to exclude from the results dictionary
    include = nothing              # dictionary keys to include (everything if nothing)

"""
layer_spec(net::Net) = net.layerspec
layer_spec(net::NetI) = net.student.layerspec
weights(net::Net) = net.weights
weights(::NetI) = nothing
function train(net, p;
               maxnorm = Inf,
               verbosity = 1,
               batchsize = nothing,
               losstype = MSE(),
               show_progress = verbosity == 1,
               progress_interval = 5,
               alg = alg_default(p),
               optim_solver = optim_solver_default(p),
               maxiterations_ode = 10^6,
               maxiterations_optim = 10^5,
               patience = 10^4,
               tauinv = nothing,
               merge = nothing,
               hessian_template = default_hessian_template(net, p, maxiterations_ode, alg, maxiterations_optim, optim_solver, verbosity),
               weights = weights(net),
               kwargs...)
    checkparams(net, p)
    dx = gradient(net, p)
    gnorminf = maximum(abs, dx)
    _, g!, h!, fgh!, fg! = get_functions(net, maxnorm;
                                         hessian_template = hessian_template,
                                         patience,
                                         losstype,
                                         merge,
                                         tauinv = isa(tauinv, ComponentArray) || isa(tauinv, NamedTuple) ? copy(ComponentArray(tauinv)) : tauinv,
                                         batchsize,
                                         show_progress,
                                         weights,
                                         progress_interval = float(progress_interval),
                                         verbosity)
    lossfunc = u -> loss(net, u; losstype, weights)
    train(net, lossfunc, g!, h!, fgh!, fg!, p;
          verbosity, losstype,
          maxiterations_ode, maxiterations_optim,
          alg, optim_solver,
          kwargs...)
end
n_samples(net::Net) = size(net.input, 2)
n_samples(net::NetI) = 1
function train(net, lossfunc, g!, h!, fgh!, fg!, p;
               alg = alg_default(p),
               optim_solver = optim_solver_default(p),
               maxiterations_ode = 10^6,
               maxiterations_optim = 10^5,
               g_tol = 1e-13,
               verbosity = 1,
               abstol = 1e-6,
               reltol = 1e-6,
               save_everystep = true,
               dense = save_everystep,
               maxT = Inf,
               maxtime_ode = 3*60,
               maxtime_optim = 2*60,
               result = :dict,
               losstype = MSE(), # anything else than MSE() is not tested
               minloss = 2e-32*n_samples(net),
               use_component_arrays = false,
               n_samples_trajectory = 100,
               include = nothing,
               exclude = String[],
               transpose_solution = false, # obsolete
               dt = nothing,
               optim_options = Optim.Options(iterations = maxiterations_optim,
                                               time_limit = maxtime_optim,
                                               f_abstol = -eps(),
                                               f_reltol = -eps(),
                                               x_abstol = -eps(),
                                               x_reltol = -eps(),
                                               allow_f_increases = true,
                                               g_abstol = g_tol,
                                               g_reltol = g_tol,
                                               callback = _ -> converged(fgh!.h.g.l.o)
                                              )
    )
    x = copy(p)
    G = zero(x)
    trajectory = nothing
    tstart = time()
    ode_stopped_by = nothing
    if maxiterations_ode > 0
        if verbosity > 0
            @info "Starting ODE solver $alg."
        end
        if isa(alg, AbstractOptimiser)
            i = 1
            t0 = time()
            if save_everystep
                trajectory = [(0, copy(x))]
            end
            while true
                g!(G, x)
                G .*= 1/n_samples(net) # scale with data
                x .-= apply!(alg, x, G)
                if save_everystep
                    push!(trajectory, (i, copy(x)))
                end
                if i == maxiterations_ode
                    ode_stopped_by = "maxiterations_ode"
                    break
                end
                if time() - t0 > maxtime_ode
                    ode_stopped_by = "maxtime_ode"
                    break
                end
                i += 1
            end
            ode_iterations = i
            ode_time_run = time() - t0
            ode_loss = lossfunc(x)
            if save_everystep
                trajectory = subsample(trajectory, n_samples_trajectory)
            end
            sol = nothing
            ode_x = copy(x)
        else
            odef = ODEFunction(swapsign(g!), jac = swapsign(h!))
            x0 = use_component_arrays ? x : Array(x)
            prob = ODEProblem(odef, x0, (0., Float64(maxT)), (; net,))
            termin = terminator(g!.l.o; maxtime = maxtime_ode,
                                  maxiter = maxiterations_ode,
#                                   minloss = minloss,
                                  losstype)
            sol = solve(prob, alg; dense, save_everystep, abstol, reltol,
                                callback = termin,
                                (dt === nothing ? NamedTuple() : (;dt=dt))...)
            ode_stopped_by = termin.condition.stopped_by
            if sol.t[end] == 0
                println("starting ODE fallback.")
                fallbackalg = CVODE_BDF(linear_solver=:GMRES)
                probfallback = ODEProblem(odef, x0, (0., 1e-2), (; net,))
                solfallback = solve(probfallback, fallbackalg;
                                    dense, save_everystep, abstol, reltol,
                                    callback = termin,
                                    (dt === nothing ? NamedTuple() : (;dt=dt))...)
                @show g!.l(solfallback[end])
                prob = ODEProblem(odef, solfallback[end],
                                  (0., Float64(maxT)), (; net,))
                sol = solve(prob, alg; dense, save_everystep, abstol, reltol,
                                    callback = termin,
                                    (dt === nothing ? NamedTuple() : (;dt=dt))...)
            end
            if sol.t[end] == maxT
                @info "Reached maxT = $maxT."
            end
            ode_time_run = time() - termin.condition.t0[]
            ode_loss = lossfunc(sol.u[end])
            ode_iterations = termin.condition.i[]
            if save_everystep
                trajectory = [(t, copy(x .= sol(t)))
                              for t in max.(sol.t[1], min.(sol.t[end], 10.0.^range(log10(sol.t[1]+1), log10(sol.t[end]+1), n_samples_trajectory) .- 1))]
            end
            x .= sol.u[end]
            ode_x = copy(x)
        end
    else
        ode_time_run = 0.
        ode_iterations = 0
        ode_loss = nothing
        sol = nothing
        ode_x = nothing
    end
    optim_stopped_by = nothing
    if maxiterations_optim > 0
        if !isnothing(g!.l.o.tauinv)
            @warn "Setting tauinv to 1. Consider setting `maxiterations_optim = 0`, if you do not want to use second order optimization."
            g!.l.o.tauinv .= 1
        end
        g!.l.o.k_last_best = g!.l.o.fk # reset patience
        if verbosity > 0
            @info "Starting optimizer $optim_solver."
        end
        if isa(optim_solver, Symbol) # NLopt
            opt = Opt(optim_solver, length(x))
            opt.min_objective = fg!
            opt.maxtime = maxtime_optim
            opt.maxeval = maxiterations_optim
            opt.stopval = minloss
            t0 = time()
            _, xsol, res = NLopt.optimize(opt, x)
            optim_time_run = time() - t0
            optim_iterations = opt.numevals
#             x .= xsol
            if verbosity > 0
                println("NLopt returned with code $res")
            end
            optim_stopped_by = if converged(g!.l.o)
                "patience ($(g!.l.o.patience))"
            elseif res == NLopt.MAXTIME_REACHED
                "maxtime"
            elseif res == NLopt.MAXEVAL_REACHED
                "maxeval"
            elseif res == NLopt.STOPVAL_REACHED
                "minloss"
            elseif res == NLopt.SUCCESS
                "converged"
            else
                string(res)
            end
        else # Optim
            res = Optim.optimize(Optim.only_fgh!(fgh!), x,
                                 optim_solver,
                                 optim_options)
#             x = res.minimizer
            optim_time_run = res.time_run
            optim_iterations = res.iterations
            optim_stopped_by = if converged(g!.l.o)
                "patience"
            elseif optim_time_run ≥ maxtime_optim
                "maxtime"
            elseif Optim.iteration_limit_reached(res)
                "maxeval"
            elseif Optim.f_converged(res)
                "minloss"
            elseif Optim.g_converged(res)
                "mingradnorm"
            elseif Optim.converged(res)
                "converged"
            elseif isa(res.ls_success, Bool) && !res.ls_success
                "line search failed"
            elseif Optim.f_increased(res) && !iteration_limit_reached(res)
                "objective increased between iterations"
            else
                "unkown reason"
            end
        end
        if !isnothing(trajectory) && save_everystep
            push!(trajectory, (trajectory[end][1] + 1, copy(g!.l.o.bestx)))
        end
    else
        res = nothing
        optim_iterations = 0
        optim_time_run = 0
    end
    x = g!.l.o.bestx
    gradient!(G, net, x, weights = g!.l.o.weights) # recompute gradient
    N = isa(net, Net) ? size(net.input, 2) : 1.
    gnorm = maximum(abs, G) / N
    g!(G, x) # recompute with regularized loss
    gnorm_regularized = maximum(abs, G) / N
    loss = lossfunc(x)
    rawres = (; ode = sol, optim = res, ode_x, optim_state = g!.l.o,
              x, init = p, loss, ode_loss, gnorm, gnorm_regularized,
              ode_time_run, ode_iterations, net, optim_time_run,
              converged = converged(g!.l.o),
              optim_stopped_by, ode_stopped_by,
              total_time = time() - tstart, weights = g!.l.o.weights,
              optim_iterations, trajectory, lossfunc, transpose_solution)
    if result == :raw
        rawres
    else
        result2dict(rawres; include, exclude)
    end
end
"""
    train(net, x0::Tuple; num_workers = min(length(x0), Sys.CPU_THREADS),
                          kwargs...)

Train from multiple initial points in parallel.
`kwargs` are passed to the train function for each initial point.
"""
function train(net::Net, x0::Tuple;
               num_workers = min(length(x0), Sys.CPU_THREADS),
               threads_per_worker = 1,
               kwargs...)
    if nworkers() < num_workers
        n = num_workers + 1 - nprocs()
        @info "Adding $n workers."
        addprocs(n, exeflags=["--threads=$threads_per_worker", "--project=$(Pkg.project().path)"])
    end
    @info "Sending net to workers."
    Distributed.remotecall_eval(Main, workers(), :(using MLPGradientFlow))
    expr = :(n = MLPGradientFlow.Net(layers = $(net.layerspec),
                                     input = $(Array(net.input)),
                                     target = $(Array(net.target)),
                                     bias_adapt_input = false);
             res = Dict{String, Any}[]
            )
    Distributed.remotecall_eval(Main, workers(), expr)
    @info "Start training."
    @sync for (i, init) in pairs(x0)
        pid = (i-1)%nworkers() + 1
        @async Distributed.remotecall_eval(Main, workers()[pid:pid], :(push!(res, MLPGradientFlow.train(n, $init; $kwargs...))))
    end
    [@fetchfrom(pid, Main.res) for pid in workers()]
end

function subsample(x, n)
    if length(x) > n
        x[floor.(Int, range(1, length(x), n))]
    else
        x
    end
end
transpose_solution(res, x) = res.transpose_solution ? transpose_params(x) : x
_extract(::Val{:loss}, res) = res.loss
_extract(::Val{:weights}, res) = res.weights
_extract(::Val{:ode_loss}, res) = res.ode_loss
_extract(::Val{:gnorm}, res) = res.gnorm
_extract(::Val{:gnorm_regularized}, res) = res.gnorm_regularized
_extract(::Val{:init}, res) = params2dict(transpose_solution(res, res.init))
_extract(::Val{:x}, res) = params2dict(transpose_solution(res, res.x))
_extract(::Val{:ode_x}, res) = isnothing(res.ode_x) ? nothing : params2dict(transpose_solution(res, res.ode_x))
_extract(::Val{:ode_time_run}, res) = res.ode_time_run
_extract(::Val{:converged}, res) = res.converged
_extract(::Val{:total_time}, res) = res.total_time
_extract(::Val{:ode_iterations}, res) = res.ode_iterations
_extract(::Val{:optim_time_run}, res) = res.optim_time_run
_extract(::Val{:optim_iterations}, res) = res.optim_iterations
_extract(::Val{:input}, res) = isa(res.net, Net) ? Array(res.net.input) : nothing
_extract(::Val{:target}, res) = isa(res.net, Net) ? Array(res.net.target) : nothing
_extract(::Val{:teacher}, res) = hasproperty(res, :teacher) ? res.teacher.p : nothing
_extract(::Val{:layerspec}, res) = _layerextract(res.net)
_layerextract(net::Net) = map(x -> (x[1], string(x[2]), x[3]), net.layerspec)
_layerextract(net::NetI) = map(x -> (x[1], string(x[2]), x[3]), net.student.layerspec)
_extract(::Val{:trajectory}, res) = isnothing(res.trajectory) ? nothing : OrderedDict(t => params2dict(transpose_solution(res, u)) for (t, u) in res.trajectory)
_extract(::Val{:loss_curve}, res) = isnothing(res.trajectory) ? nothing : [res.lossfunc(u) for (_, u) in res.trajectory]
_extract(::Val{:optim_stopped_by}, res) = res.optim_stopped_by
_extract(::Val{:ode_stopped_by}, res) = res.ode_stopped_by
"""
    params2dict(p)

Convert a ComponentArray parameter to a dictionary
"""
params2dict(p) = Dict(string(k) => Array(getproperty(p, k)) for k in keys(p))
"""
    result2dict(res; exclude = String[])

Transform the result to a dictionary.
Some data can be exluded, e.g. exclude = ["trajectory", "loss"] excludes the trajectory and the loss. Fieldnames to exclude are "loss", "gnorm", "gnorm_regularized", "init", "x", "ode_x", "optim_time_run", "optim_iterations", "trajectory".
"""
function result2dict(res;
                     exclude = String[],
                     include = nothing)
    if include === nothing
        include = ["loss", "converged", "optim_stopped_by", "ode_stopped_by", "total_time", "ode_loss", "gnorm", "gnorm_regularized", "init", "x", "ode_x", "ode_time_run", "ode_iterations", "optim_time_run", "optim_iterations", "input", "target", "layerspec", "trajectory", "loss_curve", "teacher", "weights"]
    end
    filter(x -> !isnothing(last(x)),
           Dict(Pair(x, _extract(Val(Symbol(x)), res))
                for x in setdiff(include, exclude)))
end
result2dict(res::Dict; exclude = String[]) = exclude_from(res, exclude)
function exclude_from(d, exclude)
    for s in exclude
        if !haskey(d, s)
            @warn("Key $s not found among keys $(keys(d)).")
        end
    end
    filter(x -> first(x) ∉ exclude, d)
end
"""
    pickle(filename, result; exclude = String[])

Save the `result` of training in `filename` in torch.pickle format.
See also `result2dict`.
"""
function pickle(filename, res; exclude = String[])
    Pickle.Torch.THsave(filename, result2dict(res; exclude))
end

"""
    unpickle(filename)

Loads the results saved with the above function `pickle(filename, result; exclude = String[])`.
"""
function unpickle(filename)
    Pickle.Torch.THload(filename)
end

input_dim(net::Net) = size(net.input, 1) - first(net.layers).bias
random_params(net::NetI; kwargs...) = random_params(net.student; kwargs...)
random_params(net; kwargs...) = random_params(Random.GLOBAL_RNG, net; kwargs...)
function random_params(rng, net::Net; distr_fn = glorot_normal)
    glorot(rng, (input_dim(net),
                 Int.(getproperty.(net.layers, :k))...), eltype(net.input),
           biases = getproperty.(net.layers, :bias); distr_fn)
end
glorot_normal(in, out, T = Float64) = glorot_normal(Random.GLOBAL_RNG, in, out, T)
glorot_uniform(in, out, T = Float64) = glorot_uniform(Random.GLOBAL_RNG, in, out, T)
glorot_normal(rng::AbstractRNG, in, out, T = Float64) = randn(rng, T, out, in) * sqrt(T(2)/(in + out))
glorot_uniform(rng::AbstractRNG, in, out, T = Float64) = (rand(rng, T, out, in) .- T(0.5)) * T(2) * sqrt(T(6)/(in + out))

glorot(dims, T = Float64; kwargs...) = glorot(Random.GLOBAL_RNG, dims, T; kwargs...)
function glorot(rng::AbstractRNG, dims, T = Float64;
                biases = falses(length(dims)-1),
                distr_fn = glorot_normal)
    params([(distr_fn(rng, dims[i], dims[i+1], T), biases[i])
            for i in 1:length(biases)]...)
end
append_bias!(w::AbstractArray) = append_bias!(w, nothing)
append_bias!(w::Tuple) = append_bias!(w...)
append_bias!(w, ::Nothing) = copy(w)
append_bias!(w, b::AbstractVector) = hcat(w, b)
append_bias!(w, is_bias::Bool) = append_bias!(w, is_bias ? zeros(eltype(w), size(w, 1)) : nothing)
"""
    params((w₁, b₁), (w₂, b₂), ...)

Where `wᵢ` is a weight matrix and `bᵢ` is a bias vector or `nothing` (`None` in python).
"""
function params(layers...)
    ComponentArray(; [Symbol(:w, i) => append_bias!(w)
                      for (i, w) in pairs(layers)]...)
end
function params(layers::AbstractDict)
    ComponentArray(; [Symbol(k) => Array(v) for (k, v) in layers]...)
end

end # module
