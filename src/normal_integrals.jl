struct NormalIntegral{X,W}
    x::X
    w::W
end
"""
    NormalIntegral(; N = 600, d = 1, prune = true, threshold = 1e-18)

Callable struct for Gauss-Hermite integration. Uses FastGaussianQuadrature.jl.

# Example
```
julia> integrator = NormalIntegral(d = 1);

julia> integrator(x -> 1.) # normalization of standard normal
0.9999999999999998

julia> integrator(identity) # mean of standard normal
7.578393534606704e-19

julia> integrator(x -> x^2) # variance of standard normal
0.9999999999999969

julia> integrator2d = NormalIntegral(d = 2);

julia> integrator2d(x -> cos(x[1] + x[2])) # integrate some 2D function for x[1] and x[2] iid standard normal
0.3678794411714446

julia> integrator2d(x -> cos(x[1] + x[2]), .5) # integrate with correlation(x[1], x[2]) = 0.5
0.22313016014843348

```
"""
function NormalIntegral(; N = 600, d = 1, prune = true, threshold = 1e-18)
    x, w = gausshermite(N, normalize = true)
    if d > 1
        if prune
            range = minimum(w)/maximum(w)
        end
        x = reshape(collect(Iterators.product([x for _ in 1:d]...)), :)
        w = reshape(prod.(Iterators.product([w for _ in 1:d]...)), :)
        if prune
            idxs = findall(≥(range*maximum(w)), w)
            x = x[idxs]
            w = w[idxs]
        end
    end
    if prune
        idxs = findall(≥(threshold), w)
        x = x[idxs]
        w = w[idxs]
    end
    NormalIntegral(x, w)
end
(g::NormalIntegral)(f) = g.w' * f.(g.x)
(g::NormalIntegral)(f, u) = g.w' * f.(correlate.(g.x, u))

correlate(x, u) = (x[1], u*x[1] + sqrt(1-u^2)*x[2])

_stride_arrayize(g) = NormalIntegral(StaticStrideArray(hcat(collect.(g.x)...)'), StaticStrideArray(g.w))
function (g::NormalIntegral{<:StrideArray})(f, u)
    x, w = g.x, g.w
    tmp = zero(eltype(w))
    u′ = sqrt(1 - u^2)
    @tturbo for i in eachindex(w)
        tmp += w[i] * f(x[i, 1], u*x[i, 1] + u′*x[i, 2])
    end
    tmp
end

# obsolete ?
(g::NormalIntegral{<:NTuple})(f) = sum(g.w .* f.(g.x))
(g::NormalIntegral{<:NTuple})(f, u) = sum(g.w .* f.(correlate.(g.x, u)))

"""
    gauss_hermite_net(net::Net, target_function; kwargs...)

Create from `net` a network with input points and weights obtained from [`NormalIntegral`](@href), to which `kwargs` are passed. The `target_function` is used to compute the target values. Note that in more than 2 input dimensions the number of points is excessively large (with default settings for `NormalIntegral` more than a million points are generated in 3 dimensions).
"""
function gauss_hermite_net(net::Net, target_function; kwargs...)
    net.Din > 2 && @info "Input dimension $(net.Din) larger than 2; this may be very slow!"
    gauss_hermite_quadrature = MLPGradientFlow.NormalIntegral(; d = net.Din, kwargs...)
    x = hcat(collect.(gauss_hermite_quadrature.x)...)
    w = gauss_hermite_quadrature.w
    Net(net, input = x, target = target_function(x), weights = w * length(w))
end

