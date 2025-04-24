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
function Base.show(io::IO, n::NormalIntegral)
    println(io, "(Gauss-Hermite) NormalIntegral in $(isa(n.x[1], Tuple) ? length(n.x[1]) : size(n.x, 2))D with $(length(n.w)) points")
end
(g::NormalIntegral)(f) = g.w' * f.(g.x)
(g::NormalIntegral)(f, u) = g.w' * f.(correlate.(g.x, u))

correlate(x, u) = (x[1], u*x[1] + sqrt(1-u^2)*x[2])

_stride_arrayize(g) = NormalIntegral(StaticStrideArray(hcat(collect.(g.x)...)'),
                                     StaticStrideArray(g.w))
function (g::NormalIntegral{<:StrideArray})(f, u)
    x, w = g.x, g.w
    tmp = zero(eltype(w))
    u′ = sqrt(1 - u^2)
    @tturbo for i in eachindex(w)
        tmp += w[i] * f(x[i, 1], u*x[i, 1] + u′*x[i, 2])
    end
    tmp
end

"""
    gauss_hermite_net(target_function, net::Net; kwargs...)

Create from `net` a network with input points and weights obtained from [`NormalIntegral`](@ref), to which `kwargs` are passed. The `target_function` is used to compute the target values. Note that in more than 2 input dimensions the number of points is excessively large (with default settings for [`NormalIntegral`](@ref) more than a million points are generated in 3 dimensions).

### Example
```
julia> net = gauss_hermite_net(x -> reshape(x[1, :] .^ 2, 1, :),
                               Net(layers = ((5, softplus, false),
                                             (1, identity, true)), Din = 2))
```
"""
function gauss_hermite_net(target_function, net::Net; kwargs...)
    net.Din > 2 && @info "Input dimension $(net.Din) larger than 2; this may be very slow!"
    gauss_hermite_quadrature = MLPGradientFlow.NormalIntegral(; d = net.Din, kwargs...)
    x = hcat(collect.(gauss_hermite_quadrature.x)...)
    w = gauss_hermite_quadrature.w
    Net(net, input = x, target = target_function(x), weights = w * length(w))
end

function BvN(h, k, r)
    r′ = sqrt(1 - r^2)
    r == 1 && return normal_cdf(min(h, k))
    r == -1 && return h + k < 0 ? 0 : normal_cdf(h) + normal_cdf(k) - 1
    h == k == 0 && return .5 - 2 * owent(0, (1-r)/r′)
    h == k && return normal_cdf(h) - 2 * owent(h, (1-r)/r′)
    if h * k > 0 || (h*k == 0 && (h > 0 || k > 0))
        return 1/2 * (normal_cdf(h) + normal_cdf(k)) - owent(h, (k - r*h)/(h * r′)) - owent(k, (h - r * k)/(k * r′))
    end
    if h * k < 0 || (h*k == 0 && (h < 0 || k < 0))
        return 1/2 * (normal_cdf(h) + normal_cdf(k) - 1) - owent(h, (k - r*h)/(h * r′)) - owent(k, (h - r * k)/(k * r′))
    end
    owent(h, k/h) + owent(k, h/k) -
    owent(h, (k - r*h)/(h * r′)) - owent(k, (h - r * k)/(k * r′)) +
    normal_cdf(h) * normal_cdf(k)
end
∂hBvN(h, k, r) = normal_pdf(h) * normal_cdf((k - r * h)/sqrt(1 - r^2))
∂kBvN(h, k, r) = ∂hBvN(k, h, r)
∂rBvN(h, k, r) = inv2π * 1/sqrt(1 - r^2) * exp(-(h^2 - 2*r*h*k + k^2)/(2*(1 - r^2)))
∂h∂hBvN(h, k, r) = -h * ∂hBvN(h, k, r) - r/sqrt(1 - r^2) * normal_pdf(h) * normal_pdf((k - r * h)/sqrt(1 - r^2))
∂h∂kBvN(h, k, r) = 1/sqrt(1 - r^2) * normal_pdf(k) * normal_pdf((h - r * k)/sqrt(1 - r^2))
∂h∂rBvN(h, k, r) = (-h/(1 - r^2) + r*k/(1 - r^2)) * ∂rBvN(h, k, r)
∂k∂hBvN(h, k, r) = ∂h∂kBvN(h, k, r)
∂k∂kBvN(h, k, r) = ∂h∂hBvN(k, h, r)
∂k∂rBvN(h, k, r) = ∂h∂rBvN(k, h, r)
∂r∂hBvN(h, k, r) = ∂h∂rBvN(h, k, r)
∂r∂kBvN(h, k, r) = ∂k∂rBvN(h, k, r)
∂r∂rBvN(h, k, r) = (h * k*(1 + r^2) - r * (h^2 + k^2 + r^2 - 1))/(1 - r^2)^2 * ∂rBvN(h, k, r)

