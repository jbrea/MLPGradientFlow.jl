###
### Quadrature
###

const TOL = (atol = Ref(1e-11), rtol = Ref(1e-11))
"""
    chofv(x)

Change of variables.
See https://giordano.github.io/Cuba.jl/stable/
"""
chofv(x) = ((2x-1)/((1-x)*x), (2x^2-2x+1)/((1-x)^2*x^2))
mvnormal(c, x, sinv) = c*exp(-1/2*(sinv[1, 1]*x[1]^2 + sinv[2, 2]*x[2]^2 + 2*sinv[1, 2]*x[1]*x[2]))
function mvnex_integrand(f, u)
    c = 1/(2π*sqrt(1-u^2))
    sinv = 1/(1-u^2) * [1 -u
                       -u 1]
    function(x, ret)
        y1, dy1 = chofv(x[1])
        y2, dy2 = chofv(x[2])
        if isinf(y1) || isinf(y2)
            ret .= 0.
        else
            factor = mvnormal(c, (y1, y2), sinv) * dy1 * dy2
            for (i, fi) in pairs(f)
                ret[i] = fi(y1, y2) * factor
            end
        end
    end
end
function mvnex_integrand_hcub(f, u)
    c = 1/(2π*sqrt(1-u^2))
    sinv = 1/(1-u^2) * [1 -u
                       -u 1]
    function(x)
        y1, dy1 = chofv(x[1])
        y2, dy2 = chofv(x[2])
        if isinf(y1) || isinf(y2)
            0.
        else
            mvnormal(c, (y1, y2), sinv)*f(y1, y2) * dy1 * dy2
        end
    end
end
mvnex(f::Tuple, u; kwargs...) = cuhre(mvnex_integrand(f, u), 2, length(f); atol = TOL.atol[], rtol = TOL.rtol[], kwargs...)[1]
mvnex(f, u; kwargs...) = mvnex((f,), u; kwargs...)[]
mvnex_hcub(f, u; kwargs...) = hcubature(mvnex_integrand_hcub(f, u), (0., 0.), (1., 1.); atol = TOL.atol[], rtol = TOL.rtol[], kwargs...)[1]
function nex_integrand(f)
    c = 1/sqrt(2π)
    function(x)
        y, dy = chofv(x[1])
        if isinf(y)
            0.
        else
            c*exp(-1/2*y^2) * f(y) * dy
        end
    end
end
nex(f; kwargs...) = hquadrature(nex_integrand(f), 0., 1.; atol = TOL.atol[], rtol = TOL.rtol[], kwargs...)[1]
nex(f::Tuple; kwargs...) = [nex(fi; kwargs...) for fi in f]

###
### Potentials and their derivatives
###

struct Standardizer
    m::Matrix{Float64}
    s::Matrix{Float64}
end
function Standardizer(x)
    N = size(x, 2)
    m = sum(x, dims = 2)/N
    Standardizer(m,
                 sqrt.(sum(abs2, x .- m, dims = 2)/(N-1)))
end
(s::Standardizer)(x) = (x .- s.m) ./ s.s
unstandardize(s, x) = x .* s.s .+ s.m
struct NNet{L, P}
    s1::Standardizer
    s2::Standardizer
    net::L
    p::P
end
function (n::NNet)(x...)
    for i in eachindex(x)
        n.net.input[i] = (x[i] - n.s1.m[i]) / n.s1.s[i]
    end
    a = n.net(n.p)
    for i in eachindex(a)
        a[i] = a[i] * n.s2.s[i] + n.s2.m[i]
    end
    a
end
struct PotentialApproximator
    f
    g #
    dgdru #
    d2gdru #
end
function Base.show(io::IO, p::PotentialApproximator)
    print(io, "PotentialApproximator for $(p.f)")
end
function load_standardizer(g, f, r; path_pretrained = artifact"approximators")
    d = unpickle(joinpath(path_pretrained, "standardizers-$g-$f-$r.pt"))
    Standardizer(d["s1m"], d["s1s"]), Standardizer(d["s2m"], d["s2s"])
end
function load_net(g, f, r, layerspec; path_pretrained = artifact"approximators")
    p = unpickle(joinpath(path_pretrained, "params-$g-$f-$r.pt"))
    s1, s2 = load_standardizer(g, f, r; path_pretrained)
    NNet(s1, s2, Net(layers = layerspec,
                     input = reshape(s1.m, :, 1),
                     target = reshape(s2.m, :, 1),
                     verbosity = 0), params(p))
end
function load_potential_approximator(f, r = 96;
        path_pretrained = artifact"approximators",
        layerspec = ((r, softplus, true), (r, softplus, true), (missing, identity, true)))
    if !isfile(joinpath(path_pretrained, "params-g3-$f-$r.pt"))
        error("No pretrained weights found in path $path_pretrained for activation function $f and number of hidden neurons $r")
    end
    g3net = load_net(g3, f, r, layerspec)
    dgrunet = load_net(dgdru, f, r, layerspec)
    d2grunet = load_net(d2gdru, f, r, layerspec)
    PotentialApproximator(f, g3net, dgrunet, d2grunet)
end
g3(f::PotentialApproximator, r) = g3(f.f, r)
g3(f::Function, r) = nex(x -> f(r*x)^2)
g3(::Val{relu}, r) = r^2/2
g3(::Val{sigmoid2}, r) = 2/pi*asin(r^2/(1 + r^2))
function g3(f::PotentialApproximator, r1, r2, u)
    f.g(r1, r2, u)[1]
end
function g3(f::Function, r1, r2, u)
    if isapprox(u, 1, atol = 1e-5)
        nex(x -> f(r1*x)*f(r2*x))
    elseif isapprox(u, -1, atol = 1e-5)
        nex(x -> f(r1*x)*f(-r2*x))
    else
        mvnex((x, y) -> f(r1*x)*f(r2*y), u)
    end
end
function g3(::Val{relu}, r1, r2, u)
    if isapprox(u, 1, atol = 1e-5)
        r1*r2/2
    elseif isapprox(u, -1, atol = 1e-5)
        0.
    else
        max(0., r1*r2/(2π)*(sqrt(1-u^2)+(π - acos(u))*u))
    end
end
g3(::Val{sigmoid2}, r1, r2, u) = 2/pi*asin(r1*r2*u/(sqrt(1 + r1^2)*sqrt(1 + r2^2)))
function dgdr(f::PotentialApproximator, r)
    dgdr(f.f, r)
end
function dgdr(f::Function, r)
    f′ = deriv(f)
    nex(x -> 2*f(r*x)*f′(r*x)*x)
end
dgdr(::Val{relu}, r) = r
dgdr(::Val{sigmoid2}, r) = (4*r)/(sqrt(2*r^2 + 1)*(π*r^2 + π))
function d2gdr(f::PotentialApproximator, r)
    d2gdr(f.f, r)
end
function d2gdr(f::Function, r; kwargs...)
    f′ = deriv(f)
    f′′ = second_deriv(f)
    nex(x -> 2*x^2*(f(r*x)*f′′(r*x)+f′(r*x)^2))
end
d2gdr(::Val{relu}, r) = 1
d2gdr(::Val{sigmoid2}, r) = -(4*(4*r^4 + r^2 - 1))/(π*(r^2 + 1)^2*(2*r^2 + 1)^(3/2))
function dgdru(f::PotentialApproximator, r1, r2, u)
    f.dgdru(r1, r2, u)
end
function dgdru(f::Function, r1, r2, u)
    f′ = deriv(f)
    if isapprox(u, 1, atol = 1e-5)
        funcs = (x -> f′(r1 * x)*x*f(r2*x),
                 x -> r1 * r2 * f′(r1*x)*f′(r2*x))
        nex(funcs)
    elseif isapprox(u, -1, atol = 1e-5)
        funcs = (x -> f′(r1 * x)*x*f(-r2*x),
                 x -> r1 * r2 * f′(r1*x)*f′(-r2*x))
        nex(funcs)
    else
        funcs = ((x, y) -> f′(r1 * x)*x*f(r2*y),
                 (x, y) -> r1 * r2 * f′(r1*x)*f′(r2*y))
        mvnex(funcs, u)
    end
end
function dgdru(::Val{relu}, r1, r2, u)
    r2/(2π)*(sqrt(1-u^2)+(π - acos(u))*u),
    r1*r2/(2π) * (π - acos(u))
end
function dgdru(::Val{sigmoid2}, r1, r2, u)
    (2*r2*u)/(π*(r1^2 + 1)*sqrt(r1^2*(1 - r2^2*(u^2 - 1)) + r2^2 + 1)),
    (2*r1*r2)/(π*sqrt(r1^2*(1 - r2^2*(u^2 - 1)) + r2^2 + 1))
end
function d2gdru(f::PotentialApproximator, r1, r2, u)
    f.d2gdru(r1, r2, u)
end
function d2gdru(f::Function, r1, r2, u)
    f′ = deriv(f)
    f′′ = second_deriv(f)
    if isapprox(u, 1, atol = 1e-5)
        funcs = (x -> f′′(r1 * x)*x^2*f(r2*x),
                 x -> f′(r1 * x)*x*f′(r2*x)*x,
                 x -> r2 * f′(r1*x)*f′(r2*x) + r1*r2*f′′(r1*x)*x*f′(r2*x),
                 x -> r1 * f′(r1*x)*f′(r2*x) + r1*r2*f′(r1*x)*x*f′′(r2*x),
                 x -> r1^2 * r2^2 * f′′(r1*x) * f′′(r2*x)
                )
        nex(funcs)
    elseif isapprox(u, -1, atol = 1e-5)
        funcs = (x -> f′′(r1 * x)*x^2*f(-r2*x),
                 x -> -f′(r1 * x)*x*f′(-r2*x)*x,
                 x -> r2 * f′(r1*x)*f′(-r2*x) + r1*r2*f′′(r1*x)*x*f′(-r2*x),
                 x -> r1 * f′(r1*x)*f′(-r2*x) - r1*r2*f′(r1*x)*x*f′′(-r2*x),
                 x -> r1^2 * r2^2 * f′′(r1*x) * f′′(-r2*x)
                )
        nex(funcs)
    else
        funcs = ((x, y) -> f′′(r1 * x)*x^2*f(r2*y),
                 (x, y) -> f′(r1 * x)*x*f′(r2*y)*y,
                 (x, y) -> r2 * f′(r1*x)*f′(r2*y) + r1*r2*f′′(r1*x)*x*f′(r2*y),
                 (x, y) -> r1 * f′(r1*x)*f′(r2*y) + r1*r2*f′(r1*x)*y*f′′(r2*y),
                 (x, y) -> r1^2 * r2^2 * f′′(r1*x) * f′′(r2*y)
                )
        mvnex(funcs, u)
    end
end
function d2gdru(::Val{relu}, r1, r2, u)
    0, # 11
    1/(2π)*(sqrt(1-u^2)+(π - acos(u))*u), # 12
    r2/(2π) * (π - acos(u)), # 13
    r1/(2π) * (π - acos(u)), # 23
    r1*r2/(2π*sqrt(1 - u^2)) # 33
end
function d2gdru(::Val{sigmoid2}, r1, r2, u)
    -(6*r1*(r1^2 + 1)*(r2^3 + r2)*u - 2*r1*(3*r1^2 + 1)*r2^3*u^3)/(π*(r1^2 + 1)^2*(r1^2 *(1 - r2^2*(u^2 - 1)) + r2^2 + 1)^(3/2)),
    (2*u)/(π*(r1^2*(1 - r2^2*(u^2 - 1)) + r2^2 + 1)^(3/2)),
    (2*r2*(r2^2 + 1))/(π*(r1^2*(-(r2^2*(u^2 - 1) - 1)) + r2^2 + 1)^(3/2)),
    (2*r1*(r1^2 + 1))/(π*(r2^2*(-(r1^2*(u^2 - 1) - 1)) + r1^2 + 1)^(3/2)),
    (2*r1^3*r2^3*u)/(π*(r1^2*(-(r2^2*(u^2 - 1) - 1)) + r2^2 + 1)^(3/2))
end


function pairwise(f, x, y)
    [f(xi, yi) for xi in eachrow(x), yi in eachrow(y)]
end
pairwise(f, x) = pairwise(f, x, x)
similarity(x, y) = clamp(x'*y/(norm(x)*norm(y)), -1, 1)
function toangles(u)
    phi = similar(u)
    n = max(0, 1 - sum(abs2, u))
    for i in reverse(eachindex(u))
        n += u[i]^2
        if u[i] ≈ 0
            phi[i] = acos(0.)
        else
            phi[i] = acos(min(1, u[i]/sqrt(n)))
        end
    end
    phi
end
toangles(u::AbstractMatrix) = hcat(toangles.(eachcol(u))...)
function tosim(phi)
    s = 1.
    [begin u = cos(ϕ)*s
         s *= sin(ϕ)
         u
     end
     for ϕ in phi]
end
tosim(phi::AbstractMatrix) = hcat(tosim.(eachcol(phi))...)
function last_u(u)
    n = sum(abs2, u)
    n ≈ 1 && return 0.
    sqrt(1 - n)
end

function u_parametrize(x; d = size(x.w1, 2), w = I(d)[1:d-1, :], angles = false)
    u = pairwise(similarity, w, x.w1)
    if angles
        u = toangles(u)
    end
    ComponentVector(a = x.w2, r = norm.(eachrow(x.w1)), u = u)
end
function w_parametrize(x; d = size(x.u, 1) + 1, w = I(d), angles = false)
    if angles
        u = tosim(x.u)
    else
        u = x.u
    end
    w1 = x.r .* (w[1:d-1, :]'*u .+ last_u.(eachcol(u))' .* w[end, :])'
    ComponentVector(; w1, w2 = x.a')
end
simsim(x, y, i, j, rx, ry) = clamp(sum(x[k, i]*y[k, j] for k in axes(x, 1))/(rx*ry), -1, 1)
function norm!(r, w)
    for i in eachindex(r)
        r[i] = zero(eltype(w))
        for k in axes(w, 1)
            r[i] += w[k, i]^2
        end
        r[i] = sqrt(r[i])
    end
    r
end
function _dldw2(i, w2, xt, gr, gs, gt)
    w2[i] * gr[i] +
    (length(w2) > 1 ? sum(w2[k] * gs[i, k] for k in eachindex(w2) if k ≠ i) : 0.) -
    sum(xt.w2[k] * gt[i, k] for k in axes(xt.w1, 2))
end
function dgdr!(f, dgr, r)
    for i in eachindex(r)
        dgr[i] = dgdr(f, r[i])
    end
end
function d2gdr!(f, d2gr, r)
    for i in eachindex(r)
        d2gr[i] = d2gdr(f, r[i])
    end
end
function dgdru!(f, dgru, r1, r2, u)
    for i in axes(u, 1), j in axes(u, 2)
        dgru[i, j][1:2] .= dgdru(f, r1[i], r2[j], u[i, j])
    end
end
function d2gdru!(f, dgru, r1, r2, u)
    for i in axes(u, 1), j in axes(u, 2)
        dgru[i, j][3:7] .= d2gdru(f, r1[i], r2[j], u[i, j])
    end
end
struct NetI{F,T1,T2,T3,T4,T5}
    f::F
    nparams::Int
    xt::T1
    rwt::T5
    rw1::T2
    gr::T2
    dgr::T2
    d2gr::T2
    gt::T3
    gs::T3
    u::T3
    v::T3
    dgru::T4
    dgrv::T4
    loss_xt::Float64
end
function NetI(x::AbstractVector{T1}, xt::AbstractVector{T2}, f;
              transpose = true) where {T1, T2}
    if transpose
        x = transpose_params(x)
        xt = transpose_params(xt)
    end
    k = length(x.w2)
    r = length(xt.w2)
    rwt = zeros(T2, r)
    norm!(rwt, xt.w1)
    loss_xt = 0.
    u = pairwise(similarity, xt.w1')
    for i in 1:r
        loss_xt += xt.w2[i]^2 * g3(f, rwt[i])/2
        for j in i+1:r
            loss_xt += xt.w2[i] * xt.w2[j] * g3(f, rwt[i], rwt[j], u[i, j])
        end
    end
    NetI(f,
         length(x),
         xt,
          rwt,
          zeros(T1, k),
          zeros(T1, k),
          zeros(T1, k),
          zeros(T1, k),
          zeros(T1, k, r),
          ones(T1, k, k),
          zeros(T1, k, r),
          ones(T1, k, k),
          [zeros(T1, 7) for _ in 1:k, __ in 1:r],
          [zeros(T1, 7) for _ in 1:k, __ in 1:k],
          loss_xt
         )
end
function Base.show(io::IO, net::NetI)
    println(io, "Network with nonlinearity \"$(net.f)\" and $(size(net.xt.w1, 1))D gaussian input")
    print(io, "student width = $(size(net.u, 1)), teacher width = $(size(net.u, 2))")
end
_ind(i, D) = (((i-1)÷D)+1, (i-1)%D+1)
function _dc(w1, r1, w2, r2, u, dgru, k, j, l, m, n)
    tmp = 0.
    if k == m
        ∂rₖₗ = _drw(w1, r1, k, l)
        ∂vₖⱼwₖₗ = _dvw(w1, r1, w2, r2, u, k, l, j)
        ∂vₖⱼwₘₙ = _dvw(w1, r1, w2, r2, u, k, n, j)
        ∂rₘₙ = _drw(w1, r1, m, n)
        tmp += (dgru[k, j][3] * ∂rₘₙ + dgru[k, j][5] * ∂vₖⱼwₘₙ) * ∂rₖₗ
        tmp -= dgru[k, j][1] * w1[l, k]*w1[n, m]/r1[k]^3
        tmp += (dgru[k, j][5] * ∂rₘₙ  + dgru[k, j][7] * ∂vₖⱼwₘₙ) * ∂vₖⱼwₖₗ
        tmp += dgru[k, j][2] * (-w2[l, j]*w1[n, m]/(r1[m]^3*r2[j]) -
                                 w1[l, k]/r1[m]^2*∂vₖⱼwₘₙ +
                                 2*w1[l, k]*u[k, j]/r1[m]^3*∂rₘₙ)
        if l == n
            tmp -= dgru[k, j][2]*u[k, j]/r1[k]^2
            tmp += dgru[k, j][1]/r1[k]
        end
    elseif j == m
        ∂rₖₗ = _drw(w1, r1, k, l)
        ∂vₖⱼwₘₙ = (w1[n, k]/(r1[k]*r1[m]) - w1[n, m]*u[k, m]/r1[m]^2)
        ∂vₖⱼwₖₗ = _dvw(w1, r1, w2, r2, u, k, l, j)
        ∂rₘₙ = _drw(w1, r1, m, n)
        tmp += (dgru[k, j][4] * ∂rₘₙ + dgru[k, j][5] * ∂vₖⱼwₘₙ) * ∂rₖₗ
        tmp += (dgru[k, j][6] * ∂rₘₙ + dgru[k, j][7] * ∂vₖⱼwₘₙ) * ∂vₖⱼwₖₗ
        tmp += dgru[k, j][2] * (-w1[l, m]/(r1[k]*r1[m]^2)*∂rₘₙ-w1[l, k]/r1[k]^2*∂vₖⱼwₘₙ)
        if l == n && r1 === r2
            tmp += dgru[k, j][2]/(r1[k]*r1[m])
        end
    end
    tmp
end
function hessian(net::NetI, x; transpose = true, kwargs...)
    h = zeros(net.nparams, net.nparams)
    if transpose
        idxs = [reshape(reshape(1:length(x.w1), :, size(x.w1, 1))', :);
                length(x.w1)+1:length(x)]
        x = transpose_params(x)
    end
    hessian!(h, net, x; kwargs...)
    if transpose
        h .= h[idxs, idxs]
    end
    h
end
function hessian!(h, net::NetI, x; forward = true, backprop = true, kwargs...)
    hess_u!(net.f, h, x, net; forward, backprop)
    h .*= 2
end
function hess_u!(f, H, x, net; backprop = true, forward = true)
    (; rw1, xt, rwt, gr, gs, u, v, dgr, d2gr, dgru, dgrv) = net
    forward && _loss(f, x, net)
    backprop && backprop!(net, x; forward = false)
    d2gdru!(f, dgru, rw1, rwt, u)
    d2gdru!(f, dgrv, rw1, rw1, v)
    d2gdr!(f, d2gr, rw1)
    w1 = x.w1; w2 = x.w2
    D = size(w1, 1)
    N1 = length(w1);
    for i in eachindex(w2), j in eachindex(w2)
        if i == j
            H[N1+i, N1+j] = gr[i]
        else
            H[N1+i, N1+j] = gs[i, j]
        end
    end
    for i in eachindex(w1), j in eachindex(w2)
        tmp = 0.
        i1, i2 = _ind(i, D)
        if i1 == j
            tmp = w2[j]*dgr[j]*w1[i]/rw1[j]
            for k in axes(xt.w1, 2)
                tmp -= xt.w2[k]*_c(w1, rw1, xt.w1, rwt, u, dgru, i1, i2, k)
            end
            for k in axes(w1, 2)
                k == i1 && continue
                tmp += w2[k]*_c(w1, rw1, w1, rw1, v, dgrv, i1, i2, k)
            end
        else
            tmp = w2[i1]*_c(w1, rw1, w1, rw1, v, dgrv, i1, i2, j)
        end
        H[i, N1 + j] = H[N1 + j, i] = tmp
    end
    for i in eachindex(w1), j in eachindex(w1)
        tmp = 0.
        j > i && continue
        i1, i2 = _ind(i, D)
        j1, j2 = _ind(j, D)
        if i1 == j1
            tmp = 1/2*w2[i1]^2*(d2gr[i1]*_drw(w1, rw1, i1, i2)*_drw(w1, rw1, j1, j2) -
                                dgr[i1]*w1[i]*w1[j]/rw1[i1]^3)
            if i2 == j2
                tmp += 1/2*w2[i1]^2*dgr[i1]/rw1[i1]
            end
            for k in eachindex(w2)
                k == j1 && continue
                tmp += w2[k]*w2[j1]*_dc(w1, rw1, w1, rw1, v, dgrv, i1, k, i2, j1, j2)
            end
            for k in eachindex(xt.w2)
                tmp -= xt.w2[k]*w2[j1]*_dc(w1, rw1, xt.w1, rwt, u, dgru, i1, k, i2, j1, j2)
            end
        else
            tmp = w2[i1]*w2[j1]*_dc(w1, rw1, w1, rw1, v, dgrv, i1, j1, i2, j1, j2)
        end
        H[i, j] = H[j, i] = tmp
    end
end
"""
∂rᵢ/∂wᵢⱼ
"""
_drw(w, r, i, j) = w[j, i]/r[i]
"""
∂uᵢₖ/∂wᵢⱼ
"""
_dvw(wi, ri, wj, rj, u, i, j, k) = (wj[j, k]/(ri[i]*rj[k]) - wi[j, i]*u[i, k]/ri[i]^2)
function _c(wi, ri, wj, rj, u, dgru, i, j, k)
    (dgru[i, k][1] * _drw(wi, ri, i, j) +
     dgru[i, k][2] * _dvw(wi, ri, wj, rj, u, i, j, k))
end
function gradient(net::NetI, x; transpose = true, kwargs...)
    checkparams(net, x)
    dx = zero(x)
    if transpose
        x = transpose_params(x)
        dxt = transpose_params(dx)
    else
        dxt = dx
    end
    gradient!(dxt, net, x; kwargs...)
    if transpose
        dx .= transpose_params(dxt)
    end
    dx
end
function gradient!(dx, net::NetI, x; forward = true, kwargs...)
    grad_u!(dx, x, net; forward)
    dx .*= 2
end
function backprop!(net::NetI, x; forward = true)
    (; rw1, rwt, u, v, dgr, dgru, dgrv) = net
    f = net.f
    forward && _loss(f, x, net)
    dgdr!(f, dgr, rw1)
    dgdru!(f, dgru, rw1, rwt, u)
    dgdru!(f, dgrv, rw1, rw1, v)
end
function grad_u!(dx, x, net; forward = true, backprop = true)
    backprop && backprop!(net, x; forward)
    (; rw1, xt, rwt, gr, gt, gs, u, v, dgr, dgru, dgrv) = net
    w1 = x.w1; w2 = x.w2
    for i in eachindex(w2)
        dx.w2[i] = _dldw2(i, w2, xt, gr, gs, gt)
        for j in axes(w1, 1)
            dx.w1[j, i] = 1/2*w2[i]^2*dgr[i]/rw1[i]*w1[j, i]
            for k in axes(xt.w1, 2)
                dx.w1[j, i] -= w2[i]*xt.w2[k]*_c(w1, rw1, xt.w1, rwt, u, dgru, i, j, k)
            end
            for k in axes(w1, 2)
                k == i && continue
                dx.w1[j, i] += w2[i]*w2[k]*_c(w1, rw1, w1, rw1, v, dgrv, i, j, k)
            end
        end
    end
end
function loss(net::NetI, x; transpose = true,
              losstype = :mse, forward = true, kwargs...)
    if transpose
        x = transpose_params(x)
    end
    _loss(net, x; losstype, forward)
end
function _loss(net::NetI, x;
               forward = false, losstype = :mse, kwargs...)
    res = _loss(net.f, x, net; forward) + net.loss_xt
    losstype == :rmse && return sqrt(2*res)
    losstype == :mse && return 2*res
    error("Losstype $losstype unkown.")
end
function _loss(f, x, net; forward = true)
    (; rw1, xt, rwt, gr, gt, gs, u, v) = net
    w1 = x.w1; w2 = x.w2
    norm!(rw1, w1)
    res = 0.
    for i in eachindex(w2)
        if forward
            gr[i] = g3(f, rw1[i])
        end
        res += w2[i]^2 * gr[i]/2
        for j in axes(xt.w1, 2)
            if forward
                u[i, j] = simsim(w1, xt.w1, i, j, rw1[i], rwt[j])
                gt[i, j] = g3(f, rw1[i], rwt[j], u[i, j])
            end
            res -= w2[i] * xt.w2[j] * gt[i, j]
        end
        for j in i+1:length(w2)
            if forward
                v[i, j] = v[j, i] = simsim(w1, w1, i, j, rw1[i], rw1[j])
                gs[i, j] = gs[j, i] = g3(f, rw1[i], rw1[j], v[i, j])
            end
            res += w2[i]*w2[j] * gs[i, j]
        end
    end
    res
end

###
### Training
###

transpose_params(p) = ComponentVector(w1 = Array(p.w1'), w2 = reshape(p.w2, :)')
function train(net::NetI, p;
               maxnorm = Inf,
               loss_scale = 1.,
               verbosity = 1,
               losstype = :mse,
               transpose = true,
               minloss = 0,
               kwargs...)
    checkparams(net, p)
    if transpose
        x = transpose_params(p)
    else
        x = p
    end
    _, g!, h!, fgh!, fg! = get_functions(net, maxnorm;
                                         hessian_template = nothing,
                                         scale = loss_scale,
                                         batcher = FullBatch(1:1),
                                         losstype = :mse,
                                         verbosity)
    lossfunc = u -> loss(net, u; losstype, transpose = false)
    train(net, lossfunc, g!, h!, fgh!, fg!, x;
          loss_scale, verbosity, losstype, transpose_solution = transpose,
          use_component_arrays = true,
          minloss,
          kwargs...)
end
