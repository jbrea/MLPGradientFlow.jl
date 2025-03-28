struct ϕ end
struct ∂rϕ end
struct ∂bϕ end
struct ∂r∂rϕ end
struct ∂r∂bϕ end
struct ∂b∂bϕ end
struct ϕϕ end
struct ∂rϕϕ end
struct ∂bϕϕ end
struct ∂r∂rϕϕ end
struct ∂r∂bϕϕ end
struct ∂b∂bϕϕ end
struct ∂r₁ϕϕ end
struct ∂b₁ϕϕ end
struct ∂uϕϕ end
struct ∂r₁∂r₁ϕϕ end
struct ∂r₁∂r₂ϕϕ end
struct ∂r₁∂b₁ϕϕ end
struct ∂r₂∂b₁ϕϕ end
struct ∂r₁∂uϕϕ end
struct ∂u∂uϕϕ end
struct ∂b₁∂b₁ϕϕ end
struct ∂b₁∂b₂ϕϕ end
struct ∂b₁∂uϕϕ end

function integrate end

for (kind, func) in ((:ϕ, :(f(h))),
                     (:∂rϕ, :(x * f′(h))),
                     (:∂bϕ, :(f′(h))),
                     (:∂r∂rϕ, :(x^2 * f′′(h))),
                     (:∂r∂bϕ, :(x * f′′(h))),
                     (:∂b∂bϕ, :(f′′(h))),
                     (:ϕϕ, :(f(h)^2)),
                     (:∂rϕϕ, :(2 * x * f(h) * f′(h))),
                     (:∂bϕϕ, :(2 * f(h) * f′(h))),
                     (:∂r∂rϕϕ, :(2 * x^2 * (f(h) * f′′(h) + f′(h)^2))),
                     (:∂r∂bϕϕ, :(2 * x * (f(h) * f′′(h) + f′(h)^2))),
                     (:∂b∂bϕϕ, :(2 * (f(h) * f′′(h) + f′(h)^2))),
                    )
    eval(quote
             function integrate(::$kind, w, _x, f, r, b)
                 tmp = zero(eltype(w))
                 f′ = deriv(f)
                 f′′ = second_deriv(f)
                 @tturbo for i in eachindex(w)
                     x = _x[i]
                     h = r * x + b
                     tmp += w[i] * $func
                 end
                 tmp
             end
         end)
end
for (kind, func) in ((:ϕϕ, :(f1(h1) * f2(h2))),
                     (:∂r₁ϕϕ, :(x * f1′(h1) * f2(h2))),
                     (:∂b₁ϕϕ, :(f1′(h1) * f2(h2))),
                     (:∂uϕϕ, :(r1 * r2 * f1′(h1) * f2′(h2))),
                     (:∂r₁∂r₁ϕϕ, :(x^2 * f1′′(h1) * f2(h2))),
                     (:∂r₁∂r₂ϕϕ, :(x * y * f1′(h1) * f2′(h2))),
                     (:∂r₁∂b₁ϕϕ, :(x * f1′′(h1) * f2(h2))),
                     (:∂r₂∂b₁ϕϕ, :(y * f1′(h1) * f2′(h2))),
                     (:∂r₁∂uϕϕ, :(r2 * (f1′(h1) + r1 * x * f1′′(h1)) * f2′(h2))),
                     (:∂u∂uϕϕ, :(r1^2 * r2^2 * f1′′(h1) * f2′′(h2))),
                     (:∂b₁∂b₁ϕϕ, :(f1′′(h1) * f2(h2))),
                     (:∂b₁∂b₂ϕϕ, :(f1′(h1) * f2′(h2))),
                     (:∂b₁∂uϕϕ, :(r1 * r2 * f1′′(h1) * f2′(h2))),
                    )
    eval(quote
             function integrate(::$kind, w, _x, f1, r1, b1, f2, r2, b2, u, u′ = sqrt(1 - u^2))
                 tmp = zero(eltype(w))
                 f1′ = deriv(f1)
                 f1′′ = second_deriv(f1)
                 f2′ = deriv(f2)
                 f2′′ = second_deriv(f2)
                 @tturbo for i in eachindex(w)
                     x = _x[i, 1]
                     y = u * x + u′ * _x[i, 2]
                     h1 = r1 * x + b1
                     h2 = r2 * y + b2
                     tmp += w[i] * $func
                 end
                 tmp
             end
         end)
end

#
function ϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    BvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end
function _∂r₁XϕϕBvN(d1, d2, r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    -r1 * b1/n1^3 * d1(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)) +
    (u*r2/(n1*n2) - u*r1^2*r2/(n1^3*n2)) * d2(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end
function ∂r₁ϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    _∂r₁XϕϕBvN(∂hBvN, ∂rBvN, r1, b1, r2, b2, u, n1, n2)
end
function ∂b₁ϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    1/n1 * ∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end
function ∂uϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    r1 * r2/ (n1 * n2) * ∂rBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end
function ∂r₁∂r₁ϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    (-b1/n1^3 + 3*r1^2*b1/n1^5) * ∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)) -
    r1 * b1/n1^3 * _∂r₁XϕϕBvN(∂h∂hBvN, ∂h∂rBvN, r1, b1, r2, b2, u, n1, n2) -
    (3 * r1* r2 * u)/(n2 * n1^5) * ∂rBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)) +
    (u*r2/(n1*n2) - u*r1^2*r2/(n1^3*n2)) * _∂r₁XϕϕBvN(∂h∂rBvN, ∂r∂rBvN, r1, b1, r2, b2, u, n1, n2)
end
function ∂r₁∂r₂ϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    -r1 * b1/n1^3 * ∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)) +
    (u*r2/(n1*n2) - u*r1^2*r2/(n1^3*n2)) * ∂rBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end
function ∂r₁∂b₁ϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    -r1 /n1^3 * ∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)) + 1/n1 * (
    -r1 * b1/n1^3 * ∂h∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)) +
    (u*r2/(n1*n2) - u*r1^2*r2/(n1^3*n2)) * ∂r∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)))
end
function ∂r₂∂b₁ϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    1/n1 * (
    -r2 * b2/n2^3 * ∂k∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)) +
    (u*r1/(n1*n2) - u*r2^2*r1/(n2^3*n1)) * ∂r∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)))
end
function ∂r₁∂uϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    (r2/ (n1 * n2) - r1^2 * r2/ (n1^3 * n2))  * ∂rBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)) + r1 * r2/ (n1 * n2) * (-r1 * b1/n1^3) * ∂h∂rBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2)) + r1 * r2/ (n1 * n2) * (u*r2/(n1*n2) - u*r1^2*r2/(n1^3*n2)) * ∂r∂rBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end
function ∂u∂uϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    (r1 * r2/ (n1 * n2))^2 * ∂r∂rBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end
function ∂b₁∂b₁ϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    1/n1^2 * ∂h∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end
function ∂b₁∂b₂ϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    1/(n1 * n2) * ∂h∂kBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end
function ∂b₁∂uϕϕBvN(r1, b1, r2, b2, u, n1 = sqrt(1 + r1^2), n2 = sqrt(1 + r2^2))
    1/n1 * r1 * r2/ (n1 * n2) * ∂r∂hBvN(b1/n1, b2/n2, u * r1 * r2/ (n1 * n2))
end

# 1-point (not performance critical)
function integrate(::ϕϕ, w, _x, f1::typeof(normal_cdf), r, b)
    n = sqrt(1 + r^2)
    normal_cdf(b/n) - 2*owent(b/n, 1/sqrt(1 + 2r^2))
end
function integrate(::ϕ, w, _x, f1::typeof(normal_cdf), r, b)
    normal_cdf(b/sqrt(1 + r^2))
end
function integrate(::∂rϕ, w, _x, f1::typeof(normal_cdf), r, b)
    n = sqrt(1 + r^2)
    -b*r/n^3 * normal_pdf(b/n)
end
function integrate(::∂bϕ, w, _x, f1::typeof(normal_cdf), r, b)
    normal_pdf(b/sqrt(1 + r^2))/sqrt(1 + r^2)
end

# 2-point (performance critical)
function integrate(::ϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                f2::typeof(normal_cdf), r2, b2,
                                u, u′ = nothing)
    ϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂r₁ϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                   f2::typeof(normal_cdf), r2, b2,
                                   u, u′ = nothing)
    ∂r₁ϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂b₁ϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                   f2::typeof(normal_cdf), r2, b2,
                                   u, u′ = nothing)
    ∂b₁ϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂uϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                  f2::typeof(normal_cdf), r2, b2,
                                  u, u′ = nothing)
    ∂uϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂r₁∂b₁ϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                   f2::typeof(normal_cdf), r2, b2,
                                   u, u′ = nothing)
    ∂r₁∂b₁ϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂r₁∂r₁ϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                   f2::typeof(normal_cdf), r2, b2,
                                   u, u′ = nothing)
    ∂r₁∂r₁ϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂r₂∂b₁ϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                   f2::typeof(normal_cdf), r2, b2,
                                   u, u′ = nothing)
    ∂r₂∂b₁ϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂r₁∂uϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                   f2::typeof(normal_cdf), r2, b2,
                                   u, u′ = nothing)
    ∂r₁∂uϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂u∂uϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                  f2::typeof(normal_cdf), r2, b2,
                                  u, u′ = nothing)
    ∂u∂uϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂b₁∂b₁ϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                  f2::typeof(normal_cdf), r2, b2,
                                  u, u′ = nothing)
    ∂b₁∂b₁ϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂b₁∂b₂ϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                  f2::typeof(normal_cdf), r2, b2,
                                  u, u′ = nothing)
    ∂b₁∂b₂ϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂b₁∂uϕϕ, w, _x, f1::typeof(normal_cdf), r1, b1,
                                  f2::typeof(normal_cdf), r2, b2,
                                  u, u′ = nothing)
    ∂b₁∂uϕϕBvN(r1, b1, r2, b2, u)
end

function integrate(::ϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                f2::typeof(sigmoid2), r2, b2,
                                u, u′ = nothing)
    n1 = sqrt(1 + r1^2)
    n2 = sqrt(1 + r2^2)
    4*ϕϕBvN(r1, b1, r2, b2, u, n1, n2) - 2*(normal_cdf(b1/n1) + normal_cdf(b2/n2)) + 1
end
function integrate(::∂r₁ϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                   f2::typeof(sigmoid2), r2, b2,
                                   u, u′ = nothing)
    n1 = sqrt(1 + r1^2)
    4*∂r₁ϕϕBvN(r1, b1, r2, b2, u, n1) + 2*r1 * b1/n1^3 * normal_pdf(b1/n1)
end
function integrate(::∂b₁ϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                   f2::typeof(sigmoid2), r2, b2,
                                   u, u′ = nothing)
    n1 = sqrt(1 + r1^2)
    4 * ∂b₁ϕϕBvN(r1, b1, r2, b2, u, n1) - 2/n1 * normal_pdf(b1/n1)
end
function integrate(::∂uϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                  f2::typeof(sigmoid2), r2, b2,
                                  u, u′ = nothing)
    4 * ∂uϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂r₁∂uϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                   f2::typeof(sigmoid2), r2, b2,
                                   u, u′ = nothing)
    4 * ∂r₁∂uϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂u∂uϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                  f2::typeof(sigmoid2), r2, b2,
                                  u, u′ = nothing)
    4 * ∂u∂uϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂b₁∂uϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                  f2::typeof(sigmoid2), r2, b2,
                                  u, u′ = nothing)
    4 * ∂b₁∂uϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂b₁∂b₁ϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                  f2::typeof(sigmoid2), r2, b2,
                                  u, u′ = nothing)
    n1 = sqrt(1 + r1^2)
    4 * ∂b₁∂b₁ϕϕBvN(r1, b1, r2, b2, u, n1) + 2 * b1/n1^3 * normal_pdf(b1/n1)
end
function integrate(::∂b₁∂b₂ϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                  f2::typeof(sigmoid2), r2, b2,
                                  u, u′ = nothing)
    4 * ∂b₁∂b₂ϕϕBvN(r1, b1, r2, b2, u)
end
function integrate(::∂r₁∂b₁ϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                   f2::typeof(sigmoid2), r2, b2,
                                   u, u′ = nothing)
    n1 = sqrt(1 + r1^2)
    4 * ∂r₁∂b₁ϕϕBvN(r1, b1, r2, b2, u) + 2*r1/n1^3 * (1 - b1^2/n1^2) * normal_pdf(b1/n1)
end
function integrate(::∂r₁∂r₁ϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                   f2::typeof(sigmoid2), r2, b2,
                                   u, u′ = nothing)
    n1 = sqrt(1 + r1^2)
    4 * ∂r₁∂r₁ϕϕBvN(r1, b1, r2, b2, u) + ((2*b1 - 4*b1*r1^2)/n1^5 + 2*r1^2*b1^3/n1^7) * normal_pdf(b1/n1)
end
function integrate(::∂r₂∂b₁ϕϕ, w, _x, f1::typeof(sigmoid2), r1, b1,
                                   f2::typeof(sigmoid2), r2, b2,
                                   u, u′ = nothing)
    4*∂r₂∂b₁ϕϕBvN(r1, b1, r2, b2, u)
end

struct NetI{T,TE,S,G1,G2,TB1,TB2,TBT1,TBT2}
    teacher::TE
    student::S
    g1::G1
    g2::G2
    gr::Vector{T}
    gs::Matrix{T}
    gt::Matrix{T}
    u::Matrix{T}
    v::Matrix{T}
    w1::Matrix{T}
    b1::TB1
    w2::Matrix{T}
    b2::TB2
    w1t::Matrix{T}
    b1t::TBT1
    w2t::Matrix{T}
    b2t::TBT2
    rw1::Vector{T}
    rwt::Vector{T}
    dgr::Matrix{T}
    dgru::Array{T,3}
    dgrv::Array{T,3}
    loss_correction_t1::T
    s::Base.RefValue{T}
    g0::Vector{T}
    dg0::Matrix{T}
    loss_correction_t2::T
end
input_dim(net::NetI) = size(net.w1t, 2)
function Base.show(io::IO, net::NetI)
    println(io, "Network with $(input_dim(net))D gaussian input")
    println(io, "student: $(net.student.layerspec)")
    print(io, "teacher: $(net.teacher.net.layerspec)")
end
function _norm!(r, w)
    for i in eachindex(r)
        r[i] = 0
        for j in axes(w, 2)
            r[i] += w[i, j]^2
        end
        r[i] = sqrt(r[i])
    end
    r
end
_weights_and_biases(w, is_bias) = w[:, 1:end-is_bias], is_bias ? w[:, end] : nothing
_weights_and_biases!(w, ::Nothing, p) = copyto!(w, p), nothing
_bias(::Nothing, i) = 0
_bias(b, i) = b[i]
_similarity(x, y, i, j, rx, ry) = clamp(sum(x[i, k]*y[j, k] for k in axes(x, 2))/(rx*ry), -1, 1)
function _weights_and_biases!(w, b, p)
    n = length(w)
    copyto!(w, 1, p, 1, n),
    copyto!(b, 1, p, n+1, length(b))
end
function NetI(teacher, student; T = eltype(student.input),
               g1 = _stride_arrayize(NormalIntegral(d = 1)),
               g2 = _stride_arrayize(NormalIntegral(d = 2)))
    @assert isa(teacher, TeacherNet)
    @assert isa(student, Net)
    @assert length(teacher.net.layerspec) == 2
    @assert length(student.layerspec) == 2
    @assert teacher.net.layerspec[2][2] == identity
    @assert student.layerspec[2][2] == identity
    @assert teacher.net.layerspec[2][1] == 1
    @assert student.layerspec[2][1] == 1
    t_spec = teacher.net.layerspec
    s_spec = student.layerspec
    r = t_spec[1][1]
    k = s_spec[1][1]
    _p = random_params(student)
    w1, b1 = _weights_and_biases(_p.w1, s_spec[1][3])
    w2, b2 = _weights_and_biases(_p.w2, s_spec[2][3])
    w1t, b1t = _weights_and_biases(teacher.p.w1, t_spec[1][3])
    w2t, b2t = _weights_and_biases(teacher.p.w2, t_spec[2][3])
    rwt = _norm!(zeros(r), w1t)
    ft = t_spec[1][2]
    loss_correction_t1 = zero(T)
    loss_correction_t2 = zero(T)
    for i in 1:r
        loss_correction_t2 += w2t[i]^2 * integrate(ϕϕ(), g1.w, g1.x, ft, rwt[i], _bias(b1t, i))
        if !isnothing(b1t)
            loss_correction_t1 += w2t[i] * integrate(ϕ(), g1.w, g1.x, ft, rwt[i], _bias(b1t, i))
        end
        for j in i+1:r
            u = _similarity(w1t, w1t, i, j, rwt[i], rwt[j])
            loss_correction_t2 += 2*w2t[i] * w2t[j] * integrate(ϕϕ(), g2.w, g2.x, ft, rwt[i], _bias(b1t, i), ft, rwt[j], _bias(b1t, j), u, sqrt(1-u^2))
        end
    end
    NetI(teacher, student, g1, g2,
          zeros(T, k),
          zeros(T, k, k),
          zeros(T, k, r),
          zeros(T, k, r),
          zeros(T, k, k),
          w1, b1, w2, b2,
          w1t, b1t, w2t, b2t,
          zeros(T, k),
          rwt,
          zeros(T, k, 5),
          zeros(T, k, r, 11),
          zeros(T, k, k, 13),
          loss_correction_t1,
          Ref(zero(T)),
          zeros(T, k),
          zeros(T, k, 5),
          loss_correction_t2)
end
function _hessian!(H, net::NetI{T}, x; backprop = true, forward = true, second_order = true, weights = nothing, losstype = MSE(), derivs = nothing) where T
    (; g1, g2, w1, b1, w2, b2, rw1, rwt, w1t, b1t, w2t, b2t, g0, gr, gs, u, v, dg0, dgr, dgru, dgrv) = net
    forward && _loss(net, x)
    backprop && backward!(net, x; forward = false)
    f = net.student.layerspec[1][2]
    ft = net.teacher.net.layerspec[1][2]
    delta_b = _bias(b2, 1) - _bias(b2t, 1)
    if second_order
        if !isnothing(b1) || delta_b ≠ 0
            d2g0drb!(dg0, g1, f, rw1, b1)
        end
        d2gdrb!(dgr, g1, f, rw1, b1)
        d2gdrub!(dgru, g2, f, rw1, b1, ft, rwt, b1t, u)
        d2gdrvb!(dgrv, g2, f, rw1, b1, v)
    end
    K, Din = size(w1)
    N0 = length(w1)
    N1 = N0 + (!isnothing(b1)) * size(w1, 1)
    for i in eachindex(w2), j in eachindex(w2) # second derivative of w2
        if i == j
            H[N1+i, N1+i] = gr[i]
        else
            H[N1+i, N1+j] = gs[i, j]
        end
    end
    for i in eachindex(w1), j in eachindex(w2) # mixed derivatives
        tmp = 0.
        i1, i2 = _ind(i, Din)
        if i1 == j
            tmp = w2[j]*dgr[j, 1]*w1[i1, i2]/rw1[j]
            for k in axes(w1t, 1)
                tmp -= w2t[k]*_c(w1, rw1, w1t, rwt, u, dgru, i1, i2, k)
            end
            for k in axes(w1, 1)
                k == i1 && continue
                tmp += w2[k]*_c(w1, rw1, w1, rw1, v, dgrv, i1, i2, k)
            end
            if delta_b ≠ 0
                tmp += delta_b * w1[i1, i2]/rw1[i1]*dg0[i1, 1]
            end
        else
            tmp = w2[i1]*_c(w1, rw1, w1, rw1, v, dgrv, i1, i2, j)
        end
        _i = (i2-1)*K+i1
        H[_i, N1 + j] = H[N1 + j, _i] = tmp
    end
    for i in eachindex(w1), j in eachindex(w1) # second derivatives of w1
        tmp = zero(T)
        j > i && continue
        i1, i2 = _ind(i, Din)
        j1, j2 = _ind(j, Din)
        if i1 == j1
            tmp = 1/2*w2[i1]^2*(dgr[i1, 3]*∂rᵢ_∂wᵢⱼ(w1, rw1, i1, i2)*∂rᵢ_∂wᵢⱼ(w1, rw1, j1, j2) -
                                dgr[i1, 1]*w1[i1, i2]*w1[j1, j2]/rw1[i1]^3)
            if i2 == j2
                tmp += 1/2*w2[i1]^2*dgr[i1, 1]/rw1[i1]
            end
            for k in eachindex(w2)
                k == j1 && continue
                tmp += w2[k]*w2[j1]*_c2(w1, rw1, w1, rw1, v, dgrv, i1, k, i2, j1, j2)
            end
            for k in eachindex(w2t)
                tmp -= w2t[k]*w2[j1]*_c2(w1, rw1, w1t, rwt, u, dgru, i1, k, i2, j1, j2)
            end
            if delta_b ≠ 0
                tmp += delta_b * w2[i1] * w1[i1, i2] * w1[j1, j2]/rw1[i1]^2 * (-dg0[i1, 1]/rw1[i1] + dg0[i1, 3])
                if i2 == j2
                    tmp += delta_b * w2[i1]/rw1[i1]*dg0[i1, 1]
                end
            end
        else
            tmp = w2[i1]*w2[j1]*_c2(w1, rw1, w1, rw1, v, dgrv, i1, j1, i2, j1, j2)
        end
        _i = (i2-1)*K+i1
        _j = (j2-1)*K+j1
        H[_i, _j] = H[_j, _i] = tmp
    end
    if !isnothing(b2)
        H[end, end] = 1
    end
    if delta_b ≠ 0
        if !isnothing(b2)
            for i in eachindex(w2)
                H[N1+i, end] = H[end, N1+i] = g0[i]
            end
            for i in eachindex(w1)
                i1, i2 = _ind(i, Din)
                _i = (i2-1)*K+i1
                H[_i, end] = H[end, _i] = w2[i1] * w1[i1, i2]/rw1[i1]*dg0[i1, 1]
            end
        end
    end
    if !isnothing(b1)
        for i in eachindex(b1)
            for j in eachindex(w2) # b1 w2
                tmp = zero(T)
                if i == j
                    tmp = delta_b * dg0[i, 2] + w2[i] * dgr[i, 2]
                    for k in eachindex(w2t)
                        tmp -= w2t[k] * dgru[i, k, 3]
                    end
                    for k in eachindex(w2)
                        k == i && continue
                        tmp += w2[k] * dgrv[i, k, 3]
                    end
                else
                    tmp = w2[i] * dgrv[i, j, 3]
                end
                H[N0 + i, N1 + j] = H[N1 + j, N0 + i] = tmp
            end
            if !isnothing(b2) # b1 b2
                H[N0 + i, end] = H[end, N0 + i] = w2[i] * dg0[i, 2]
            end
            for j in eachindex(b1) # b1 b1
                tmp = zero(T)
                if i == j
                    tmp = delta_b * w2[i] * dg0[i, 4] + w2[i]^2/2 * dgr[i, 4]
                    for k in eachindex(w2t)
                        tmp -= w2[i] * w2t[k] * dgru[i, k, 9]
                    end
                    for k in eachindex(w2)
                        k == i && continue
                        tmp += w2[i] * w2[k] * dgrv[i, k, 9]
                    end
                else
                    tmp = w2[i] * w2[j] * dgrv[i, j, 10]
                end
                H[N0 + i, N0 + j] = H[N0 + j, N0 + i] = tmp
            end
            for j in eachindex(w1)
                j1, j2 = _ind(j, Din)
                drdw = ∂rᵢ_∂wᵢⱼ(w1, rw1, j1, j2)
                if i == j1
                    tmp = w2[i]^2/2 * ∂rᵢ_∂wᵢⱼ(w1, rw1, i, j2) * dgr[i, 5] + delta_b * w2[i] * drdw * dg0[i, 5]
                    for k in eachindex(w2t)
                        tmp -= w2[i] * w2t[k] * (drdw * dgru[i, k, 10] +
                                                 ∂uᵢₖ_∂wᵢⱼ(w1, rw1, w1t, rwt, u, i, j2, k) * dgru[i, k, 11])
                    end
                    for k in eachindex(w2)
                        k == i && continue
                        tmp += w2[i] * w2[k] * (drdw * dgrv[i, k, 11] +
                                                ∂uᵢₖ_∂wᵢⱼ(w1, rw1, w1, rw1, v, i, j2, k) * dgrv[i, k, 13])
                    end
                else
                    tmp = w2[i] * w2[j1] * (drdw * dgrv[i, j1, 12] +
                                            ∂uᵢₖ_∂wᵢⱼ(w1, rw1, w1, rw1, v, j1, j2, i) * dgrv[i, j1, 13])
                end
                _j = (j2-1)*K+j1
                H[N0 + i, _j] = H[_j, N0 + i] = tmp
            end
        end
    end
    H .*= 2
    H
end
function backward!(net::NetI, x; forward = true)
    (; g1, g2, b1, b1t, b2, b2t, dg0, rw1, dgr, dgru, rwt, u, dgrv, v) = net
    f = net.student.layerspec[1][2]
    ft = net.teacher.net.layerspec[1][2]
    forward && _loss(net, x)
    if !isnothing(b1) || _bias(b2, 1) - _bias(b2t, 1) ≠ 0
        dg0drb!(dg0, g1, f, rw1, b1)
    end
    dgdrb!(dgr, g1, f, rw1, b1)
    dgdrub!(dgru, g2, f, rw1, b1, ft, rwt, b1t, u)
    dgdrvb!(dgrv, g2, f, rw1, b1, v)
end
function _gradient!(dx, net::NetI, x; backward = true, forward = true, weights = nothing, derivs = nothing, losstype = MSE())
    (; w1, b1, w2, b2, w1t, w2t, b2t, rw1, rwt, gr, gt, gs, dg0, dgr, dgru, dgrv, u, v) = net
    forward && _loss(net, x)
    backward && backward!(net, x; forward = false)
    dw1 = getweights(net.student.layers[1], dx)
    dw2 = getweights(net.student.layers[2], dx)
    for i in eachindex(w2)
        dw2[i] = 2*_dldw2(i, w2, w2t, gr, gs, gt)
        for j in axes(w1, 2)
            dw1[i, j] = w2[i]^2*dgr[i, 1]/rw1[i]*w1[i, j]
            for k in eachindex(w2t)
                dw1[i, j] -= 2*w2[i]*w2t[k]*_c(w1, rw1, w1t, rwt, u, dgru, i, j, k)
            end
            for k in eachindex(w2)
                k == i && continue
                dw1[i, j] += 2*w2[i]*w2[k]*_c(w1, rw1, w1, rw1, v, dgrv, i, j, k)
            end
        end
    end
    delta_b = _bias(b2, 1) - _bias(b2t, 1)
    if !isnothing(b2)
        dw2[end] = 2 * (net.s[] - net.loss_correction_t1 + delta_b)
    end
    if delta_b ≠ 0
        for i in eachindex(w2)
            dw2[i] += 2 * delta_b * net.g0[i]
        end
        for i in axes(w1, 1), j in axes(w1, 2)
            dw1[i, j] += 2 * delta_b * w2[i] * w1[i, j]/rw1[i]*dg0[i, 1]
        end
    end
    if !isnothing(b1)
        for i in eachindex(b1)
            dw1[i, end] = 2*delta_b*w2[i]*dg0[i, 2] + w2[i]^2 * dgr[i, 2]
            for k in eachindex(w2t)
                dw1[i, end] -= 2*w2[i]*w2t[k]*dgru[i, k, 3]
            end
            for k in eachindex(w2)
                k == i && continue
                dw1[i, end] += 2*w2[i]*w2[k]*dgrv[i, k, 3]
            end
        end
    end
    dx
end
loss(net::NetI, x; kwargs...) = _loss(net, x; kwargs...)
function __loss(net::NetI, x; forward = true, weights = nothing, losstype = MSE(), derivs = nothing)
    (; w1, b1, w2, b2, w1t, b1t, w2t, b2t, rw1, rwt, gr, gt, gs, g0, u, v) = net
    f = net.student.layerspec[1][2]
    ft = net.teacher.net.layerspec[1][2]
    _weights_and_biases!(w1, b1, getweights(net.student.layers[1], x))
    _weights_and_biases!(w2, b2, getweights(net.student.layers[2], x))
    _norm!(rw1, w1)
    res = 0.
    for i in eachindex(w2)
        if forward
            gr[i] = integrate(ϕϕ(), net.g1.w, net.g1.x, f, rw1[i], _bias(b1, i))
        end
        res += w2[i]^2 * gr[i]
        for j in axes(w1t, 1)
            if forward
                u[i, j] = _u = _similarity(w1, w1t, i, j, rw1[i], rwt[j])
                gt[i, j] = integrate(ϕϕ(), net.g2.w, net.g2.x, f, rw1[i], _bias(b1, i), ft, rwt[j], _bias(b1t, j), _u, sqrt(1 - _u^2))
            end
            res -= 2 * w2[i] * w2t[j] * gt[i, j]
        end
        for j in i+1:length(w2)
            if forward
                v[i, j] = v[j, i] = _similarity(w1, w1, i, j, rw1[i], rw1[j])
                gs[i, j] = gs[j, i] = integrate(ϕϕ(), net.g2.w, net.g2.x, f, rw1[i], _bias(b1, i), f, rw1[j], _bias(b1, j), v[i, j], sqrt(1-v[i,j]^2))
            end
            res += 2 * w2[i] * w2[j] * gs[i, j]
        end
    end
    delta_b = (_bias(b2, 1) - _bias(b2t, 1))
    net.s[] = 0
    if delta_b ≠ 0
        s = 0
        for i in eachindex(w2)
            if forward
                g0[i] = integrate(ϕ(), net.g1.w, net.g1.x, f, rw1[i], _bias(b1, i))
            end
            s += w2[i] * g0[i]
        end
        net.s[] = s
    end
    res + net.loss_correction_t2 + delta_b^2 + 2 * delta_b * (net.s[] - net.loss_correction_t1)
end


function _dldw2(i, w2, w2t, gr, gs, gt)
    w2[i] * gr[i] +
    (length(w2) > 1 ? sum(w2[k] * gs[i, k] for k in eachindex(w2) if k ≠ i) : 0.) -
    sum(w2t[k] * gt[i, k] for k in eachindex(w2t))
end
@inline function dg0drb!(dg0, g1, f, r, b)
    for i in eachindex(r)
        dg0[i, 1] = integrate(∂rϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
        if !isnothing(b)
            dg0[i, 2] = integrate(∂bϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
        end
    end
end
@inline function dgdrb!(dgr, g1, f, r, b)
    for i in eachindex(r)
        dgr[i, 1] = integrate(∂rϕϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
        if !isnothing(b)
            dgr[i, 2] = integrate(∂bϕϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
        end
    end
end
@inline function dgdrub!(dgru, g2, f1, r1, b1, f2, r2, b2, u)
    for i in axes(u, 1), j in axes(u, 2)
        _u =  u[i, j]
        _u′ = sqrt(1 - u[i, j]^2)
        dgru[i, j, 1] = integrate(∂r₁ϕϕ(), g2.w, g2.x, f1, r1[i], _bias(b1, i), f2, r2[j], _bias(b2, j), _u, _u′)
        dgru[i, j, 2] = integrate(∂uϕϕ(), g2.w, g2.x, f1, r1[i], _bias(b1, i), f2, r2[j], _bias(b2, j), _u, _u′)
        if !isnothing(b1)
            dgru[i, j, 3] = integrate(∂b₁ϕϕ(), g2.w, g2.x, f1, r1[i], _bias(b1, i), f2, r2[j], _bias(b2, j), _u, _u′)
        end
    end
end
@inline function dgdrvb!(dgru, g2, f, r, b, u)
    for i in axes(u, 1), j in axes(u, 2)
        j ≤ i && continue
        _u =  u[i, j]
        _u′ = sqrt(1 - _u^2)
        dgru[i, j, 1] = integrate(∂r₁ϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _u, _u′)
        dgru[j, i, 1] = integrate(∂r₁ϕϕ(), g2.w, g2.x, f, r[j], _bias(b, j), f, r[i], _bias(b, i), _u, _u′)
        dgru[i, j, 2] = dgru[j, i, 2] = integrate(∂uϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _u, _u′)
        if !isnothing(b)
            dgru[i, j, 3] = integrate(∂b₁ϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _u, _u′)
            dgru[j, i, 3] = integrate(∂b₁ϕϕ(), g2.w, g2.x, f, r[j], _bias(b, j), f, r[i], _bias(b, i), _u, _u′)
        end
    end
end
@inline function d2g0drb!(dg0, g1, f, r, b)
    for i in eachindex(r)
        dg0[i, 3] = integrate(∂r∂rϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
        if !isnothing(b)
            dg0[i, 4] = integrate(∂b∂bϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
            dg0[i, 5] = integrate(∂r∂bϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
        end
    end
end
@inline function d2gdrb!(dgr, g1, f, r, b)
    for i in eachindex(r)
        dgr[i, 3] = integrate(∂r∂rϕϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
        if !isnothing(b)
            dgr[i, 4] = integrate(∂b∂bϕϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
            dgr[i, 5] = integrate(∂r∂bϕϕ(), g1.w, g1.x, f, r[i], _bias(b, i))
        end
    end
end
@inline function d2gdrvb!(dgrv, g2, f, r, b, v)
    for i in axes(v, 1), j in axes(v, 2)
        j ≤ i && continue
        _v = v[i, j]
        _v′ = sqrt(1 - _v^2)
        dgrv[i, j, 4] = integrate(∂r₁∂r₁ϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _v, _v′)
        dgrv[j, i, 4] = integrate(∂r₁∂r₁ϕϕ(), g2.w, g2.x, f, r[j], _bias(b, j), f, r[i], _bias(b, i), _v, _v′)
        dgrv[i, j, 5] = dgrv[j, i, 5] = integrate(∂r₁∂r₂ϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _v, _v′)
        dgrv[i, j, 6] = dgrv[j, i, 7] = integrate(∂r₁∂uϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _v, _v′)
        dgrv[j, i, 6] = dgrv[i, j, 7] = integrate(∂r₁∂uϕϕ(), g2.w, g2.x, f, r[j], _bias(b, j), f, r[i], _bias(b, i), _v, _v′)
        dgrv[i, j, 8] = dgrv[j, i, 8] = integrate(∂u∂uϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _v, _v′)
        if !isnothing(b)
            dgrv[i, j, 9] = integrate(∂b₁∂b₁ϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _v, _v′)
            dgrv[j, i, 9] = integrate(∂b₁∂b₁ϕϕ(), g2.w, g2.x, f, r[j], _bias(b, j), f, r[i], _bias(b, i), _v, _v′)
            dgrv[i, j, 10] = dgrv[j, i, 10] = integrate(∂b₁∂b₂ϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _v, _v′)
            dgrv[i, j, 11] = integrate(∂r₁∂b₁ϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _v, _v′)
            dgrv[j, i, 11] = integrate(∂r₁∂b₁ϕϕ(), g2.w, g2.x, f, r[j], _bias(b, j), f, r[i], _bias(b, i), _v, _v′)
            dgrv[i, j, 12] = integrate(∂r₂∂b₁ϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _v, _v′)
            dgrv[j, i, 12] = integrate(∂r₂∂b₁ϕϕ(), g2.w, g2.x, f, r[j], _bias(b, j), f, r[i], _bias(b, i), _v, _v′)
            dgrv[i, j, 13] = integrate(∂b₁∂uϕϕ(), g2.w, g2.x, f, r[i], _bias(b, i), f, r[j], _bias(b, j), _v, _v′)
            dgrv[j, i, 13] = integrate(∂b₁∂uϕϕ(), g2.w, g2.x, f, r[j], _bias(b, j), f, r[i], _bias(b, i), _v, _v′)
        end
    end
end
@inline function d2gdrub!(dgru, g2, f1, r1, b1, f2, r2, b2, u)
    for i in axes(u, 1), j in axes(u, 2)
        _u = u[i, j]
        _u′ = sqrt(1 - _u^2)
        dgru[i, j, 4] = integrate(∂r₁∂r₁ϕϕ(), g2.w, g2.x, f1, r1[i], _bias(b1, i), f2, r2[j], _bias(b2, j), _u, _u′)
        dgru[i, j, 6] = integrate(∂r₁∂uϕϕ(), g2.w, g2.x, f1, r1[i], _bias(b1, i), f2, r2[j], _bias(b2, j), _u, _u′)
        dgru[i, j, 8] = integrate(∂u∂uϕϕ(), g2.w, g2.x, f1, r1[i], _bias(b1, i), f2, r2[j], _bias(b2, j), _u, _u′)
        if !isnothing(b1)
            dgru[i, j, 9] = integrate(∂b₁∂b₁ϕϕ(), g2.w, g2.x, f1, r1[i], _bias(b1, i), f2, r2[j], _bias(b2, j), _u, _u′)
            dgru[i, j, 10] = integrate(∂r₁∂b₁ϕϕ(), g2.w, g2.x, f1, r1[i], _bias(b1, i), f2, r2[j], _bias(b2, j), _u, _u′)
            dgru[i, j, 11] = integrate(∂b₁∂uϕϕ(), g2.w, g2.x, f1, r1[i], _bias(b1, i), f2, r2[j], _bias(b2, j), _u, _u′)
        end
    end
end

# helpers

_ind(i, D) = (((i-1)÷D)+1, (i-1)%D+1)
∂rᵢ_∂wᵢⱼ(w, r, i, j) = w[i, j]/r[i]
∂uᵢₖ_∂wᵢⱼ(wi, ri, wj, rj, u, i, j, k) = (wj[k, j]/(ri[i]*rj[k]) - wi[i, j]*u[i, k]/ri[i]^2)
function _c(wi, ri, wj, rj, u, dgru, i, j, k)
    dgru[i, k, 1] * ∂rᵢ_∂wᵢⱼ(wi, ri, i, j) +
    dgru[i, k, 2] * ∂uᵢₖ_∂wᵢⱼ(wi, ri, wj, rj, u, i, j, k)
end
function _c2(w1, r1, w2, r2, u, dgru, k, j, l, m, n)
    tmp = 0.
    if k == m
        ∂rₖₗ = ∂rᵢ_∂wᵢⱼ(w1, r1, k, l)
        ∂vₖⱼwₖₗ = ∂uᵢₖ_∂wᵢⱼ(w1, r1, w2, r2, u, k, l, j)
        ∂vₖⱼwₘₙ = ∂uᵢₖ_∂wᵢⱼ(w1, r1, w2, r2, u, k, n, j)
        ∂rₘₙ = ∂rᵢ_∂wᵢⱼ(w1, r1, m, n)
        tmp += (dgru[k, j, 4] * ∂rₘₙ + dgru[k, j, 6] * ∂vₖⱼwₘₙ) * ∂rₖₗ
        tmp -= dgru[k, j, 1] * w1[k, l]*w1[m, n]/r1[k]^3
        tmp += (dgru[k, j, 6] * ∂rₘₙ  + dgru[k, j, 8] * ∂vₖⱼwₘₙ) * ∂vₖⱼwₖₗ
        tmp += dgru[k, j, 2] * (-w2[j, l]*w1[m, n]/(r1[m]^3*r2[j]) -
                                 w1[k, l]/r1[m]^2*∂vₖⱼwₘₙ +
                                 2*w1[k, l]*u[k, j]/r1[m]^3*∂rₘₙ)
        if l == n
            tmp -= dgru[k, j, 2]*u[k, j]/r1[k]^2
            tmp += dgru[k, j, 1]/r1[k]
        end
    elseif j == m
        ∂rₖₗ = ∂rᵢ_∂wᵢⱼ(w1, r1, k, l)
        ∂vₖⱼwₘₙ = (w1[k, n]/(r1[k]*r1[m]) - w1[m, n]*u[k, m]/r1[m]^2)
        ∂vₖⱼwₖₗ = ∂uᵢₖ_∂wᵢⱼ(w1, r1, w2, r2, u, k, l, j)
        ∂rₘₙ = ∂rᵢ_∂wᵢⱼ(w1, r1, m, n)
        tmp += (dgru[k, j, 5] * ∂rₘₙ + dgru[k, j, 6] * ∂vₖⱼwₘₙ) * ∂rₖₗ
        tmp += (dgru[k, j, 7] * ∂rₘₙ + dgru[k, j, 8] * ∂vₖⱼwₘₙ) * ∂vₖⱼwₖₗ
        tmp += dgru[k, j, 2] * (-w1[m, l]/(r1[k]*r1[m]^2)*∂rₘₙ-w1[k, l]/r1[k]^2*∂vₖⱼwₘₙ)
        if l == n && r1 === r2
            tmp += dgru[k, j, 2]/(r1[k]*r1[m])
        end
    end
    tmp
end
