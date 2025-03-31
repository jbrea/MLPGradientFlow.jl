using MLPGradientFlow
using ForwardDiff, ComponentArrays, Statistics, SpecialFunctions, Random
using Distributed
using Test

Random.seed!(123)

@testset "activation functions" begin
    import MLPGradientFlow: deriv, second_deriv, A_mul_B!, alloc_a′, alloc_a′′
    for f in (g, exp, sigmoid, square, relu, gelu, tanh, tanh_fast, softplus, sigmoid2, Poly(.2, .3, -.3, .4), selu, silu, normal_cdf)
        @info "testing activation function $f."
        inp = [-.2, 3.]'
        y = f.(inp)
        ff = f == gelu ? x -> x/2*(1 + erf(x/sqrt(2))) : f == sigmoid2 ? x -> erf(x/sqrt(2)) : f == normal_cdf ? x -> (1 + erf(x/sqrt(2)))/2 : f
        @test ff.(inp) ≈ f.(inp)
        y′ = ForwardDiff.derivative.(ff, inp)
        y′′ = ForwardDiff.derivative.(x -> ForwardDiff.derivative(ff, x), inp)
        w = ones(1, 1)
        a = zeros(1, 2)
        a′ = alloc_a′(f, Float64, 1, 2, false)
        a′′ = alloc_a′′(f, Float64, 1, 2)
        A_mul_B!(a, f, w, inp)
        @test a ≈ y
        A_mul_B!(a, a′, f, w, inp)
        @test a ≈ y
        @test a′ ≈ y′
        @test deriv(f).(inp) ≈ y′
        A_mul_B!(a, a′, a′′, f, w, inp)
        @test a ≈ y
        @test a′ ≈ y′
        @test a′′ ≈ y′′
        @test second_deriv(f).(inp) ≈ y′′
    end
end

@testset "Net" begin
    import MLPGradientFlow: random_params, Net
    n = Net(layers = ((2, sigmoid, false), (4, identity, true)),
            input = randn(2, 3), target = randn(4, 3))
    x = random_params(n)
    @test length(x) == n.nparams
end

to_ntuple(w, b, i) = (; Symbol(:w, i) => w, Symbol(:b, i) => b)
to_ntuple(w, ::Nothing, i) = (; Symbol(:w, i) => w)
function flatten(θ)
    ComponentArray(merge([to_ntuple(w, b, i) for (i, (w, b)) in enumerate(θ)]...))
end
function unflatten(θ)
    tuple([(getproperty(θ, Symbol(:w, i)),
            hasproperty(θ, Symbol(:b, i)) ? getproperty(θ, Symbol(:b, i)) : nothing)
           for i in 1:length(filter(x -> string(x)[1] == 'w', keys(θ)))]...)
end
fw_layer(x, ((w, b), f)::Tuple{Any, typeof(softmax)}) = softmax(w*x .+ (b === nothing ? 0 : b))
fw_layer(x, ((w, b), f)) = f.(w*x .+ (b === nothing ? 0 : b))
fw_forward(x, f, input) = foldl(fw_layer, zip(x, f), init = input)
fw_forward(x::ComponentArray, f, input) = fw_forward(unflatten(x), f, input)
fw_layer2(x, (w, f)) = f.(w*x)
fw_forward2(x, f, input) = foldl(fw_layer2, zip(getproperty.(Ref(x), keys(x)), f), init = input)
fw_forward3(x, f, input) = foldl(fw_layer2, zip(x, f), init = input)
merge_loss(x, i, j) = sum(abs2, x[1][1][i, :] - x[1][1][j, :]) + (x[1][2][i] - x[1][2][j])^2
merge_loss(x::ComponentArray, i, j) = sum(abs2, x.w1[i, :] - x.w1[j, :]) + (x.b1[i] - x.b1[j])^2
function fw_lossfunc(input, target, f;
                     sizes = nothing, losstype = :mse,
                     scale = 1/size(input, 2),
                     merge = nothing,
                     maxnorm = Inf,
                     forward = sizes !== nothing ? fw_forward3 : fw_forward)
    x -> begin
        if sizes !== nothing
            off = 0
            x = [begin
                     lw = prod(s)
                     w = reshape(x[off+1:off+lw], s...)
                     off += lw
                     w
                 end
                 for s in sizes]
        end
        if losstype == :crossentropy
            output = softmax(forward(x, f, input))
            res = -sum(getindex.(Ref(log.(output)), target, 1:size(input, 2)))*scale
        else
            res = sum(abs2, forward(x, f, input) - target)*scale
        end
        if maxnorm < Inf
            nx = MLPGradientFlow.weightnorm(x)
            if nx > maxnorm
                res += (nx - maxnorm)^3/3
            end
        end
        if merge !== nothing
            i, j = merge.pair
            λ = merge.lambda/2
            res += λ * merge_loss(x, i, j)
        end
        res
    end
end

@testset "test helper" begin
    input = randn(2, 3)
    target = randn(2, 3)
    θ = ((randn(4, 2), randn(4)),
         (randn(2, 4), nothing),
         (randn(2, 2), randn(2)),
         (randn(2, 2), nothing))
    f = (identity, identity, identity, identity)
    @test unflatten(flatten(θ)) == θ
    @test fw_forward(θ, f, input) ≈ fw_forward(flatten(θ), f, input)
end

@testset "loss" begin
    import MLPGradientFlow: loss, params, Net, g, sigmoid, forward!
    input = randn(2, 3)
    target = randn(2, 3)
    θ = ((randn(4, 2), randn(4)), (randn(2, 4), nothing), (randn(2, 2), randn(2)))
    f = (sigmoid, identity, g)
    n = Net(; layers = [(size(w, 1), f, b !== nothing)
                      for (w, f, b) in zip(first.(θ), f, last.(θ))],
            input, target)
    x = params(θ...)
    @test fw_forward(θ, f, input) ≈ forward!(n, x)
    fw_loss = fw_lossfunc(input, target, f)
    @test fw_loss(θ) ≈ loss(n, x)
    @test fw_loss(θ) ≈ loss(n, flatten(θ))
    nx = MLPGradientFlow.weightnorm(x)
    @test fw_lossfunc(input, target, f, maxnorm = nx/2)(flatten(θ)) ≈ loss(n, x, maxnorm = nx/2)
    pred = n(x)
    @test sum(abs2, pred - target)/size(target, 2) ≈ loss(n, x)
    θ_wrong = ((randn(4, 2), randn(4)), (randn(2, 4), nothing), (randn(2, 2), nothing))
    @test_throws AssertionError loss(n, params(θ_wrong...))
    @test_throws AssertionError loss(n, randn(10))
    @test loss(n, Array(x)) == loss(n, x)
    input = randn(2, 5)
    target = randn(2, 5)
    # the following test is not supposed to work anymore
#     @test fw_forward(θ, f, input) ≈ forward!(n, x, input, derivs = 0)
    fw_loss2 = fw_lossfunc(input, target, f)
    @test fw_loss2(θ) ≈ loss(n, x; input, target)
    merge = (layer = 1, pair = (1, 2), lambda = 1e-2)
    fw_loss3 = fw_lossfunc(input, target, f; merge)
    @test fw_loss3(θ) ≈ loss(n, x; input, target, merge)
end

@testset "gradient" begin
    import MLPGradientFlow: loss, params, Net, g, sigmoid, gradient, gradient!
    input = randn(2, 5)
    target = randn(2, 5)
    θ = ((randn(4, 2), randn(4)),
         (randn(2, 4), nothing),
         (randn(2, 2), randn(2)),
         (randn(2, 2), nothing))
    f = (sigmoid, identity, g, identity)
    n = Net(; layers = [(size(w, 1), f, b !== nothing)
                      for (w, f, b) in zip(first.(θ), f, last.(θ))],
            input, target)
    x = params(θ...)
    fw_loss = fw_lossfunc(input, target, f, scale = 1)
    G = gradient(n, x)
    @test G ≈ ForwardDiff.gradient(fw_loss, flatten(θ))/size(input, 2)
    G2 = gradient(n, flatten(θ))
    @test G ≈ G2
    fw_loss2 = fw_lossfunc(input, target, f)
    @test G ≈ ForwardDiff.gradient(fw_loss2, flatten(θ))
    oldG = copy(G)
    gradient!(G, n, x)
    @test oldG == G/size(input, 2)
    merge = (layer = 1, pair = (1, 2), lambda = 1e-1)
    fw_loss3 = fw_lossfunc(input, target, f; merge)
    G = gradient(n, x; merge)
    @test G ≈ ForwardDiff.gradient(fw_loss3, flatten(θ))
    nx = MLPGradientFlow.weightnorm(x)
    fw_loss4 = fw_lossfunc(input, target, f, maxnorm = nx/2)
    @test gradient(n, x, maxnorm = nx/2) ≈ ForwardDiff.gradient(fw_loss4, flatten(θ))
end

@testset "hessian" begin
    import MLPGradientFlow: params, Net, g, sigmoid, hessian, hessian!,
                            hessian_spectrum
    input = randn(2, 3)
    target = randn(5, 3)
    θ = ((randn(3, 2), randn(3)),
         (randn(2, 3), nothing),
         (randn(2, 2), randn(2)),
         (randn(5, 2), nothing))
    f = (sigmoid, identity, g, identity)
    n = Net(; layers = [(size(w, 1), f, b !== nothing)
                      for (w, f, b) in zip(first.(θ), f, last.(θ))],
            input, target)
    x = params(θ...)
    fw_loss = fw_lossfunc(input, target, f, scale = 1)
    H = hessian(n, x)
    @test H ≈ ForwardDiff.hessian(fw_loss, flatten(θ))/size(input, 2)
    merge = (layer = 1, pair = (1, 2), lambda = 1e-2)
    fw_loss3 = fw_lossfunc(input, target, f; merge)
    H = hessian(n, x; merge)
    @test H ≈ ForwardDiff.hessian(fw_loss3, flatten(θ))
    e, v = hessian_spectrum(n, x)
    @test size(v) == (31, 31)
    nx = MLPGradientFlow.weightnorm(x)
    fw_loss4 = fw_lossfunc(input, target, f, maxnorm = nx/2)
    @test hessian(n, x, maxnorm = nx/2) ≈ ForwardDiff.hessian(fw_loss4, flatten(θ))
end

@testset "ode" begin
    import MLPGradientFlow: params, Net, g, sigmoid, train, get_functions, Hessian
    input = randn(2, 5)
    target = randn(1, 5)
    θ = ((randn(2, 2), nothing),
         (randn(1, 2), nothing))
    f = (sigmoid, identity)
    n = Net(; layers = [(size(w, 1), f, b !== nothing)
                      for (w, f, b) in zip(first.(θ), f, last.(θ))],
            input, target)
    x = params(θ...)
    f!, g!, h!, fgh!, fg! = get_functions(n, 70, hessian_template = Hessian(x))
    r = function(x)
        nx = sum(abs2, x)/(2*length(x))
        nx > 70 ? (nx-70)^3/3 : 0.
    end
    @test f!(x) ≈ loss(n, x) * size(input, 2)
    xl = 100x
    @test f!(xl) ≈ (loss(n, xl) + r(xl)) * size(input, 2)
    fw_loss = fw_lossfunc(input, target, f, scale = 1)
    fw_loss_r = x -> fw_loss(x) + r(x) * size(input, 2)
    G = zero(x)
    g!(G, x)
    @test G ≈ ForwardDiff.gradient(fw_loss_r, flatten(θ))
    g!(G, xl)
    @test G ≈ ForwardDiff.gradient(fw_loss_r, 100*flatten(θ))
    H = zeros(length(x), length(x))
    h!(H, x)
    @test H ≈ ForwardDiff.hessian(fw_loss_r, flatten(θ))
    h!(H, xl)
    @test H ≈ ForwardDiff.hessian(fw_loss_r, 100*flatten(θ))
    x = random_params(n)
    v = fgh!(true, G, H, x)
    tmpG = copy(G)
    tmpH = copy(H)
    @test v == f!(x)
    g!(G, x)
    @test G == tmpG
    h!(H, x)
    @test H == tmpH
    res = train(n, x, maxtime_ode = 1, maxtime_optim = 1, verbosity = 1, result = :raw)
    @test loss(n, res.init) > loss(n, res.ode.u[end])
    @test loss(n, res.ode.u[end]) ≥ loss(n, res.x) - sqrt(eps())
end

@testset "distributed" begin
    import MLPGradientFlow: Net, g, sigmoid, train, random_params, params, gradient, hessian, RK4
    # without bias
    n = Net(layers = ((2, g, false), (1, sigmoid, false)),
            input = rand(2, 10), target = rand(1, 10))
    x = ntuple(_ -> random_params(n), 2)
    res = train(n, x[1], maxtime_ode = Inf, maxtime_optim = Inf,
                alg = RK4(),
                maxiterations_ode = 3, maxiterations_optim = 0)
    res_distr = train(n, x, maxtime_ode = Inf, maxtime_optim = Inf,
                alg = RK4(),
                maxiterations_ode = 3, maxiterations_optim = 0)
    i = findfirst(d -> params(d[1]["init"]) == x[1], res_distr)
    @test params(res["x"]) ≈ params(res_distr[i][1]["x"]) atol = 1e-5
    @test res["loss"] ≈ res_distr[i][1]["loss"] atol = 1e-5
end

@testset "batch" begin
    import MLPGradientFlow: get_functions
    Random.seed!(12)
    net = Net(layers = ((2, relu, true), (1, identity, false)), input = randn(2, 30), target = randn(1, 30))
    funcs_fullbatch = get_functions(net, Inf);
    funcs_weighted = get_functions(net, Inf, weights = [zeros(10); ones(10); zeros(10)]);
    funcs_minibatch = get_functions(net, Inf, batchsize = 10);
    p = random_params(net)
    @test funcs_fullbatch[1](p) == funcs_minibatch[1](p) # loss evaluation on full batch in both cases
    dp_fullbatch = zero(p)
    funcs_fullbatch[2](dp_fullbatch, p)
    dp_weighted = zero(p)
    funcs_weighted[2](dp_weighted, p)
    dp_minibatch = zero(p)
    dp_tmp = zero(p)
    funcs_minibatch[2](dp_tmp, p) # first batch
    dp_minibatch .+= dp_tmp
    funcs_minibatch[2](dp_tmp, p) # second batch
    dp_minibatch .+= dp_tmp
    @test dp_tmp ≈ dp_weighted
    funcs_minibatch[2](dp_tmp, p) # third batch
    dp_minibatch .+= dp_tmp
    @test dp_fullbatch ≈ dp_minibatch
end

@testset "crossentropy" begin
    import MLPGradientFlow: NegativeLogLikelihood
    inp = randn(10, 100)
    y = rand(1:10, 100)
    y2 = zero(inp); setindex!.(Ref(y2), 1, y, 1:100)
    net = Net(layers = ((2, tanh, false), (10, softmax, false)),
              input = inp, target = y)
    net2 = Net(layers = ((2, tanh, false), (10, softmax, false)),
               input = inp, target = y2)
    θ = ((randn(2, 10), nothing), (randn(10, 2), nothing))
    x = MLPGradientFlow.params(θ...)
    fw_loss = fw_lossfunc(inp, y, (tanh, identity), losstype = :crossentropy)
    @test loss(net, x, losstype = NegativeLogLikelihood()) ≈ fw_loss(θ)
    @test loss(net2, x, losstype = NegativeLogLikelihood()) ≈ fw_loss(θ)
    dx = gradient(net, x, losstype = NegativeLogLikelihood())
    dx2 = gradient(net2, x, losstype = NegativeLogLikelihood())
    @test dx ≈ ForwardDiff.gradient(fw_loss, flatten(θ))
    @test dx2 ≈ dx
    h = hessian(net, x, losstype = NegativeLogLikelihood())
    h2 = hessian(net2, x, losstype = NegativeLogLikelihood())
    @test h ≈ ForwardDiff.hessian(fw_loss, flatten(θ))
    @test h ≈ h2
end

@testset "teacher net" begin
    input = randn(2, 10)
    teacher = TeacherNet(; layers = ((2, softplus, true), (1, identity, true)), input)
    @test size(teacher(input)) == (1, 10)
    teacher = TeacherNet(; layers = ((1, softplus, true),), Din = 3)
    @test size(teacher(randn(3, 10))) == (1, 10)
end

@testset "tauinv" begin
    net = Net(layers = ((2, relu, true), (1, identity, true)), input = randn(2, 10), target = randn(1, 10))
    p = random_params(net)
    tauinv = zero(p) # nothing moves
    res = train(net, p; tauinv, maxiterations_ode = 100, maxiterations_optim = 0)
    @test params(res["x"]) == p
    tauinv.w1[:, end] .= 1 # only biases of input layer move
    res = train(net, p; tauinv, maxiterations_ode = 100, maxiterations_optim = 0)
    @test params(res["x"]).w2 == p.w2
    @test params(res["x"]).w1[:, 1:end-1] == p.w1[:, 1:end-1]
    @test params(res["x"]).w1[:, end] ≠ p.w1[:, end]
end

@testset "standard normal input" begin
    input = randn(1, 10^4)
    for b1 in (true, false), b2 in (true, false)
        teacher = TeacherNet(; layers = ((2, softplus, true), (1, identity, true)), input)
        teacher.p .= randn(length(teacher.p))
        target = teacher(input)
        teacher.net.target .= target
        student = Net(; layers = ((2, tanh_fast, b1), (1, identity, b2)), input, target)
        infinite_student = gauss_hermite_net(teacher, student)
        p = random_params(student)
        p .= randn(length(p))
        @test loss(student, p) ≈ loss(infinite_student, p) atol = 1e-1
        @test gradient(student, p) ≈ gradient(infinite_student, p) atol = 1e-1
        @test hessian(student, p) ≈ hessian(infinite_student, p) atol = 5e-1
        neti = NetI(teacher, student)
        @test loss(neti, p) ≈ loss(infinite_student, p)
        @test gradient(neti, p) ≈ gradient(infinite_student, p)
        @test hessian(neti, p) ≈ hessian(infinite_student, p)
        if b1 == true && b2 == true
            res2 = train(neti, p, maxT = 20, maxtime_ode = 5*60, maxiterations_optim = 0)
            res1 = train(infinite_student, p, maxT = 20, tauinv = 1/MLPGradientFlow.n_samples(infinite_student), maxtime_ode = 10, maxiterations_optim = 0)
            @test res1["loss"] ≈ res2["loss"]
            @test params(res1["x"]) ≈ params(res2["x"])
        end
    end
end

import MLPGradientFlow: integrate, NormalIntegral, _stride_arrayize, ϕ, ϕϕ, ∂r₁ϕϕ, ∂b₁ϕϕ, ∂uϕϕ, ∂rϕϕ, ∂rϕ, ∂bϕ, ∂bϕϕ, ∂u∂uϕϕ, ∂b₁∂uϕϕ, ∂b₁∂b₁ϕϕ, ∂r₁∂uϕϕ, ∂r₁∂b₁ϕϕ, ∂b₁∂b₂ϕϕ, ∂r₂∂b₁ϕϕ, ∂r₁∂r₁ϕϕ
@testset "fast integrals" begin
    sigmoid_test = x -> erf(x/sqrt(2))
    G_test = x -> (1 + erf(x/sqrt(2)))/2
    MLPGradientFlow.deriv(::typeof(sigmoid_test)) = x -> ForwardDiff.derivative(sigmoid_test, x)
    MLPGradientFlow.second_deriv(::typeof(sigmoid_test)) = x -> ForwardDiff.derivative(MLPGradientFlow.deriv(sigmoid_test), x)
    MLPGradientFlow.deriv(::typeof(G_test)) = x -> ForwardDiff.derivative(G_test, x)
    MLPGradientFlow.second_deriv(::typeof(G_test)) = x -> ForwardDiff.derivative(MLPGradientFlow.deriv(G_test), x)
    g1 = _stride_arrayize(NormalIntegral(d = 1))
    g2 = _stride_arrayize(NormalIntegral(d = 2))
    for (f1, f2) in ((sigmoid2, sigmoid_test), (normal_cdf, G_test))
        for kind in (ϕ, ϕϕ, ∂rϕ, ∂rϕϕ, ∂bϕ, ∂bϕϕ)
            @test integrate(kind(), g1.w, g1.x, f1, .3, .1) ≈
                  integrate(kind(), g1.w, g1.x, f2, .3, .1)
            @test integrate(kind(), g1.w, g1.x, f1, .3, 0.) ≈
                  integrate(kind(), g1.w, g1.x, f2, .3, 0.) atol = eps()
            @test integrate(kind(), g1.w, g1.x, f1, .3, -.1) ≈
                  integrate(kind(), g1.w, g1.x, f2, .3, -.1)
        end
        for kind in (ϕϕ, ∂r₁ϕϕ, ∂b₁ϕϕ, ∂uϕϕ, ∂u∂uϕϕ, ∂b₁∂uϕϕ, ∂b₁∂b₁ϕϕ, ∂r₁∂uϕϕ, ∂r₁∂b₁ϕϕ, ∂b₁∂b₂ϕϕ, ∂r₂∂b₁ϕϕ, ∂r₁∂r₁ϕϕ)
            @test integrate(kind(), g2.w, g2.x, f1, .3, .1, f1, .4, -.2, .3) ≈
                  integrate(kind(), g2.w, g2.x, f2, .3, .1, f2, .4, -.2, .3)
            @test integrate(kind(), g2.w, g2.x, f1, .3, 0., f1, .4, -.2, 1.) ≈
                  integrate(kind(), g2.w, g2.x, f2, .3, 0., f2, .4, -.2, 1.)
            @test integrate(kind(), g2.w, g2.x, f1, .3, 0., f1, .4, -.2, -1.) ≈
                  integrate(kind(), g2.w, g2.x, f2, .3, 0., f2, .4, -.2, -1.)
            @test integrate(kind(), g2.w, g2.x, f2, .3, 0., f2, .4, 0., .9) ≈
                  integrate(kind(), g2.w, g2.x, f1, .3, 0., f1, .4, 0., .9) atol = 1e-14
        end
    end
end
