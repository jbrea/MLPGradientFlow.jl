using MLPGradientFlow
using ForwardDiff, ComponentArrays, Statistics, SpecialFunctions, FiniteDiff, Random
using Distributed
using Test

Random.seed!(123)

@testset "activation functions" begin
    import MLPGradientFlow: g, sigmoid, tanh, square, relu, gelu, softplus, deriv, second_deriv, A_mul_B!, alloc_a′, alloc_a′′
    for f in (g, sigmoid, square, relu, gelu, tanh, softplus, sigmoid2)
        @info "testing activation function $f."
        inp = [-.2, 3.]'
        y = f.(inp)
        ff = f == gelu ? x -> x/2*(1 + erf(x/sqrt(2))) : f == sigmoid2 ? x -> erf(x/sqrt(2)) : f
        @test ff.(inp) ≈ f.(inp)
        y′ = ForwardDiff.derivative.(ff, inp)
        y′′ = ForwardDiff.derivative.(x -> ForwardDiff.derivative(ff, x), inp)
        w = ones(1, 1)
        a = zeros(1, 2)
        a′ = alloc_a′(f, Float64, 1, 2, false)
        a′′ = alloc_a′′(f, Float64, 1, 2)
        A_mul_B!(a, f, w, inp, 1:2)
        @test a ≈ y
        A_mul_B!(a, a′, f, w, inp, 1:2)
        @test a ≈ y
        @test a′ ≈ y′
        @test deriv(f).(inp) ≈ y′
        A_mul_B!(a, a′, a′′, f, w, inp, 1:2)
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
function fw_lossfunc(input, target, f;
                     sizes = nothing, losstype = :mse,
                     scale = 1/size(input, 2),
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
            -sum(getindex.(Ref(log.(output)),
                           target, 1:size(input, 2)))*scale
        else
            sum(abs2, forward(x, f, input) - target)*scale
        end
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
    @test fw_loss(θ) ≈ loss(n, x, losstype = :mse)
    pred = n(x)
    @test sqrt(sum(abs2, pred - target)/size(target, 2)) ≈ loss(n, x, losstype = :rmse)
    @test sum(abs2, pred - target)/size(target, 2) ≈ loss(n, x, losstype = :mse)
    θ_wrong = ((randn(4, 2), randn(4)), (randn(2, 4), nothing), (randn(2, 2), nothing))
    @test_throws AssertionError loss(n, params(θ_wrong...))
    @test_throws AssertionError loss(n, randn(10))
    @test loss(n, Array(x)) == loss(n, x)
    input = randn(2, 5)
    target = randn(2, 5)
    @test fw_forward(θ, f, input) ≈ forward!(n, x, input, derivs = 0)
    fw_loss2 = fw_lossfunc(input, target, f)
    @test fw_loss2(θ) ≈ loss(n, x, input, target)
    @test 8.3*fw_loss2(θ) ≈ loss(n, x, input, target, scale = 8.3)
end

@testset "gradient" begin
    import MLPGradientFlow: loss, params, Net, g, sigmoid, gradient, gradient!
    input = randn(2, 3)
    target = randn(2, 3)
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
    Gs = gradient(n, x, scale = 7.2)
    @test Gs ≈ 7.2*G
    oldG = copy(G)
    gradient!(G, n, x)
    @test oldG == G/size(input, 2)
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
    Hs = hessian(n, x, scale = 3.4)
    @test Hs ≈ 3.4*H
    e, v = hessian_spectrum(n, x)
    @test size(v) == (31, 31)
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
        nx > 70 ? (nx-70)^2/2 : 0.
    end
    @test f!(x) ≈ loss(n, x, losstype = :se)
    xl = 100x
    @test f!(xl) ≈ loss(n, xl, losstype = :se) + r(xl)
    fw_loss = fw_lossfunc(input, target, f, scale = 1)
    fw_loss_r = x -> fw_loss(x) + r(x)
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
    @test loss(n, res.init) > loss(n, res.ode[end])
    @test loss(n, res.ode[end]) ≥ loss(n, res.x) - sqrt(eps())
end

@testset "infinite data" begin
    inp = randn(2, 10^5)
    xt = ComponentArray(w1 = randn(2, 2), w2 = randn(1, 2))
    targ = xt.w2 * softplus.(xt.w1 * inp)
    n = Net(layers = ((3, softplus, false), (1, identity, false)),
            input = inp, target = targ)
    x = random_params(n)
    ni = NetI(x, xt, softplus)
    @test loss(ni, x) ≈ loss(n, x) atol = 1e-1
    gi = gradient(ni, x)
    @test FiniteDiff.finite_difference_gradient(x -> loss(ni, x), x) ≈ gi atol = 1e-4
    @test gradient(n, x) ≈ gi atol = 1e-1
    hi = hessian(ni, x)
    hfd = FiniteDiff.finite_difference_hessian(x -> loss(ni, x), x)
    @test hfd ≈ hi atol = 1e-4
    h = hessian(n, x)
    @test h ≈ hi atol = 1e-1
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

# @testset "batch" begin
#     import MLPGradientFlow: get_functions, Hessian
#     inp = randn(2, 10^3)
#     xt = ComponentArray(w1 = randn(2, 2), w2 = randn(1, 2))
#     targ = xt.w2 * softplus.(xt.w1 * inp)
#     n = Net(layers = ((3, softplus, false), (1, identity, false)),
#             input = inp, target = targ)
#     x = random_params(n)
#     n2 = Net(layers = ((3, softplus, false), (1, identity, false)),
#              input = inp[:, 11:20], target = targ[:, 11:20])
#     n3 = Net(layers = ((3, softplus, false), (1, identity, false)),
#              input = inp[:, 21:30], target = targ[:, 21:30])
#     batcher = MiniBatch(inp, 10)
#     f!, g!, h!, fgh!, fg! = get_functions(n, Inf; hessian_template = Hessian(x),
#                                           batcher)
#     dx = zero(x)
#     g!(dx, x)
#     @test dx ≈ gradient(n2, x)
#     @test f!(x) ≈ loss(n2, x, losstype = :se)
#     H = zeros(length(x), length(x))
#     h!(H, x)
#     @test H ≈ hessian(n2, x)
#     g!(dx, x)
#     @test dx ≈ gradient(n3, x)
#     @test f!(x) ≈ loss(n3, x, losstype = :se)
#     H = zeros(length(x), length(x))
#     h!(H, x)
#     @test H ≈ hessian(n3, x)
#     for _ in 1:100-2
#         MLPGradientFlow.step!(batcher)
#     end
#     @test batcher() == 1:10
# end

@testset "crossentropy" begin
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
    @test loss(net, x, losstype = :crossentropy) ≈ fw_loss(θ)
    @test loss(net2, x, losstype = :crossentropy) ≈ fw_loss(θ)
    dx = gradient(net, x, losstype = :crossentropy)
    dx2 = gradient(net2, x, losstype = :crossentropy)
    @test dx ≈ ForwardDiff.gradient(fw_loss, flatten(θ))
    @test dx2 ≈ dx
    h = hessian(net, x, losstype = :crossentropy)
    h2 = hessian(net2, x, losstype = :crossentropy)
    @test h ≈ ForwardDiff.hessian(fw_loss, flatten(θ))
    @test h ≈ h2
end

@testset "approximators" begin
    import MLPGradientFlow: glorot_normal
    xt = ComponentArray(w1 = glorot_normal(2, 3), w2 = glorot_normal(3, 1))
    x = ComponentArray(w1 = glorot_normal(2, 4), w2 = glorot_normal(4, 1))
    for f in (g, sigmoid, gelu, MLPGradientFlow.tanh, softplus, relu, sigmoid2)
        @show f
        ni = NetI(x, xt, f)
        ax = if f ∈ (relu, sigmoid2)
            Val(f)
        else
            load_potential_approximator(f)
        end
        ni2 = NetI(x, xt, ax)
        l1 = loss(ni, x)
        l2 = loss(ni2, x)
        g1 = gradient(ni, x)
        g2 = gradient(ni2, x)
        h1 = hessian(ni, x)
        h2 = hessian(ni2, x)
        @show abs(l1 - l2) sqrt(sum(abs2, g1 - g2)) sqrt(sum(abs2, h1 - h2))
        @test l1 ≈ l2 atol = 1e-5
        @test g1 ≈ g2 atol = 1e-3
        if f ≠ relu # the integrator is wrong for relu because of derivative of heaviside
            @test h1 ≈ h2 atol = 1e-1
        end
    end
end

