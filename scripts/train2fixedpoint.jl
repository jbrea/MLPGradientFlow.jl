using Distributed

addprocs(48, exeflags="--project=$(joinpath(@__DIR__, "."))")

@everywhere begin
    include(joinpath(@__DIR__, "helper.jl"))
end

const activation_function = length(ARGS) > 0 ? getproperty(MLPGradientFlow, Meta.parse(ARGS[1])) : g

settings = collect(Iterators.product(1:25, (2, 4), (1, 3//2, 2, 3), (random_teacher, aifeynman_11)))

@sync @distributed for (seed, k, ρ, teacher) in settings
    @show seed k ρ teacher
    net, x, xt = setup(; seed = seed, Din = k, k = k, r = k*ρ, f = activation_function, teacher)
    res = train(net, x,
                maxtime_ode = 12*3600,
                maxtime_optim = 12*3600,
                maxiterations_ode = 10^10,
                maxiterations_optim = 10^10,
                maxnorm = 10^3,
                g_tol = 1e-16,
                patience = 10^4)
    serialize("fixedpoint-$activation_function-$seed-$k-$ρ-$teacher.dat", (; seed, k, ρ, teacher, res, xt))
end
