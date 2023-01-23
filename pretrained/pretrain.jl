using Distributed
addprocs(30, exeflags="--project=$(joinpath(@__DIR__, ".."))")
@everywhere begin
using MLPGradientFlow, Serialization, Optim, OrdinaryDiffEq
import MLPGradientFlow: net, train, softplus, random_params, params, pickle
import MLPGradientFlow: g3, dgdru, d2gdru, tanh, sigmoid, g, gelu, softplus

function load_integrals(g, activation)
    data = deserialize("numerical_integrals2-$g-$activation.dat")
    (inp = hcat(collect.(first.(data))...),
     targ = hcat(last.(data)...))
end
function create_net(data, r)
    s1 = MLPGradientFlow.Standardizer(data.inp)
    s2 = MLPGradientFlow.Standardizer(data.targ)
    n = net(layers = ((r, softplus, true),
                      (r, softplus, true),
                      (size(data.targ, 1), identity, true)),
            input = s1(data.inp), target = s2(data.targ))
    p = random_params(n)
    (; n, p, s1, s2)
end
end

conditions = collect(Iterators.product((g3, dgdru, d2gdru),
                                       (g, tanh, sigmoid, gelu, softplus),
                                       (64, 96)))

onetraining = ARGS[1] == "1"
@sync @distributed for (func, activation, r) in conditions
#     isfile("res-$func-$activation-$r.dat") && continue
    @show func activation r onetraining
    data = load_integrals(func, activation)
    n, p, s1, s2 = create_net(data, r)
    pickle("standardizers-$func-$activation-$r.pt", Dict("s1m" => s1.m,
                                                         "s1s" => s1.s,
                                                         "s2m" => s2.m,
                                                         "s2s" => s2.s))
    res = train(n, p, maxiterations_ode = 0,
#                 maxtime_ode = 4*3600,
#                 alg = RK4(),
#                 maxT = 10^4,
                optim_solver = BFGS(),
                maxiterations_optim = 10^8,
                maxtime_optim = (onetraining+1)*24*3600)
#     res = deserialize("res-$func-$activation-$r.dat")
    if onetraining
        serialize("res-$func-$activation-$r.dat", res)
    else
        serialize("res1-$func-$activation-$r.dat", res)
        res = train(n, params(res["x"]), maxiterations_ode = 0,
    #                 alg = RK4(),
    #                 maxT = 10^4,
                    optim_solver = BFGS(),
                    maxiterations_optim = 10^8,
              loss_scale = 1/(2*res["loss"]),
                    maxtime_optim = 24*3600)
        serialize("res2-$func-$activation-$r.dat", res)
    end
    pickle("params-$func-$activation-$r-$(onetraining ? 1 : 2).pt", res["x"])
end
