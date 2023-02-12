using Distributed
addprocs(30, exeflags="--project=$(joinpath(@__DIR__, ".."))")
@everywhere begin
using MLPGradientFlow, Serialization, Optim, OrdinaryDiffEq
import MLPGradientFlow: g3, dgdru, d2gdru, tanh, sigmoid, g, gelu, softplus, pickle, params

function load_integrals(g, activation)
    data = deserialize("numerical_integrals2-$g-$activation.dat")
    (inp = hcat(collect.(first.(data))...),
     targ = hcat(last.(data)...))
end
function create_net(data, r)
    s1 = MLPGradientFlow.Standardizer(data.inp)
    s2 = MLPGradientFlow.Standardizer(data.targ)
    n = Net(layers = ((r, softplus, true),
                      (r, softplus, true),
                      (size(data.targ, 1), identity, true)),
            input = s1(data.inp), target = s2(data.targ))
    p = random_params(n)
    (; n, p, s1, s2)
end
end

conditions = collect(Iterators.product((g3, dgdru, d2gdru),
                                       (g, tanh, sigmoid, gelu, softplus),
                                       (96,)))

@sync @distributed for (func, activation, r) in conditions
    @show func activation r
    data = load_integrals(func, activation)
    n, p, s1, s2 = create_net(data, r)
    pickle("standardizers-$func-$activation-$r.pt", Dict("s1m" => s1.m,
                                                         "s1s" => s1.s,
                                                         "s2m" => s2.m,
                                                         "s2s" => s2.s))
    for day in 1:4
        if day > 1
            res = deserialize("res$(day-1)-$func-$activation-$r.dat")
            p = params(res["x"])
        end
        res = train(n, p,
                    maxiterations_ode = 0,
                    optim_solver = BFGS(),
                    loss_scale = day == 1 ? 1. : 1/sqrt(res["loss"]),
                    progress_interval = 120,
                    maxiterations_optim = 10^8,
                    maxtime_optim = 24*3600)
        serialize("res$day-$func-$activation-$r.dat", res)
        if day == 4
            pickle("params-$func-$activation-$r.pt", res["x"])
        end
    end
end
