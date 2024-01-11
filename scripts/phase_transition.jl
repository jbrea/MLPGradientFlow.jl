using Distributed
@everywhere begin
    cd(@__DIR__)
    using MLPGradientFlow, LinearAlgebra, ComponentArrays, OrdinaryDiffEq, Random
    LinearAlgebra.BLAS.set_num_threads(1)
end

num_teach = eval(Meta.parse(ARGS[2]))
configs = collect(Iterators.product(eval(Meta.parse(ARGS[1])),
                                    1:num_teach,
                                    (num_teach,)))

@sync @distributed for (seed, num_student, num_teach) in configs
    @show seed num_student
    fn = "erf-stud=$num_student-teach=$num_teach-seed=$seed.pkl"
    param_teach = ComponentVector(w1 = [I(num_teach) zeros(num_teach)],
                                  w2 = ones(num_teach))
    if isfile(fn)
        res0 = MLPGradientFlow.unpickle(fn)
        if isnothing(match(r"maxtime", res0["ode_stopped_by"]))
            continue
        elseif res0["gnorm"] < 1e-8
            continue
        else
            p0 = MLPGradientFlow.params(res0["x"])
        end
        ni = NetI(p0, param_teach, Val(sigmoid2))
        res = train(ni, p0,
                    maxiterations_optim = 0,
                    maxiterations_ode = 10^5,
                    maxtime_ode = 2*60*60) # 2hr
        res["init1"] = res["init"]
        res["init"] = res0["init"]
        res["loss_curve"] = [res0["loss_curve"]; res["loss_curve"]]
        ode_t0 = maximum(keys(res0["trajectory"]))
        for (k, v) in res["trajectory"]
            res0["trajectory"][k + ode_t0] = v
        end
        res["trajectory"] = res0["trajectory"]
    else
        rng = Xoshiro(seed)
        p0 = MLPGradientFlow.glorot(rng, (num_teach+1, num_student, 1))
        ni = NetI(p0, param_teach, Val(sigmoid2))
        res = train(ni, p0,
                    maxiterations_optim = 0,
                    maxiterations_ode = 10^5,
                    maxtime_ode = 60*60) # 1hr
    end
    MLPGradientFlow.pickle(fn, res)
end
