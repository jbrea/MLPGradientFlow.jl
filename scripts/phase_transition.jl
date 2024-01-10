using Distributed
@everywhere begin
    cd(@__DIR__)
    using MLPGradientFlow, LinearAlgebra, ComponentArrays, OrdinaryDiffEq, Random
end

num_teach = eval(Meta.parse(ARGS[2]))
configs = collect(Iterators.product(eval(Meta.parse(ARGS[1])),
                                    1:num_teach,
                                    (num_teach,)))

@sync @distributed for (seed, num_student, num_teach) in configs
    @show seed num_student
    param_teach = ComponentVector(w1 = [I(num_teach) zeros(num_teach)],
                                  w2 = ones(num_teach))
    rng = Xoshiro(seed)
    p0 = MLPGradientFlow.glorot(rng, (num_teach+1, num_student, 1))
    ni = NetI(p0, param_teach, Val(sigmoid2))
    res = train(ni, p0,
                maxiterations_optim = 0,
                maxiterations_ode = 10^5,
                maxtime_ode = 60*60) # 1hr
    MLPGradientFlow.pickle("erf-stud=$num_student-teach=$num_teach-seed=$seed.pkl", res)
end
