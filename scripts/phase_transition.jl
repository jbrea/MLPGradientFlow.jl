using Distributed
@everywhere begin
    cd(@__DIR__)
    using MLPGradientFlow, LinearAlgebra, ComponentArrays, OrdinaryDiffEq, Random
end

configs = collect(Iterators.product(1:20, 1:50))

@sync @distributed for (seed, num_student) in configs
    @show seed num_student
    param_teach = ComponentVector(w1 = [I(50) zeros(50)], w2 = ones(50))
    rng = Xoshiro(seed)
    p0 = MLPGradientFlow.glorot(rng, (51, num_student, 1))
    ni = NetI(p0, param_teach, Val(sigmoid2))
    res = train(ni, p0,
                maxiterations_optim = 0,
                maxiterations_ode = 10^5,
                maxtime_ode = 3) # 1hr
    MLPGradientFlow.pickle("erf-stud=$num_student-teach=50-seed=$seed.pkl", res)
end
