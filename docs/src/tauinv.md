To experiment with different timescales for different parameters, we can use separate `tauinv` for each parameter. In the following example the biases change on a much faster timescale than the weights.

```@example tauinv
using MLPGradientFlow, Random

Random.seed!(14)

input = randn(2, 5_000)
teacher = TeacherNet(; layers = ((5, sigmoid2, true), (1, identity, true)),
                       p = params((randn(5, 2), randn(5)), (randn(1, 5), randn(1))),
                       input)

student = Net(; layers = ((4, sigmoid2, true), (1, identity, true)),
                input, target = teacher(input))

p = random_params(student)

neti = NetI(teacher, student)

res_standard = train(neti, p; maxtime_ode = 10, maxiterations_optim = 0)
```

```@example tauinv
tauinv = zero(p) .+ 1
tauinv.w1[:, end] .= 1e-4
tauinv.w2[:, end] .= 1e-4

res = train(neti, p; tauinv, maxtime_ode = 20, maxiterations_optim = 0)
```

We see that the two dynamics converge to different solutions.
