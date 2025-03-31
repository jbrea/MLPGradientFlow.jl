```@example teacherstudent
using MLPGradientFlow, Random

Random.seed!(71)

teacher = TeacherNet(layers = ((5, g, false), (1, identity, false)), Din = 2,
                     p = params((randn(5, 2), nothing), (randn(1, 5), nothing)))

input = randn(2, 10_000)
target = teacher(input)

student = Net(; layers = ((5, g, false), (1, identity, false)),
                input, target)

p = random_params(student)

res = train(student, p,
            maxtime_ode = 10, maxiterations_optim = 0,
            n_samples_trajectory = 10^3)
```

Let us compare the solution found by the student to the teacher parameters:

```@example teacherstudent
p_res = params(res["x"])
p_res.w1
```
```@example teacherstudent
teacher.p.w1
```

```@example teacherstudent
p_res.w2
```

```@example teacherstudent
teacher.p.w2
```

We see that the student perfectly reproduces the teacher up to permutation of the hidden neurons.
