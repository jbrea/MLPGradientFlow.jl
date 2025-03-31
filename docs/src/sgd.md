```@example sgd
using MLPGradientFlow, Random, Optimisers

Random.seed!(1)

input = randn(2, 10_000)
teacher = TeacherNet(; layers = ((5, softplus, false), (1, identity, false)), input)
target = teacher(input)

net = Net(; layers = ((4, softplus, false), (1, identity, false)),
            input, target)
p = random_params(net)

res_gradientflow = train(net, p, maxT = 30,
                         maxiterations_optim = 0,
                         n_samples_trajectory = 10^4)

```

Let us compare to training without and with minibatches and gradient descent.

```@example sgd
res_descent_fullbatch = train(net, p, alg = Descent(eta = 1e-1),
                              maxiterations_ode = 10^8,
                              maxtime_ode = 30, maxiterations_optim = 0,
                              n_samples_trajectory = 10^3)
```
```@example sgd
res_descent = train(net, p, alg = Descent(eta = 1e-1), batchsize = 100,
                    maxiterations_ode = 10^8,
                    maxtime_ode = 20, maxiterations_optim = 0,
                    n_samples_trajectory = 10^3)

```

Not surprisingly, gradient descent takes more time than gradient flow (which uses second order information), and therefore does not find a point of equally low loss and gradient as gradient flow.

```@example sgd
tdb, ttb, _ = MLPGradientFlow.trajectory_distance(res_descent_fullbatch, res_gradientflow)
td, tt, _ = MLPGradientFlow.trajectory_distance(res_descent, res_gradientflow)

using CairoMakie

f = Figure()
ax = Axis(f[1, 1], ylabel = "distance", yscale = Makie.pseudolog10, xscale = Makie.pseudolog10, xlabel = "time")
lines!(ax, ttb, tdb, label = "full batch")
lines!(ax, tt, td, label = "batchsize = 100")
axislegend(ax)
f
save("trajectory_distance1.png", ans); # hide
```

![](trajectory_distance1.png)

Gradient descent stays close to gradient flow, both in full batch mode and with minibatches of size 100.

```@example sgd
res_adam_fullbatch = train(net, p, alg = Adam(),
                           maxtime_ode = 20, maxiterations_optim = 0,
                           n_samples_trajectory = 10^3)
```
```@example sgd
res_adam = train(net, p, alg = Adam(), batchsize = 100, maxiterations_ode = 10^8,
                 maxtime_ode = 20, maxiterations_optim = 0,
                 n_samples_trajectory = 10^3)
```
```@example sgd
tdb, ttb, _ = MLPGradientFlow.trajectory_distance(res_adam_fullbatch, res_gradientflow)
td, tt, _ = MLPGradientFlow.trajectory_distance(res_adam, res_gradientflow)

using CairoMakie

f = Figure()
ax = Axis(f[1, 1], ylabel = "distance", yscale = Makie.pseudolog10, xlabel = "trajectory steps")
lines!(ax, 1:length(tdb), tdb, label = "full batch")
lines!(ax, 1:length(td), td, label = "batchsize = 100")
axislegend(ax, position = :rb)
f
save("trajectory_distance2.png", ans); # hide
```

![](trajectory_distance2.png)

This is not the case for `Adam` which uses effectively different timescales for the different parameters.


