There are two settings in which we can approximate gradient descent on the loss function integrated over an infinite amount of normally distributed input data, i.e.
``\mathbb E_x\left[(\mathrm{net}(p, x) - y)^2\right]``  with normally distributed ``x`` with mean 0 and standard deviation 1:
1. When the input dimension is sufficiently small (`Din ≤ 2`) such that we can use Gauss-Hermite quadrature. This works for arbitrary teacher functions and networks.
2. When we have a single hidden layer teacher and student network. This is particularly fast, when using `erf`-based activation functions, like `normal_cdf` or `sigmoid2` (see [Activation Functions](@ref)).
We illustrate both cases in a 2D example.

```@example gaussian
using MLPGradientFlow, Statistics, Random

Random.seed!(12)

input = randn(2, 5_000)
teacher = TeacherNet(; layers = ((5, sigmoid2, true), (1, identity, false)), input)

student = Net(; layers = ((4, sigmoid2, true), (1, identity, false)),
                input, target = teacher(input))

p = random_params(student)

sample_losses = [let input = randn(2, 5_000), target = teacher(input)
                    loss(student, p; input, target)
                 end
                 for _ in 1:10]
(mean(sample_losses), std(sample_losses))
```

```@example gaussian
net_gh = gauss_hermite_net(teacher, student)

loss(net_gh, p)
```

```@example gaussian
neti = NetI(teacher, student)

loss(neti, p)
```

Let us compare training trajectories of the different networks. The timescales `tauinv` need to be adjusted for the dynamics based on the `student` or the Gauss-Hermite `net_gh`, such that the integration times match.

```@example gaussian
res_sample = train(student, p, tauinv = 1/MLPGradientFlow.n_samples(student),
                   maxT = 10^4, maxiterations_optim = 0, n_samples_trajectory = 1000)
```
```@example gaussian
res_gh = train(net_gh, p, tauinv = 1/MLPGradientFlow.n_samples(net_gh), maxT = 10^4,
               maxiterations_optim = 0, n_samples_trajectory = 1000)
```
```@example gaussian
resi = train(neti, p, maxT = 10^4,
             maxiterations_optim = 0, n_samples_trajectory = 1000)
```


```@example gaussian
td1, tt1, _ = MLPGradientFlow.trajectory_distance(res_sample, resi)
td2, tt2, _ = MLPGradientFlow.trajectory_distance(res_gh, resi)
td3, tt3, _ = MLPGradientFlow.trajectory_distance(res_sample, res_gh)

using CairoMakie

f = Figure()
ax = Axis(f[1, 1], ylabel = "distance", yscale = Makie.pseudolog10, xlabel = "time")
lines!(ax, tt1, td1, label = "sample vs. teacher-student", linewidth = 3)
lines!(ax, tt3, td3, label = "sample vs. gauss-hermite")
lines!(ax, tt2, td2, label = "gauss-hermite vs. teacher-student")
axislegend(ax, position = :lt)
f
save("trajectory_normal.png", ans); # hide
```

![](trajectory_normal.png)

The Gauss-Hermite approximation and the teacher-student network based on symbolic integration of Gaussian integrals give almost indistinguishable results, whereas gradient flow on a finite number of normally distributed samples leads to slight deviations from the infinite data setting.
