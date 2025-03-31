In the following example, we train a network with one hidden layer of 5 softplus neurons on random in- and output.

```@example train1
using MLPGradientFlow, Random

Random.seed!(123)

input = randn(2, 1_000)  # 2-dimensional random input
target = randn(1, 1_000) # 1-dimensional random output

net = Net(; layers = ((5, softplus, true),     # 5 relu neurons with biases
                      (1, identity, true)), # 1 identity neuron with bias
            input,
            target)

p = random_params(net)

result = train(net, p, maxtime_ode = 20., maxtime_optim = 20., n_samples_trajectory = 10^3)
```

We see that optimization has found a point with a very small gradient:

```@example train1
gradient(net, params(result["x"]))
```

Let us inspect the spectrum of the hessian:
```@example train1
hessian_spectrum(net, params(result["x"]))
```

The eigenvalues are all positive, indicating that we are in a local minimum.


```@example train1
using CairoMakie

function plot_losscurve(result; kwargs...)
    f = Figure()
    ax = Axis(f[1, 1], yscale = log10, xscale = log10, ylabel = "loss", xlabel = "time", kwargs...)
    scatter!(ax, collect(keys(result["trajectory"])) .+ 1, result["loss_curve"])
    f
end

plot_losscurve(result)
save("loss_curve1.png", ans); nothing # hide
```

![](loss_curve1.png)

