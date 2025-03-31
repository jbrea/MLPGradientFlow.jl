# Activation Functions

```@example
using MLPGradientFlow, CairoMakie

f = Figure()
x = -5:1e-3:5
for (i, activation) in pairs((g, square, gelu, cube, softplus, relu, selu, silu, sigmoid, tanh_fast, sigmoid2, normal_cdf))
    ax = Axis(f[(i-1) % 3, (i-1) ÷ 3], title = string(activation))
    lines!(ax, x, activation.(x))
end
f
save("activations.png", f); nothing # hide
```

![](activations.png)
