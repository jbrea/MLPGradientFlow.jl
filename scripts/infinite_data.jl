include(joinpath(@__DIR__, "helper.jl"))

function finite_data(; kwargs...)
    setup(; parameter_rng = Xoshiro(32), Din = 9, k = 8, r = 3, biases = false, kwargs...)
end

f = softplus
_, x, xt = finite_data(; f)
neti = NetI(x, xt, load_potential_approximator(f))

res = train(neti, x, maxiterations_optim = 0)
solution = params(res["x"])
loss(neti, solution)
hessian_spectrum(neti, solution).values

neti2 = NetI(x, xt, f)
loss(neti2, solution)
maximum(abs, gradient(neti2, solution))
hessian_spectrum(neti2, solution).values

results = DataFrame(seed = Int[], Nsamples = Int[],
                    duration = Float64[], distance = Float64[])
for (seed, Nsamples) in Iterators.product(1:20, (10^3, 10^4, 10^5))
    @show seed, Nsamples
    net, x, _ = finite_data(; seed, Nsamples, f)
    tmp = train(net, x, maxiterations_optim = 0,
                maxtime_ode = 120)
    distance = sum(abs2, solution - params(tmp["x"]))
    push!(results, (seed, Nsamples, tmp["ode_time_run"], distance))
end
results.rel_duration = results.duration ./ res["ode_time_run"]
serialize("infinite_data.dat", (; results, res))

fig = @pgf Axis({xmode = "log", ymode = "log",
           xlabel = "distance to infinite data solution", ylabel = "relative optimization duration"},
          Plot({"scatter",
                "only marks",
                "scatter src" = "explicit symbolic",
                "scatter/classes" =
                    {
                     "1000" = {mark = "*", color = colors[1]},
                     "10000" = {mark = "*", color = colors[2]},
                     "100000" = {mark = "*", color = colors[3]},
                    }
               },
               Table({x = "distance", y = "rel_duration", meta = "Nsamples"},
                     results)
              ),
          Legend([raw"$10^3$ samples", raw"$10^4$ samples", raw"$10^5$ samples"])
         )
pgfsave("infinite_data.tikz", fig)

X = [ones(nrow(results)) log.(results.Nsamples)]
y = log.(results.distance)
linreg_coeffs = X \ y
fit = union(exp(linreg_coeffs[1]) * results.Nsamples .^ linreg_coeffs[2])
fig2 = @pgf Axis({xmode = "log", ymode = "log",
           ylabel = "distance to infinite data solution",
           xlabel = "number of samples \$N\$"},
          Plot({
                "only marks",
               },
               Table({y = "distance", x = "Nsamples"},
                     results)
              ),
          Plot({no_marks, red}, Coordinates(union(results.Nsamples), fit)),
          raw"\node[red] at (3e4, 4e-2) {$\propto 1/N$};"
         )
pgfsave("infinite_data2.tikz", fig2)
