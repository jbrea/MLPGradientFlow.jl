include(joinpath(@__DIR__, "helper.jl"))
using Sundials

trajectory_distance(res1::Dict, res2::Dict) = trajectory_distance(res1["trajectory"], res2["trajectory"])
function trajectory_distance(traj, ref)
    xtraj = MLPGradientFlow.params.(collect(values(traj)))
    xref = MLPGradientFlow.params.(collect(values(ref)))
    dists = Float64[]
    idxs = Int[]
    i0 = 1
    for x in xtraj
        d = Inf
        for j in i0:length(xref)
            dj = MLPGradientFlow.weightnorm(xref[j] - x)
            if dj ≤ d
                d = dj
                i0 = j
            else
                break
            end
        end
        push!(dists, d)
        push!(idxs, i0)
        if i0 == length(xref)
            @warn "reached end of reference"
            break
        end
    end
    dists, collect(keys(ref))[idxs], idxs
end
function dist_and_t(traj, ref)
    d, t, _ = trajectory_distance(traj, ref)
    GC.gc()
    mean(d), maximum(t)
end

function run_trajectory_comparison(algs, id;
        ref_alg = KenCarp58(), maxtime = 20, seed = 123, kwargs...)
    net, x, = setup(; seed, kwargs...)
    ref = train(net, x, alg = ref_alg, maxtime_ode = 1.2*maxtime,
                maxiterations_optim = 0,
                loss_scale = 1.,
                n_samples_trajectory = 10^6, exclude = ["loss_curve"])
    res = [train(net, x; alg,
                       maxtime_ode = maxtime,
                       loss_scale = 1.,
                       maxiterations_optim = 0,
                       n_samples_trajectory = 10^3)
           for (_, alg) in algs]
    losses = (x -> x["loss"]).(res)
    res = [dist_and_t(x["trajectory"], ref["trajectory"]) for x in res]
    DataFrame(method = collect(keys(algs)),
              dist = first.(res),
              t = last.(res),
              loss = losses,
              id = id, seed = seed)
end

# TODO: QNDF, FBDF and TRBDF2
methods() = Dict("KenCarp58" => KenCarp58(),
                 "RK4" => RK4(),
                 "CVODE" => CVODE_BDF(linear_solver=:GMRES),
#                  "FBDF" => FBDF(autodiff = false),
#                  "QNDF" => QNDF(autodiff = false),
                 "TRBDF2" => TRBDF2(autodiff = false),
                 "Rodas5" => Rodas5(autodiff = false),
                 "Descent001" => Descent(1e-3),
                 "Descent01" => Descent(.01),
                 "Adam0001" => Adam(1e-4),
                 "Adam001" => Adam(1e-3))

# warmup
run_trajectory_comparison(methods(), "small", seed = 1, maxtime = 1)

results_small = vcat([run_trajectory_comparison(methods(), "small",
                                                seed = i,
                                                ref_alg = Rodas5(autodiff = false))
                      for i in 111:120]...)
serialize("ode_results_small.dat", results_small)

# warmup
run_trajectory_comparison(methods(), "large", seed = 1, maxtime = 1,
                          Din = 4, r = 100)

results_large = vcat([run_trajectory_comparison(methods(), "large", seed = i,
                                                Din = 4, r = 100, maxtime = 120,
                                                ref_alg = CVODE_BDF(linear_solver=:GMRES))
                      for i in 101:110]...)
serialize("ode_results_large.dat", results_large)

function plot(results; y, ylabel, title = "")
    @pgf Axis({
           ymode = "log",
           xmode = "log",
           xlabel = "distance to reference \$d_m\$",
           ylabel = ylabel,
           title = title,
#            legend_to_label = "ode_legend_$y",
           legend_columns = 1,
           legend_pos = "outer north east",
           legend_style = {draw = "none"},
           font = "\\small",
          },
          Plot({scatter,
                "only marks",
                "scatter src" = "explicit symbolic",
                "scatter/classes" =
                {
                 Rodas5 = {mark = "*", color = colors[5]},
                 KenCarp58 = {mark = "*", color = colors[4]},
                 RK4 = {mark = "*", color = colors[3]},
                 CVODE = {mark = "square*", color = colors[3]},
                 Descent01 = {mark = "*", color = colors[2]},
                 Descent001 = {mark = "square*", color = colors[2]},
                 Adam001 = {mark = "*", color = colors[1]},
                 Adam0001 = {mark = "square*", color = colors[1]},
                 TRBDF2 = {mark = "square", color = colors[5]},
                 QNDF = {mark = "o", color = colors[5]},
                 FBDF = {mark = "o", color = colors[4]}
                }
               },
               Table({x = "dist", y = y, meta = "method"},
                     results)
              ),
          Legend(["Rodas5", "KenCarp58", "RK4", "CVODE\\_BDF",
                  "Descent(0.01)", "Descent(0.001)",
                  "Adam(0.001)", "Adam(0.0001)",
                  "TRBDF2", "QNDF", "FBDF"])
         )
end

results_small = deserialize("ode_results_small.dat")
results_large = deserialize("ode_results_large.dat")

f1 = plot(results_small, y = "t", ylabel = "progress \$t_m\$", title = "33 parameters")
f2 = plot(results_small, y = "loss", ylabel = "final loss")
f3 = plot(results_large, y = "t", ylabel = "progress \$t_m\$", title = "601 parameters")
f4 = plot(results_large, y = "loss", ylabel = "final loss")

pgfsave("results_small_time.tikz", f1)
pgfsave("results_small_loss.tikz", f2)
pgfsave("results_large_time.tikz", f3)
pgfsave("results_large_loss.tikz", f4)
