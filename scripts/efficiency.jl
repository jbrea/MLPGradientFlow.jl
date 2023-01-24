include(joinpath(@__DIR__, "helper.jl"))

using CSV

results = CSV.read(joinpath(@__DIR__, "efficiency.csv"), DataFrame)
results.grad_t2j = results.grad_torch ./ results.grad_jl
results.hess_t2j = results.hess_torch ./ results.hess_jl
gs = combine(groupby(results, ["N", "Nparams"]),
             :grad_t2j => median => :gmed,
             :grad_t2j => minimum => :gmin,
             :grad_t2j => maximum => :gmax,
             :hess_t2j => median => :hmed,
             :hess_t2j => minimum => :hmin,
             :hess_t2j => maximum => :hmax,
            )
gs.gyp = gs.gmax - gs.gmed
gs.gyn = gs.gmed - gs.gmin
gs.hyp = gs.hmax - gs.hmed
gs.hyn = gs.hmed - gs.hmin
gs.Nparams = (x -> "$x").(gs.Nparams)
union(gs.Nparams)

function plot_efficiency(x, gs, title)
    @pgf Axis({xmode = "log", ymode = "log", ymin = .5,
               font = "\\small",
               ylabel = "speedup relative to pytorch",
               legend_columns = 3,
               legend_style = {draw = "none"},
               legend_to_label = "pytorch_comparison",
               xlabel = "batch size", title = title},
              Plot({"scatter",
                    "only marks",
                    "error bars/y dir = both",
                    "error bars/y explicit",
                    "scatter src" = "explicit symbolic",
                    "scatter/classes" =
                    {
                     "41" = {mark = "*", color = colors[2]},
                     "201" = {mark = "*", color = colors[3]},
                     "401" = {mark = "*", color = colors[4]},
                    }
                   },
                   Table({"y error plus" = "$(x)yp",
                          "y error minus" = "$(x)yn",
                          x = "N", y = "$(x)med",
                          meta = "Nparams"
                         },
                         gs
                        )
                  ),
              HLine({dashed, black}, 1),
              Legend(["41 parameters", "201 parameters", "401 parameters"])
             )
end
f1 = plot_efficiency("g", gs, "gradient")
f2 = plot_efficiency("h", gs, "hessian")
pgfsave("grad_speedup_pytorch.tikz", f1)
pgfsave("hess_speedup_pytorch.tikz", f2)
