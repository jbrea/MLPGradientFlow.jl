using Documenter, MLPGradientFlow

makedocs(
    modules = [MLPGradientFlow],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Johanni Brea",
    sitename = "MLPGradientFlow.jl",
    pages = Any["Overview" => "index.md",
                "Examples" => ["Training to convergence" => "train.md",
                               "Teacher student setup" => "teacherstudent.md",
                               "Comparison of Gradient Flow to SGD" => "sgd.md",
                               "Standard Normal Input" => "normal.md",
                               "Separation of Timescales" => "tauinv.md"],
                "Usage from Python" => "python.md",
                "Activation Functions" => "activations.md",
                "Docstrings" => "docstrings.md"],
    # strict = true,
    # clean = true,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/jbrea/MLPGradientFlow.jl.git",
    push_preview = true
)
