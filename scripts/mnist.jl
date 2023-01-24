using Pkg
Pkg.activate(@__DIR__)
using MLPGradientFlow, OpenML, DataFrames, Optim, OrdinaryDiffEq, Sundials

mnist_x, mnist_y = let df = OpenML.load(554) |> DataFrame
    Array(Array(df[:, 1:end-1] ./ 255)'), # scaling the input
    parse.(Int, Array(df.class)) .+ 1
end

net = Net(layers = ((100, relu, true), (10, softmax, true)),
          input = mnist_x, target = mnist_y)
x = random_params(net)

res = train(net, x,
            alg = CVODE_BDF(linear_solver=:GMRES, method = :Function),
            hessian_template = nothing,
            maxtime_ode = 360,
            optim_solver = LBFGS(),
            maxtime_optim = 180)
