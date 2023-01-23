using MLPGradientFlow
using Distributed, Serialization

function generate_input(; rgrid = [10.0.^(-6:.5:-1); .2:.1:3; 4:10],
        ugrid = (x -> vcat(x, -reverse(x)))([-.999; -.99; -.9; range(-.89, -.01, 22)]))
    collect(Iterators.product(rgrid, rgrid, ugrid))
end
function compute_numerical_integrals(inp, g, activation; tol = 1e-14)
    MLPGradientFlow.TOL.atol[] = tol; MLPGradientFlow.TOL.rtol[] = tol;
    y = @distributed vcat for x in inp
        (x, g(activation, x...))
    end
    serialize("numerical_integrals2-$g-$activation.dat", y)
    y
end

addprocs(31, exeflags="--project=$(joinpath(@__DIR__, ".."))")
@everywhere begin
	using MLPGradientFlow
	import MLPGradientFlow: g3, dgdru, d2gdru, tanh, sigmoid, g, gelu, softplus
end

inp = generate_input()
for func in (g3, dgdru, d2gdru), activation in (g, tanh, sigmoid, gelu, softplus)
    @show func activation
    compute_numerical_integrals(inp, func, activation)
end
