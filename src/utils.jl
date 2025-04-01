"""
    trajectory_distance(res1, res2)

"""
trajectory_distance(res1::Dict, res2::Dict) = trajectory_distance(res1["trajectory"], res2["trajectory"])
"""
    trajectory_distance(trajectory, reference_trajectory)

Searches for the closest points of `trajectory` in `reference_trajectory` and returns the distances, time points of the `reference_trajectory` and the indices in `reference_trajectory`.
"""
function trajectory_distance(traj, ref)
    xtraj = params.(collect(values(traj)))
    xref = params.(collect(values(ref)))
    dists = Float64[]
    idxs = Int[]
    i0 = 1
    for x in xtraj
        d = Inf
        for j in i0:length(xref)
            dj = weightnorm(xref[j] - x)
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

struct LinearSubspace{V,U,B}
    ref::V
    v1::V
    v2::V
    u::U
    b::B
end
"""
    LinearSubspace(ref, v1, v2)

Construct a 2D linear subspace from `ref` point in directions `v1` and `v2`. See also [`subspace_minloss`](@ref)
"""
function LinearSubspace(ref, v1, v2)
    N = length(v1)
    _v1 = normalize(v1)
    _v2 = normalize(v2)
    u, = svd([2*_v1 _v2 zeros(N, N - 2)]) # 2*_v1 for ordering
    b = u[:, 3:end]
    LinearSubspace(ref, _v1, _v2, u, b)
end
_gradient!(G, ls::LinearSubspace, net, x, v) = G .= ls.b' * gradient(net, ls.b * x + v)
_loss(ls::LinearSubspace, net, x, v) = loss(net, ls.b * x + v)
"""
    to_local_coords(ls::LinearSubspace, p)

Project point `p` to the linear subspace `ls`.
"""
to_local_coords(ls::LinearSubspace, p) = l.u[:, 1:2]' * (ls.ref - p)
"""
    subspace_minloss(net, ref, v1, v2, a1, a2)

Minimize loss in the subspace orthogonal to `v1` and `v2` with the point in the 2D subspace fixed to `ref + a1 * v1 + a2 * v2`.
"""
function subspace_minloss(net, ref, v1, v2, a1, a2)
    ls = LinearSubspace(ref, v1, v2, a1, a2)
    subspace_minloss(net, ls, a1, a2)
end
"""
    subspace_minloss(net, ls::LinearSubspace, a1, a2)

Minimize loss in the subspace orthogonal to `ls` with the point in `ls` fixed to `ls.ref + a1 * ls.v1 + a2 * ls.v2`.
"""
function subspace_minloss(net, ls::LinearSubspace, a1, a2)
    v = ls.ref + a1 * ls.v1 + a2 * ls.v2
    _grad = (G, x) -> _gradient!(G, ls, net, x, v)
    _loss = x -> MLPGradientFlow._loss(ls, net, x, v)
    sol = Optim.optimize(_loss, _grad, zeros(length(ls.ref)-2), LBFGS())
    x = Optim.minimizer(sol)
    loss = _loss(x)
    (; loss, p = ls.b * x + v, delta_loss = _loss(zero(x)) - loss)
end
"""
    grow_net(net)

Add one neuron to the hidden layer. Works only for networks with a single hidden layer.
"""
function grow_net(net::Net)
    l = net.layerspec
    net = Net(layers = ((l[1][1]+1, l[1][2], l[1][3]), l[2]),
              bias_adapt_input = false,
              input = net.input, target = net.target, derivs = 2, verbosity=0)
end
"""
    shrink_net(net)

Remove one neuron from the hidden layer. Works only with networks with a single hidden layer.
"""
function shrink_net(net::Net)
    l = net.layerspec
    net = Net(layers = ((l[1][1]-1, l[1][2], l[1][3]), l[2]),
              bias_adapt_input = false,
              input = net.input, target = net.target, derivs = 2, verbosity=0)
end
grow_net(net::NetI) = NetI(net.teacher, grow_net(net.student))
shrink_net(net::NetI) = NetI(net.teacher, shrink_net(net.student))
"""
    split_neuron(p, i, γ, j = i+1)

Duplicate hidden neuron `i` with mixing ratio `γ` and insert the new neuron at position `j`. Works only for the parameters `p` of a network with a single hidden layer.
"""
function split_neuron(p::ComponentArray, i, γ, j = i+1)
    _w1 = p.w1
    _w2 = p.w2
    w1 = [_w1[1:j-1, :]; _w1[i:i, :]; _w1[j:end, :]]
    w2 = [_w2[:, 1:i-1] γ * _w2[:, i] _w2[:, i+1:j-1]  _w2[:, i] * (1-γ) _w2[:, j:end]]
    ComponentArray(; w1, w2)
end
