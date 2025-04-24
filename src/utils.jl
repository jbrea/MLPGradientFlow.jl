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
subspace_dim(ls::LinearSubspace) = length(ls.ref)-2
_gradient!(G, ls::LinearSubspace, net, x, v) = G .= ls.b' * gradient(net, ls.b * x + v)
_loss(ls::LinearSubspace, net, x, v) = loss(net, ls.b * x + v)
"""
    to_local_coords(ls::LinearSubspace, p)

Project point `p` to the linear subspace `ls`.
"""
to_local_coords(ls::LinearSubspace, p) = ls.u[:, 1:2]' * (ls.ref - p)
"""
    subspace_minloss(net, ref, v1, v2, a1, a2)

Minimize loss in the subspace orthogonal to `v1` and `v2` with the point in the 2D subspace fixed to `ref + a1 * v1 + a2 * v2`.
"""
function subspace_minloss(net, ref, v1, v2, a1, a2)
    ls = LinearSubspace(ref, v1, v2, a1, a2)
    subspace_minloss(net, ls, a1, a2)
end
function restart_ref_and_origin(i, ls, restart_params)
    iseven(i) && return restart_ref(i, ls, restart_params)
    return restart_origin(i, ls, restart_params)
end
function restart_ref_and_origin_and_nearby(i, ls, restart_params)
    i % 3 == 0 && return restart_ref(i, ls, restart_params)
    i % 3 == 1 && return restart_origin(i, ls, restart_params)
    return restart_nearby(i, ls, restart_params)
end
restart_origin(i, ls, restart_params) = restart_ref(i, ls, restart_params) - ls.b' * ls.ref
restart_ref(i, ls, restart_params) = restart_params.sigma * randn(subspace_dim(ls))
restart_nearby(i, ls, restart_params) = rand(restart_params.nearby)

"""
    subspace_minloss(net, ls::LinearSubspace, a1, a2)

Minimize loss in the subspace orthogonal to `ls` with the point in `ls` fixed to `ls.ref + a1 * ls.v1 + a2 * ls.v2`.
"""
function subspace_minloss(net, ls::LinearSubspace, a1, a2;
        restarts = 3*subspace_dim(ls), sigma = 1e-2, restart_params = (; sigma),
        restart_strategy = restart_ref_and_origin, optim_options = Optim.Options(time_limit = 1),
        finetune = true, finetune_options = Optim.Options(time_limit = 1))
    v = ls.ref + a1 * ls.v1 + a2 * ls.v2
    _grad = (G, x) -> _gradient!(G, ls, net, x, v)
    _loss = x -> MLPGradientFlow._loss(ls, net, x, v)
    best_loss = Inf
    best_x = zeros(subspace_dim(ls))
    converged = false
    for i in 1:restarts
        try
            sol = Optim.optimize(_loss, _grad, restart_strategy(i, ls, restart_params), LBFGS(), optim_options)
            x = Optim.minimizer(sol)
            loss = _loss(x)
            if loss ≤ best_loss
                best_loss = loss
                best_x = x
                converged = Optim.converged(sol)
            end
        catch e
            isa(e, InterruptException) && rethrow(e)
            @warn e
        end
    end
    if finetune
        try
            sol = Optim.optimize(_loss, _grad, best_x, LBFGS(), finetune_options)
            best_x = Optim.minimizer(sol)
            best_loss = _loss(best_x)
            converged = Optim.converged(sol)
        catch e
            isa(e, InterruptException) && rethrow(e)
            @warn e
        end
    end
    (; loss = best_loss, p = ls.b * best_x + v, best_x, delta_loss = _loss(zero(best_x)) - best_loss, converged)
end
function nearby_points(results, ls, a1, a2, n = subspace_dim(ls))
    isempty(results) && return [zeros(subspace_dim(ls))]
    getproperty.(sort(results, by = x -> abs(x.a1 - a1) + abs(x.a2 - a2))[1:min(length(results), n)], :best_x)
end
function subspace_minloss_on_grid(net, ls, grid; sigma = 1e-3, n = 3, results = [], reverse_finetune = true, kwargs...)
    for (a1, a2) in grid
        res = subspace_minloss(net, ls, a1, a2;
                               restart_strategy = restart_ref_and_origin_and_nearby,
                               restart_params = (; sigma, nearby = nearby_points(results, ls, a1, a2, n)), kwargs...)
        push!(results, (; a1, a2, res...))
    end
    if reverse_finetune
        for (i, (a1, a2)) in Iterators.reverse(pairs(IndexLinear(), grid))
            res = subspace_minloss(net, ls, a1, a2;
                                   restart_strategy = restart_nearby, restarts = subspace_dim(ls),
                                   restart_params = (; sigma, nearby = nearby_points(results, ls, a1, a2, n)), kwargs...)
            results[i] = (; a1, a2, res...)
        end
    end
    results
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

"""
    cosine_similarity(x, y) = x'*y/(norm(x)*norm(y))
"""
cosine_similarity(x, y) = x'*y/(norm(x)*norm(y))
