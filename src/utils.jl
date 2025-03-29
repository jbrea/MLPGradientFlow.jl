
trajectory_distance(res1::Dict, res2::Dict) = trajectory_distance(res1["trajectory"], res2["trajectory"])
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

function minloss(net, θ, v1, v2, a1, a2)
    N = length(v1)
    _v1 = 2*normalize(v1)
    _v2 = normalize(v2)
    u, = svd([_v1 _v2 zeros(N, N - 2)])
    v = θ + a1*_v1/2 + a2*_v2
    b = u[:, 3:end]
    _grad(G, u) = G .= b' * gradient(net, b * u + v)
    _loss(u) = loss(net, b * u + v)
    sol = Optim.optimize(_loss, _grad, zeros(N-2), LBFGS())
    x = Optim.minimizer(sol)
    (; loss = _loss(x), p = b * x + v, x, delta_loss = _loss(zeros(N-2)) - _loss(x), to_local_coords = p -> u[:, 1:2]' * (θ - p))
end
