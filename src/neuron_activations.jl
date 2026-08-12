###
### Per-neuron activation functions
###
#
# A layer spec entry may give a *tuple* of activation functions, one per neuron:
#
#     Net(layers = ((3, (relu, relu, tanh), true), (1, identity, true)), ...)
#
# `@tturbo` requires a uniform loop body, so we cannot branch on the neuron index
# inside a single kernel. Instead neurons are partitioned into *contiguous runs*
# sharing an activation, and the ordinary kernel is run once per run over a row
# sub-range. The vectorised axis is the sample axis `n`, so restricting the neuron
# axis `m` costs nothing on the SIMD dimension.
#
# Runs are taken *as given*: (relu, tanh, relu, tanh) is four runs of one, not two
# runs of two. Neuron index therefore always equals parameter row index, which keeps
# `params2dict`, the Hessian block layout, `split_neuron`, `grow_net` and
# teacher-student setups untouched. Ordering activations contiguously is faster.
#
# A tuple whose entries are all identical collapses back to the scalar
# representation, so the single-activation fast path is bit-for-bit unchanged.

"""
    ActGroup(f, lo, hi)

A maximal run of neurons `lo:hi` sharing activation function `f`.
"""
struct ActGroup{F}
    f::F
    lo::Int
    hi::Int
end

"""
    NeuronActivations(fs::Tuple)

Per-neuron activation functions. `fs` holds one activation per neuron; `groups`
holds the run-length encoding used to dispatch kernels.
"""
struct NeuronActivations{FS,G}
    fs::FS
    groups::G
end
function NeuronActivations(fs::Tuple)
    NeuronActivations(fs, activation_groups(fs))
end

"""
    activation_groups(fs::Tuple)

Run-length encode `fs` into a tuple of [`ActGroup`](@ref)s. Adjacent entries are
merged when they are `===`, so grouping never reorders neurons.
"""
function activation_groups(fs::Tuple)
    groups = []
    i = 1
    n = length(fs)
    while i ≤ n
        j = i
        while j < n && fs[j+1] === fs[i]
            j += 1
        end
        push!(groups, ActGroup(fs[i], i, j))
        i = j + 1
    end
    tuple(groups...)
end

n_groups(na::NeuronActivations) = length(na.groups)
activations(na::NeuronActivations) = na.fs
activations(f) = f
function Base.show(io::IO, na::NeuronActivations)
    print(io, "(", join(string.(na.fs), ", "), ")")
end

###
### Constant derivatives
###
#
# Some activations have a derivative that is a compile-time constant, and the
# existing code exploits this by preallocating the buffer and never writing to it
# (see `alloc_a′`/`alloc_a′′`). Per-neuron, that becomes a per-row-range property.
#
# Returning `nothing` means "must be computed".

const_a′(::Any) = nothing
const_a′(::Union{typeof(identity),typeof(softmax)}) = 1

const_a′′(::Any) = nothing
const_a′′(::Union{typeof(identity),typeof(relu),typeof(softmax)}) = 0
const_a′′(::typeof(square)) = 1

###
### Allocation
###

function alloc_a′(na::NeuronActivations, T, k, N, nextbias)
    a′ = zeros(T, k + nextbias, N)
    for grp in na.groups
        c = const_a′(grp.f)
        c === nothing || (a′[grp.lo:grp.hi, :] .= c)
    end
    # the trailing bias row (if any) must stay 0
    StaticStrideArray(a′)
end
function alloc_a′′(na::NeuronActivations, T, k, N)
    a′′ = zeros(T, k, N)
    for grp in na.groups
        c = const_a′′(grp.f)
        c === nothing || (a′′[grp.lo:grp.hi, :] .= c)
    end
    StaticStrideArray(a′′)
end

###
### Range-restricted kernels
###
#
# These mirror the full-layer kernels in MLPGradientFlow.jl, with `indices(w, 1)`
# replaced by an explicit row range. The full-layer methods are deliberately left
# untouched so the single-activation path keeps its exact current codegen.

@inline function _fwd!(a::AbstractMatrix{T}, f, lo, hi, w, input) where {T}
    @tturbo for m in lo:hi, n in indices(input, 2)
        amn = zero(T)
        for k in indices(input, 1)
            amn += w[m, k] * input[k, n]
        end
        a[m, n] = f(amn)
    end
end
@inline function _fwd1!(a::AbstractMatrix{T}, a′, f, lo, hi, w, input) where {T}
    f′ = deriv(f)
    @tturbo for m in lo:hi, n in indices(input, 2)
        amn = zero(T)
        for k in indices(input, 1)
            amn += w[m, k] * input[k, n]
        end
        y = f(amn)
        a′[m, n] = f′(amn, y)
        a[m, n] = y
    end
end
@inline function _fwd2!(a::AbstractMatrix{T}, a′, a′′, f, lo, hi, w, input) where {T}
    f′ = deriv(f)
    f′′ = second_deriv(f)
    @tturbo for m in lo:hi, n in indices(input, 2)
        amn = zero(T)
        for k in indices(input, 1)
            amn += w[m, k] * input[k, n]
        end
        y = f(amn)
        y′ = f′(amn, y)
        a′′[m, n] = f′′(amn, y, y′)
        a′[m, n] = y′
        a[m, n] = y
    end
end

# Which kernel a group needs is decided by `const_a′`/`const_a′′`, which depend only
# on the activation's *type*, so these branches constant-fold away.
@inline function _group!(a, ::Nothing, ::Nothing, grp::ActGroup, w, input)
    _fwd!(a, grp.f, grp.lo, grp.hi, w, input)
end
@inline function _group!(a, a′, ::Nothing, grp::ActGroup, w, input)
    if const_a′(grp.f) === nothing
        _fwd1!(a, a′, grp.f, grp.lo, grp.hi, w, input)
    else
        _fwd!(a, grp.f, grp.lo, grp.hi, w, input)
    end
end
@inline function _group!(a, a′, a′′, grp::ActGroup, w, input)
    if const_a′(grp.f) === nothing && const_a′′(grp.f) === nothing
        _fwd2!(a, a′, a′′, grp.f, grp.lo, grp.hi, w, input)
    elseif const_a′(grp.f) === nothing
        _fwd1!(a, a′, grp.f, grp.lo, grp.hi, w, input)
    else
        _fwd!(a, grp.f, grp.lo, grp.hi, w, input)
    end
end

@inline _groups!(a, a′, a′′, ::Tuple{}, w, input) = nothing
@inline function _groups!(a, a′, a′′, groups::Tuple, w, input)
    _group!(a, a′, a′′, first(groups), w, input)
    _groups!(a, a′, a′′, Base.tail(groups), w, input)
end

@inline A_mul_B!(a::AbstractMatrix, na::NeuronActivations, w, input) =
    _groups!(a, nothing, nothing, na.groups, w, input)
@inline A_mul_B!(a::AbstractMatrix, a′, na::NeuronActivations, w, input) =
    _groups!(a, a′, nothing, na.groups, w, input)
@inline A_mul_B!(a::AbstractMatrix, a′, a′′, na::NeuronActivations, w, input) =
    _groups!(a, a′, a′′, na.groups, w, input)

###
### Spec normalisation
###

_normalize_activations(f, k, i, nlayers) = f
function _normalize_activations(fs::Tuple, k, i, nlayers)
    if i == nlayers
        error("Per-neuron activation functions are currently supported in hidden layers only, but layer $i is the output layer. Pass a single activation function instead.")
    end
    if length(fs) ≠ k
        error("Layer $i declares $k neurons but got $(length(fs)) activation functions. Provide exactly one activation per neuron.")
    end
    if any(f -> f === softmax, fs)
        error("`softmax` acts across a whole layer and cannot be used as a per-neuron activation function (layer $i).")
    end
    # all identical -> fall back to the scalar representation, keeping the existing
    # fast path (and its codegen) exactly as it was
    allequal(fs) ? first(fs) : NeuronActivations(fs)
end

# Used by `_layerextract` when converting results to a dict / pickle. A per-neuron
# layer must serialise as a *list* of names, not `string((relu, tanh))`.
_actname(f) = string(f)
_actname(fs::Tuple) = [string(f) for f in fs]
_actname(na::NeuronActivations) = [string(f) for f in na.fs]

has_neuron_activations(::Any) = false
has_neuron_activations(::NeuronActivations) = true
has_neuron_activations(net::Net) = any(l -> has_neuron_activations(l.f), net.layers)
