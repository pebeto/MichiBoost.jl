abstract type LossFunction end

struct RMSELoss <: LossFunction end

struct MAELoss <: LossFunction end

struct LoglossLoss <: LossFunction end

struct MultiClassLoss <: LossFunction
    n_classes::Int
end

"""
    Pool

Data container for training and prediction.

Stores numerical features as a `Float64` matrix, categorical features as
integer-encoded vectors, labels, sample weights, and feature metadata.
String columns are automatically detected as categorical.

Construct with `Pool(data; label, cat_features, ...)`.

# Fields
- `n_samples::Int` — number of rows
- `n_features::Int` — total number of columns (numerical + categorical)
- `label` — target vector (`Float64`), or `nothing`
- `weight` — sample weights, or `nothing`
- `feature_names` — column names as `Symbol`s
"""
mutable struct Pool
    features_numerical::Matrix{Float64}
    features_categorical::Vector{Vector{UInt32}}
    cat_mapping::Vector{Dict{Any,UInt32}}
    label::Union{Vector{Float64},Nothing}
    label_mapping::Union{Dict{Any,Float64},Nothing}
    label_classes::Union{Vector,Nothing}
    feature_names::Vector{Symbol}
    numerical_feature_indices::Vector{Int}
    categorical_feature_indices::Vector{Int}
    n_samples::Int
    n_features::Int
    weight::Union{Vector{Float64},Nothing}
    group_id::Union{Vector{Any},Nothing}
end

struct QuantizedFeatures
    bins::Matrix{UInt16}
    borders::Vector{Vector{Float64}}
    n_bins::Vector{Int}
end

struct OrderedTargetEncoder
    prior::Float64
    alpha::Float64
    category_stats::Vector{Dict{UInt32,Tuple{Float64,Int}}}
end

struct SymmetricTree
    depth::Int
    split_feature_indices::Vector{Int}
    split_feature_types::Vector{Symbol}
    split_thresholds::Vector{Float64}
    leaf_values::Vector{Float64}
end

struct SymmetricTreeMultiClass
    depth::Int
    split_feature_indices::Vector{Int}
    split_feature_types::Vector{Symbol}
    split_thresholds::Vector{Float64}
    leaf_values::Matrix{Float64}
end

# Concrete element type for leaf groups.  `_apply_split!` returns a vector of
# these; keeping the element type concrete avoids dynamic dispatch on every
# `group[k]` / `length(group)` in the hot loops — a `Vector{Any}` here cost
# ~800k boxed-Int allocations per boosting round.
const LeafGroupView = SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}

struct SplitCandidate
    feature_index::Int
    # Plain Bool keeps SplitCandidate isbits so Vector{SplitCandidate}
    # stores elements inline; a Symbol field here was the source of
    # ~200k heap allocations per boosting round.
    is_categorical::Bool
    threshold::Float64
    gain::Float64
end

struct SplitBuffers
    total_g::Vector{Float64}         # (max_leaves,) per-thread totals
    total_h::Vector{Float64}
    total_n::Vector{Int}
    left_g::Vector{Float64}
    left_h::Vector{Float64}
    left_c::Vector{Int}
    indices::Vector{Int}
    indices_tmp::Vector{Int}
    # Leaf-local compact arrays for cache-friendly access
    local_gradients::Vector{Float64}
    local_hessians::Vector{Float64}
    local_bins::Vector{UInt16}
    local_cat_values::Vector{Float64}
    # Scratch to hold one parent histogram row during subtraction trick
    parent_hist_g_scratch::Vector{Float64}
    parent_hist_h_scratch::Vector{Float64}
    parent_hist_c_scratch::Vector{Int}
end

function SplitBuffers(max_leaves::Int, max_bins::Int, n_samples::Int)
    return SplitBuffers(
        zeros(Float64, max_leaves),
        zeros(Float64, max_leaves),
        zeros(Int, max_leaves),
        zeros(Float64, max_leaves),
        zeros(Float64, max_leaves),
        zeros(Int, max_leaves),
        zeros(Int, n_samples),
        zeros(Int, n_samples),
        zeros(Float64, n_samples),
        zeros(Float64, n_samples),
        zeros(UInt16, n_samples),
        zeros(Float64, n_samples),
        zeros(Float64, max_bins),
        zeros(Float64, max_bins),
        zeros(Int, max_bins),
    )
end

"""
    HistCache

Per-feature histogram cache that persists across levels within a single
tree build. Enables the histogram subtraction trick: when a parent leaf
splits into two children, only the smaller child's histogram is built
from scratch; the larger child's histogram is derived as
`parent_hist - smaller_hist`, typically halving the per-level histogram
construction cost.

Histograms are laid out per feature with rows indexed by leaf position in
`leaf_groups`. A `_valid` flag marks whether the cached entry holds the
previous level's histogram (safe to use for subtraction); a `_filled`
flag tracks features written at the current level and is rotated into
`_valid` between levels via `rotate_hist_cache!`.
"""
mutable struct HistCache
    num_hist_g::Vector{Matrix{Float64}}
    num_hist_h::Vector{Matrix{Float64}}
    num_hist_c::Vector{Matrix{Int}}
    num_hist_valid::Vector{Bool}
    num_hist_filled::Vector{Bool}

    cat_hist_g::Vector{Matrix{Float64}}
    cat_hist_h::Vector{Matrix{Float64}}
    cat_hist_c::Vector{Matrix{Int}}
    cat_hist_valid::Vector{Bool}
    cat_hist_filled::Vector{Bool}
end

function HistCache(
    max_leaves::Int,
    num_n_bins::AbstractVector{Int},
    cat_sorted_vals::AbstractVector{<:AbstractVector},
)
    n_num = length(num_n_bins)
    n_cat = length(cat_sorted_vals)
    num_hist_g = [zeros(Float64, max_leaves, num_n_bins[j] + 1) for j in 1:n_num]
    num_hist_h = [zeros(Float64, max_leaves, num_n_bins[j] + 1) for j in 1:n_num]
    num_hist_c = [zeros(Int, max_leaves, num_n_bins[j] + 1) for j in 1:n_num]
    cat_hist_g = [zeros(Float64, max_leaves, max(length(cat_sorted_vals[j]), 1)) for j in 1:n_cat]
    cat_hist_h = [zeros(Float64, max_leaves, max(length(cat_sorted_vals[j]), 1)) for j in 1:n_cat]
    cat_hist_c = [zeros(Int, max_leaves, max(length(cat_sorted_vals[j]), 1)) for j in 1:n_cat]
    return HistCache(
        num_hist_g,
        num_hist_h,
        num_hist_c,
        fill(false, n_num),
        fill(false, n_num),
        cat_hist_g,
        cat_hist_h,
        cat_hist_c,
        fill(false, n_cat),
        fill(false, n_cat),
    )
end

function reset_hist_cache!(cache::HistCache)
    fill!(cache.num_hist_valid, false)
    fill!(cache.num_hist_filled, false)
    fill!(cache.cat_hist_valid, false)
    fill!(cache.cat_hist_filled, false)
    return cache
end

function rotate_hist_cache!(cache::HistCache)
    @inbounds for j in eachindex(cache.num_hist_valid)
        cache.num_hist_valid[j] = cache.num_hist_filled[j]
        cache.num_hist_filled[j] = false
    end
    @inbounds for j in eachindex(cache.cat_hist_valid)
        cache.cat_hist_valid[j] = cache.cat_hist_filled[j]
        cache.cat_hist_filled[j] = false
    end
    return cache
end

struct SplitBuffersMC
    total_g::Matrix{Float64}         # (max_leaves × n_classes) per-thread totals
    total_h::Matrix{Float64}
    total_n::Vector{Int}
    left_g::Matrix{Float64}          # (max_leaves × n_classes)
    left_h::Matrix{Float64}
    left_c::Vector{Int}
    indices::Vector{Int}
    indices_tmp::Vector{Int}
    # Leaf-local compact arrays for cache-friendly access
    local_gradients_mc::Matrix{Float64}  # (n_samples × n_classes)
    local_hessians_mc::Matrix{Float64}
    local_bins::Vector{UInt16}
    local_cat_values::Vector{Float64}
    # Scratch for one parent histogram row during subtraction trick
    parent_hist_g_scratch::Matrix{Float64}  # (max_bins × n_classes)
    parent_hist_h_scratch::Matrix{Float64}
    parent_hist_c_scratch::Vector{Int}      # (max_bins,)
end

function SplitBuffersMC(max_leaves::Int, max_bins::Int, n_classes::Int, n_samples::Int)
    return SplitBuffersMC(
        zeros(Float64, max_leaves, n_classes),
        zeros(Float64, max_leaves, n_classes),
        zeros(Int, max_leaves),
        zeros(Float64, max_leaves, n_classes),
        zeros(Float64, max_leaves, n_classes),
        zeros(Int, max_leaves),
        zeros(Int, n_samples),
        zeros(Int, n_samples),
        zeros(Float64, n_samples, n_classes),
        zeros(Float64, n_samples, n_classes),
        zeros(UInt16, n_samples),
        zeros(Float64, n_samples),
        zeros(Float64, max_bins, n_classes),
        zeros(Float64, max_bins, n_classes),
        zeros(Int, max_bins),
    )
end

"""
    HistCacheMC

Multiclass counterpart of `HistCache`. Per-feature histograms are stored
as 3D arrays of shape `(max_leaves, n_bins, n_classes)` for gradients
and hessians, plus 2D `(max_leaves, n_bins)` for counts (counts are
class-independent). See `HistCache` docstring for the subtraction-trick
machinery; the layout choice keeps the per-class inner loop contiguous.
"""
mutable struct HistCacheMC
    num_hist_g::Vector{Array{Float64,3}}
    num_hist_h::Vector{Array{Float64,3}}
    num_hist_c::Vector{Matrix{Int}}
    num_hist_valid::Vector{Bool}
    num_hist_filled::Vector{Bool}

    cat_hist_g::Vector{Array{Float64,3}}
    cat_hist_h::Vector{Array{Float64,3}}
    cat_hist_c::Vector{Matrix{Int}}
    cat_hist_valid::Vector{Bool}
    cat_hist_filled::Vector{Bool}
end

function HistCacheMC(
    max_leaves::Int,
    num_n_bins::AbstractVector{Int},
    cat_sorted_vals::AbstractVector{<:AbstractVector},
    n_classes::Int,
)
    n_num = length(num_n_bins)
    n_cat = length(cat_sorted_vals)
    num_hist_g = [zeros(Float64, max_leaves, num_n_bins[j] + 1, n_classes) for j in 1:n_num]
    num_hist_h = [zeros(Float64, max_leaves, num_n_bins[j] + 1, n_classes) for j in 1:n_num]
    num_hist_c = [zeros(Int, max_leaves, num_n_bins[j] + 1) for j in 1:n_num]
    cat_hist_g = [
        zeros(Float64, max_leaves, max(length(cat_sorted_vals[j]), 1), n_classes) for j in 1:n_cat
    ]
    cat_hist_h = [
        zeros(Float64, max_leaves, max(length(cat_sorted_vals[j]), 1), n_classes) for j in 1:n_cat
    ]
    cat_hist_c = [zeros(Int, max_leaves, max(length(cat_sorted_vals[j]), 1)) for j in 1:n_cat]
    return HistCacheMC(
        num_hist_g,
        num_hist_h,
        num_hist_c,
        fill(false, n_num),
        fill(false, n_num),
        cat_hist_g,
        cat_hist_h,
        cat_hist_c,
        fill(false, n_cat),
        fill(false, n_cat),
    )
end

function reset_hist_cache!(cache::HistCacheMC)
    fill!(cache.num_hist_valid, false)
    fill!(cache.num_hist_filled, false)
    fill!(cache.cat_hist_valid, false)
    fill!(cache.cat_hist_filled, false)
    return cache
end

function rotate_hist_cache!(cache::HistCacheMC)
    @inbounds for j in eachindex(cache.num_hist_valid)
        cache.num_hist_valid[j] = cache.num_hist_filled[j]
        cache.num_hist_filled[j] = false
    end
    @inbounds for j in eachindex(cache.cat_hist_valid)
        cache.cat_hist_valid[j] = cache.cat_hist_filled[j]
        cache.cat_hist_filled[j] = false
    end
    return cache
end

"""
    MichiBoostModel

The fitted model produced by `train` (or stored inside a
`MichiBoostRegressor` / `MichiBoostClassifier` after
`fit!`).  Contains the tree ensemble and everything needed for
prediction.

Not typically constructed directly — use `fit!` or `train` instead.
"""
mutable struct MichiBoostModel
    trees::Union{Vector{SymmetricTree},Vector{SymmetricTreeMultiClass}}
    learning_rate::Float64
    initial_pred::Union{Float64,Vector{Float64}}
    loss_name::String
    encoder::Union{OrderedTargetEncoder,Nothing}
    borders::Vector{Vector{Float64}}
    feature_names::Vector{Symbol}
    n_classes::Int
    class_labels::Vector
    is_multiclass::Bool
    numerical_feature_indices::Vector{Int}
    categorical_feature_indices::Vector{Int}
end

"""
    MichiBoostRegressor

A gradient-boosted regression model using symmetric (oblivious) decision trees.

Create with `MichiBoostRegressor(; kwargs...)`, train with
`fit!`, and generate predictions with `predict`.

After training, the fitted `MichiBoostModel` is accessible via the
`.model` field.
"""
mutable struct MichiBoostRegressor
    params::Dict{Symbol,Any}
    model::Union{MichiBoostModel,Nothing}
end

"""
    MichiBoostClassifier

A gradient-boosted classification model using symmetric (oblivious) decision
trees.  Supports binary (Logloss) and multi-class (Softmax) targets.

Create with `MichiBoostClassifier(; kwargs...)`, train with
`fit!`, and generate predictions with `predict`,
`predict_proba`, or `predict_classes`.

After training, the fitted `MichiBoostModel` is accessible via the
`.model` field.
"""
mutable struct MichiBoostClassifier
    params::Dict{Symbol,Any}
    model::Union{MichiBoostModel,Nothing}
end

const MichiBoostWrapper = Union{MichiBoostRegressor,MichiBoostClassifier}
