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

struct SplitCandidate
    feature_index::Int
    feature_type::Symbol
    threshold::Float64
    gain::Float64
end

struct SplitBuffers
    hist_g::Matrix{Float64}          # (max_leaves × max_bins)
    hist_h::Matrix{Float64}
    hist_c::Matrix{Int}
    total_g::Vector{Float64}         # (max_leaves,)
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
end

function SplitBuffers(max_leaves::Int, max_bins::Int, n_samples::Int)
    return SplitBuffers(
        zeros(Float64, max_leaves, max_bins),
        zeros(Float64, max_leaves, max_bins),
        zeros(Int, max_leaves, max_bins),
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
    )
end

struct SplitBuffersMC
    hist_g::Array{Float64,3}         # (max_leaves × max_bins × n_classes)
    hist_h::Array{Float64,3}
    hist_c::Matrix{Int}              # (max_leaves × max_bins)
    total_g::Matrix{Float64}         # (max_leaves × n_classes)
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
end

function SplitBuffersMC(max_leaves::Int, max_bins::Int, n_classes::Int, n_samples::Int)
    return SplitBuffersMC(
        zeros(Float64, max_leaves, max_bins, n_classes),
        zeros(Float64, max_leaves, max_bins, n_classes),
        zeros(Int, max_leaves, max_bins),
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
    )
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
