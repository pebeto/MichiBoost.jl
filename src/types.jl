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

Construct with [`Pool(data; label, cat_features, ...)`](@ref).

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

"""
    MichiBoostModel

The fitted model produced by [`train`](@ref) (or stored inside a
[`MichiBoostRegressor`](@ref) / [`MichiBoostClassifier`](@ref) after
[`fit!`](@ref)).  Contains the tree ensemble and everything needed for
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
end

"""
    MichiBoostRegressor

A gradient-boosted regression model using symmetric (oblivious) decision trees.

Create with [`MichiBoostRegressor(; kwargs...)`](@ref), train with
[`fit!`](@ref), and generate predictions with [`predict`](@ref).

After training, the fitted [`MichiBoostModel`](@ref) is accessible via the
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

Create with [`MichiBoostClassifier(; kwargs...)`](@ref), train with
[`fit!`](@ref), and generate predictions with [`predict`](@ref),
[`predict_proba`](@ref), or [`predict_classes`](@ref).

After training, the fitted [`MichiBoostModel`](@ref) is accessible via the
`.model` field.
"""
mutable struct MichiBoostClassifier
    params::Dict{Symbol,Any}
    model::Union{MichiBoostModel,Nothing}
end

const MichiBoostWrapper = Union{MichiBoostRegressor,MichiBoostClassifier}
