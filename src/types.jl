"""
    LossFunction

Abstract supertype for boosting loss functions. Subtype this to plug a custom
loss into [`MichiBoostRegressor`](@ref) (regression) or
[`MichiBoostClassifier`](@ref) (binary or multi-class).

```julia
import MichiBoost: gradient_hessian!, initial_prediction, loss
# Optional traits (defaults: :regression and identity)
import MichiBoost: task_type, link_inverse

struct MyLoss <: MichiBoost.LossFunction end

# Optional: declare the task. One of :regression (default), :binary, :multiclass.
# Drives label encoding and the shape of the buffers passed to gradient_hessian!.
task_type(::MyLoss) = :regression

# Optional: applied at predict time to turn raw scores into probabilities.
# Identity by default; sigmoid for :binary, row-wise softmax for :multiclass.
link_inverse(::MyLoss, raw) = raw

# Required: fused, in-place gradient + hessian. Buffer shapes match task_type:
#   :regression, :binary   → vectors  (length n_samples)
#   :multiclass            → matrices (n_samples × n_classes)
# `scratch` is a same-shape buffer the engine reuses; ignore it if unused.
gradient_hessian!(g, h, ::MyLoss, y, pred, scratch) = ...

# Required: starting boosting prediction. Scalar for regression and binary;
# `Vector{Float64}` of length n_classes for multiclass.
initial_prediction(::MyLoss, y) = ...

# Required: scalar training loss. Used for the verbose log and as the default
# early-stopping signal.
loss(::MyLoss, y, pred) = ...
```

The wrapper validates that `task_type` matches the wrapper kind:
`:regression` requires [`MichiBoostRegressor`](@ref); `:binary` and
`:multiclass` require [`MichiBoostClassifier`](@ref). See the
"Custom Loss Functions" guide for full examples.
"""
abstract type LossFunction end

struct RMSELoss <: LossFunction end

struct MAELoss <: LossFunction end

struct LoglossLoss <: LossFunction end

struct MultiClassLoss <: LossFunction end

"""
    Pool(data; label=nothing, cat_features=nothing, text_features=nothing,
         feature_names=nothing, weight=nothing)

Data container for training and prediction.

Stores numerical features as a `Float64` matrix, categorical features as
integer-encoded vectors, labels, sample weights, and feature metadata. String
columns are detected as categorical without further configuration.

# Arguments
- `data`: any `Tables.jl`-compatible table (e.g. `DataFrame`, `NamedTuple` of
  vectors) or an `AbstractMatrix`.
- `label`: target vector. Numeric values are kept as-is; string and
  categorical values are encoded to `Float64`.
- `cat_features`: 1-based column indices (`Int`) or column names (`Symbol`
  or `String`) to treat as categorical. String columns are detected even
  without this argument.
- `text_features`: same format as `cat_features`. Columns listed here are
  treated identically to categorical columns.
- `feature_names`: optional vector of column names.
- `weight`: optional per-sample weights (`Vector{<:Real}`).

# Fields
- `n_samples::Int`: number of rows.
- `n_features::Int`: total number of columns (numerical plus categorical).
- `label`: target vector (`Float64`), or `nothing`.
- `weight`: sample weights, or `nothing`.
- `feature_names`: column names as `Symbol`s.

# Examples
```julia
# From a matrix
pool = Pool([1.0 2.0; 3.0 4.0]; label=[0.0, 1.0])

# From a table; `color` is detected as categorical, `size` stays numerical
pool = Pool((color=["red","blue","red"], size=[1.0, 2.0, 3.0]);
            label=[0.0, 1.0, 0.0])

# With per-sample weights
pool = Pool(X; label=y, weight=[1.0, 2.0, 0.5, 1.0])
```
"""
mutable struct Pool
    features_numerical::Matrix{Float64}
    features_categorical::Vector{Vector{UInt32}}
    label::Union{Vector{Float64},Nothing}
    label_classes::Union{Vector,Nothing}
    feature_names::Vector{Symbol}
    numerical_feature_indices::Vector{Int}
    categorical_feature_indices::Vector{Int}
    n_samples::Int
    n_features::Int
    weight::Union{Vector{Float64},Nothing}
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
# `group[k]` / `length(group)` in the hot loops. A `Vector{Any}` here cost
# ~800k boxed-Int allocations per boosting round.
const LeafGroupView = SubArray{Int64,1,Vector{Int64},Tuple{UnitRange{Int64}},true}

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
    cat_hist_g = [
        zeros(Float64, max_leaves, max(length(cat_sorted_vals[j]), 1)) for j in 1:n_cat
    ]
    cat_hist_h = [
        zeros(Float64, max_leaves, max(length(cat_sorted_vals[j]), 1)) for j in 1:n_cat
    ]
    cat_hist_c = [
        zeros(Int, max_leaves, max(length(cat_sorted_vals[j]), 1)) for j in 1:n_cat
    ]
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
    # Class is the first (stride-1) axis on every per-class buffer so the hot
    # `for c in 1:n_classes` inner loops sweep contiguous memory.
    total_g::Matrix{Float64}         # (n_classes × max_leaves)
    total_h::Matrix{Float64}
    total_n::Vector{Int}
    # Precomputed total_g²/(total_h + l2_leaf_reg) per (c, leaf). This is the
    # b-invariant subtractive term in `_sweep_gain_mc`; lifting it out of the
    # bin sweep saves ~n_bins × n_leaves × n_classes divisions per feature.
    total_score::Matrix{Float64}     # (n_classes × max_leaves)
    left_g::Matrix{Float64}          # (n_classes × max_leaves)
    left_h::Matrix{Float64}
    left_c::Vector{Int}
    indices::Vector{Int}
    indices_tmp::Vector{Int}
    parent_hist_g_scratch::Matrix{Float64}  # (n_classes × max_bins)
    parent_hist_h_scratch::Matrix{Float64}
    parent_hist_c_scratch::Vector{Int}      # (max_bins,)
end

function SplitBuffersMC(max_leaves::Int, max_bins::Int, n_classes::Int, n_samples::Int)
    return SplitBuffersMC(
        zeros(Float64, n_classes, max_leaves),
        zeros(Float64, n_classes, max_leaves),
        zeros(Int, max_leaves),
        zeros(Float64, n_classes, max_leaves),
        zeros(Float64, n_classes, max_leaves),
        zeros(Float64, n_classes, max_leaves),
        zeros(Int, max_leaves),
        zeros(Int, n_samples),
        zeros(Int, n_samples),
        zeros(Float64, n_classes, max_bins),
        zeros(Float64, n_classes, max_bins),
        zeros(Int, max_bins),
    )
end

"""
    HistCacheMC

Multiclass counterpart of `HistCache`. Per-feature gradient and hessian
histograms are stored as 3D arrays of shape `(n_classes, max_leaves, n_bins)`,
plus 2D `(max_leaves, n_bins)` for the class-independent counts. Putting the
class axis first makes the hot `for c in 1:n_classes` inner loop stride-1.
See `HistCache` for the subtraction-trick machinery.
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
    num_hist_g = [zeros(Float64, n_classes, max_leaves, num_n_bins[j] + 1) for j in 1:n_num]
    num_hist_h = [zeros(Float64, n_classes, max_leaves, num_n_bins[j] + 1) for j in 1:n_num]
    num_hist_c = [zeros(Int, max_leaves, num_n_bins[j] + 1) for j in 1:n_num]
    cat_hist_g = [
        zeros(Float64, n_classes, max_leaves, max(length(cat_sorted_vals[j]), 1)) for
        j in 1:n_cat
    ]
    cat_hist_h = [
        zeros(Float64, n_classes, max_leaves, max(length(cat_sorted_vals[j]), 1)) for
        j in 1:n_cat
    ]
    cat_hist_c = [
        zeros(Int, max_leaves, max(length(cat_sorted_vals[j]), 1)) for j in 1:n_cat
    ]
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

The fitted model stored inside a [`MichiBoostRegressor`](@ref) or
[`MichiBoostClassifier`](@ref) after [`fit!`](@ref). Holds the tree ensemble,
the per-feature quantization borders, the categorical target-statistics
encoder, and any state recorded by early stopping.

Construct it by calling [`fit!`](@ref) on a wrapper, or by deserialising a
saved file with [`load_model`](@ref). Once you hold a `MichiBoostModel`, pass
it as the first argument to [`predict`](@ref), [`predict_classes`](@ref), or
[`feature_importance`](@ref) the same way you would a wrapper.
"""
mutable struct MichiBoostModel
    trees::Union{Vector{SymmetricTree},Vector{SymmetricTreeMultiClass}}
    learning_rate::Float64
    initial_pred::Union{Float64,Vector{Float64}}
    encoder::Union{OrderedTargetEncoder,Nothing}
    borders::Vector{Vector{Float64}}
    feature_names::Vector{Symbol}
    n_classes::Int
    class_labels::Vector
    is_multiclass::Bool
    numerical_feature_indices::Vector{Int}
    categorical_feature_indices::Vector{Int}
    # Early-stopping state surfaced to the public API. `best_iteration` is the
    # iteration whose eval-metric value was best (1-indexed) when early stopping
    # was active; otherwise it equals the number of trees actually built.
    # `best_score` is the eval-metric value at `best_iteration`, or `nothing`
    # when early stopping was not active.
    best_iteration::Int
    best_score::Union{Float64,Nothing}
    # Set when the user trained with a custom `LossFunction`. `predict` and
    # `predict_proba` route through `link_inverse(custom_loss, raw)` instead of
    # the built-in sigmoid/softmax when this is non-`nothing`. Required for
    # round-trip through save_model / load_model.
    custom_loss::Union{LossFunction,Nothing}
end

"""
    MichiBoostRegressor(; kwargs...) -> MichiBoostRegressor

A gradient-boosted regression model using symmetric (oblivious) decision trees.

Train with [`fit!`](@ref) and generate predictions with [`predict`](@ref).
After training, the fitted [`MichiBoostModel`](@ref) is accessible via the
`.model` field.

# Keyword Arguments
- `iterations::Int=1000`: number of boosting rounds (trees to build).
- `learning_rate::Float64=0.03`: step-size shrinkage applied to each tree.
- `depth::Int=6`: depth of each symmetric tree.
- `l2_leaf_reg::Float64=3.0`: L2 regularisation on leaf values.
- `loss_function::Union{Type{<:Losses.LossKind},String,Symbol,LossFunction}="RMSE"`:
  built-in regression options: `Losses.RMSE`, `Losses.MAE`; classification
  options: `Losses.Logloss`, `Losses.MultiClass` (auto-detected on classifier
  targets). Pass a custom [`LossFunction`](@ref) instance to use your own loss.
  Custom losses declare their task via `task_type(::MyLoss)`; the wrapper
  enforces that `:regression` losses go to `MichiBoostRegressor` and `:binary`
  or `:multiclass` losses go to `MichiBoostClassifier`. See
  [`LossFunction`](@ref).
- `border_count::Int=254`: max quantization borders per numerical feature.
- `min_data_in_leaf::Int=1`: minimum samples required in a leaf.
- `random_seed::Union{Int,Nothing}=nothing`: seed for reproducibility.
- `verbose::Bool=false`: print training progress.
- `boosting_type::Union{Type{<:BoostingTypes.BoostingType},String,Symbol}="Ordered"`:
  `BoostingTypes.Ordered` uses a random permutation when computing categorical
  target statistics, reducing leakage. `BoostingTypes.Plain` encodes on the
  full training set. Gradient computation is plain in both modes.
- `early_stopping_rounds::Union{Int,Nothing}=nothing`: stop after this many
  rounds without improvement on `eval_set`.
- `eval_metric::Union{Type{<:Metric},String,Symbol,Nothing}=nothing`: metric
  used to drive the early-stopping comparison. `nothing` (default) falls back
  to the training loss. Supported tags: `Metrics.RMSE`, `Metrics.MAE`,
  `Metrics.R2` for regression; `Metrics.Logloss`, `Metrics.Accuracy`,
  `Metrics.F1`, `Metrics.AUC` for binary; `Metrics.MultiLogloss`,
  `Metrics.Accuracy` for multi-class. Comparison direction is inferred per
  metric.
- `init_model::Union{MichiBoostModel,Nothing}=nothing`: resume training from
  a previously fitted model. The new fit inherits every tree of `init_model`
  and adds `iterations` more on top. See the "Continuing Training" section in
  the Advanced guide.
- `snapshot_path::Union{AbstractString,Nothing}=nothing`: when set, the
  partial model is serialised to this file every `snapshot_interval`
  iterations and once more after training finishes. Load the file with
  [`load_model`](@ref) and resume training by passing the result as
  `init_model`.
- `snapshot_interval::Int=100`: iterations between snapshot writes. Must be
  `>= 1` when `snapshot_path` is set.
- `monotone_constraints=nothing`: enforce a monotone relationship between a
  numerical feature and the prediction. Pass a `Dict` keyed by feature name or
  1-based column index (`Dict(:price => 1, :age => -1)`), or a length-`n_features`
  vector of signs in column order. `+1` forces the prediction non-decreasing in
  that feature, `-1` non-increasing, `0` leaves it free. Regression and binary
  classification only.

Tag-valued keywords (`loss_function`, `boosting_type`, `eval_metric`) accept
their value as the tag (e.g. `Losses.RMSE`), the matching string (`"RMSE"`),
or the matching Symbol (`:RMSE`); the tag form is checked at parse time, so
typos surface as `UndefVarError` rather than as a runtime error.

# Example
```julia
model = MichiBoostRegressor(; iterations=200, depth=4)
fit!(model, X, y)
ŷ = predict(model, X_test)
```
"""
mutable struct MichiBoostRegressor
    params::Dict{Symbol,Any}
    model::Union{MichiBoostModel,Nothing}
end

"""
    MichiBoostClassifier(; kwargs...) -> MichiBoostClassifier

A gradient-boosted classification model using symmetric (oblivious) decision
trees. Supports binary (Logloss) and multi-class (Softmax) targets; multi-class
is auto-detected when the target has more than two unique values.

Train with [`fit!`](@ref) and generate predictions with [`predict`](@ref),
[`predict_proba`](@ref), or [`predict_classes`](@ref). After training, the
fitted [`MichiBoostModel`](@ref) is accessible via the `.model` field.

# Keyword Arguments
Accepts the same keyword arguments as [`MichiBoostRegressor`](@ref), plus:

- `loss_function::Union{Type{<:Losses.LossKind},String,Symbol}="Logloss"`:
  `Losses.Logloss` for binary, `Losses.MultiClass` for multi-class.
  Multi-class is auto-detected if omitted.
- `class_weights::Union{AbstractDict,Nothing}=nothing`: per-class weights as a
  `Dict(label => weight)`. Multiplied into the per-sample weights at fit
  time. Use it for imbalanced classification when adjusting per-row weights
  via [`Pool`](@ref) is inconvenient.
- `auto_class_weights::Union{Type{<:AutoClassWeights.AutoClassWeightMode},String,Symbol,Nothing}=nothing`:
  automatic class weighting derived from training-label frequencies.
  `AutoClassWeights.Balanced` sets each weight to `n / (n_classes * count[c])`;
  `AutoClassWeights.SqrtBalanced` sets it to `sqrt(n / count[c])`. Mutually
  exclusive with `class_weights`.

`loss_function` and `auto_class_weights` follow the same tag, string, or
Symbol convention as the [`MichiBoostRegressor`](@ref) keywords.

# Example
```julia
model = MichiBoostClassifier(; iterations=200, depth=4)
fit!(model, X, y)
probs = predict_proba(model, X_test)   # probabilities
labels = predict(model, X_test)        # class labels

# Imbalanced binary: upweight the positive class 5×
clf = MichiBoostClassifier(; iterations=200, class_weights=Dict(0.0 => 1.0, 1.0 => 5.0))
fit!(clf, X, y)
```
"""
mutable struct MichiBoostClassifier
    params::Dict{Symbol,Any}
    model::Union{MichiBoostModel,Nothing}
end

const MichiBoostWrapper = Union{MichiBoostRegressor,MichiBoostClassifier}
