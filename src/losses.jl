@inline _sigmoid(x::Real) = one(x) / (one(x) + exp(-x))

# Public traits for custom losses
#
# `task_type(loss)` decides how the engine routes a custom loss:
#   :regression  → vector pred / vector y, no class encoding (default)
#   :binary      → label-to-Float64 mapping, vector pred / vector y
#   :multiclass  → one-hot y, matrix pred, matrix g/h/scratch
#
# `link_inverse(loss, raw)` is applied at prediction time to turn raw boosted
# scores into probabilities. Identity by default; sigmoid for binary builtins;
# row-wise softmax for multiclass builtins. Custom binary losses should make
# this return probabilities in [0,1] so that `predict_classes` can threshold
# at 0.5.

task_type(::LossFunction) = :regression
link_inverse(::LossFunction, raw) = raw

task_type(::RMSELoss) = :regression
task_type(::MAELoss) = :regression
task_type(::LoglossLoss) = :binary
task_type(::MultiClassLoss) = :multiclass

link_inverse(::LoglossLoss, raw::AbstractVector) = _sigmoid.(raw)
link_inverse(::MultiClassLoss, raw::AbstractMatrix) = _softmax_matrix(raw)

function _softmax(logits::AbstractVector)
    m = maximum(logits)
    e = exp.(logits .- m)
    return e ./ sum(e)
end

function _softmax_matrix(logits::AbstractMatrix)
    result = similar(logits)
    for i in axes(logits, 1)
        result[i, :] = _softmax(view(logits, i, :))
    end
    return result
end

# In-place softmax: writes softmax of each row of `logits` into `out`.
# Avoids the per-row allocations that _softmax / _softmax_matrix do.
function _softmax_matrix!(out::AbstractMatrix, logits::AbstractMatrix)
    n_rows, n_cols = size(logits)
    @inbounds for i in 1:n_rows
        m = logits[i, 1]
        for j in 2:n_cols
            x = logits[i, j]
            x > m && (m = x)
        end
        s = 0.0
        for j in 1:n_cols
            e = exp(logits[i, j] - m)
            out[i, j] = e
            s += e
        end
        inv_s = 1.0 / s
        for j in 1:n_cols
            out[i, j] *= inv_s
        end
    end
    return out
end

loss(::RMSELoss, y, pred) = sqrt(mean((y .- pred) .^ 2))
initial_prediction(::RMSELoss, y) = mean(y)

loss(::MAELoss, y, pred) = mean(abs.(y .- pred))
initial_prediction(::MAELoss, y) = median(y)

# Weighted median over a sequence of values and non-negative weights.  Returns
# the smallest `vals[i]` such that cumulative weight crosses half the total.
function weighted_median(vals::AbstractVector{Float64}, w::AbstractVector{Float64})
    n = length(vals)
    n == 0 && return 0.0
    n == 1 && return vals[1]
    perm = sortperm(vals)
    total = 0.0
    @inbounds for i in 1:n
        total += w[i]
    end
    half = total / 2
    cum = 0.0
    @inbounds for j in 1:n
        cum += w[perm[j]]
        cum >= half && return vals[perm[j]]
    end
    @inbounds return vals[perm[end]]
end

function loss(::LoglossLoss, y, pred)
    p = clamp.(_sigmoid.(pred), 1e-15, 1.0 - 1e-15)
    return -mean(y .* log.(p) .+ (1.0 .- y) .* log.(1.0 .- p))
end

function initial_prediction(::LoglossLoss, y)
    p = clamp(mean(y), 1e-7, 1.0 - 1e-7)
    return log(p / (1.0 - p))
end

function loss(::MultiClassLoss, y_onehot::AbstractMatrix, pred::AbstractMatrix)
    probs = clamp.(_softmax_matrix(pred), 1e-15, 1.0)
    return -mean(sum(y_onehot .* log.(probs); dims=2))
end

function initial_prediction(::MultiClassLoss, y_onehot::AbstractMatrix)
    class_probs = clamp.(vec(mean(y_onehot; dims=1)), 1e-7, 1.0 - 1e-7)
    return log.(class_probs)
end

# Fused in-place gradient + hessian.  Writes both outputs in a single pass
# and, for losses that need an intermediate (softmax/sigmoid), uses the
# caller's `scratch` buffer so nothing is allocated per boosting round.
# RMSE/MAE ignore `scratch`.

function gradient_hessian!(
    g::AbstractVector,
    h::AbstractVector,
    ::RMSELoss,
    y::AbstractVector,
    pred::AbstractVector,
    _scratch,
)
    g .= y .- pred
    fill!(h, 1.0)
    return nothing
end

function gradient_hessian!(
    g::AbstractVector,
    h::AbstractVector,
    ::MAELoss,
    y::AbstractVector,
    pred::AbstractVector,
    _scratch,
)
    g .= sign.(y .- pred)
    fill!(h, 1.0)
    return nothing
end

function gradient_hessian!(
    g::AbstractVector,
    h::AbstractVector,
    ::LoglossLoss,
    y::AbstractVector,
    pred::AbstractVector,
    scratch::AbstractVector,
)
    scratch .= 1.0 ./ (1.0 .+ exp.(.-pred))
    g .= y .- scratch
    h .= scratch .* (1.0 .- scratch)
    return nothing
end

function gradient_hessian!(
    g::AbstractMatrix,
    h::AbstractMatrix,
    ::MultiClassLoss,
    y_onehot::AbstractMatrix,
    pred::AbstractMatrix,
    scratch::AbstractMatrix,
)
    _softmax_matrix!(scratch, pred)
    g .= y_onehot .- scratch
    h .= scratch .* (1.0 .- scratch)
    return nothing
end

function make_loss(name::AbstractString)
    upper = uppercase(name)
    upper == "RMSE" && return RMSELoss()
    upper == "MAE" && return MAELoss()
    upper in ("LOGLOSS", "CROSSENTROPY") && return LoglossLoss()
    upper in ("MULTICLASS", "MULTILOGLOSS") && return MultiClassLoss()
    return error("Unknown loss function: $name. Supported: RMSE, MAE, Logloss, MultiClass")
end

"""
    MichiBoost.Losses

Singleton tag types for the `loss_function` keyword argument. Pass the bare
type instead of a string. Typos in the tag form surface as `UndefVarError` at
parse time, where strings would only fail at training.

```julia
model = MichiBoostRegressor(; loss_function=Losses.RMSE)
```

The CatBoost-style string form (`"RMSE"`) and the matching Symbol (`:RMSE`)
are also accepted at the wrapper boundary.
"""
module Losses

abstract type LossKind end

struct RMSE <: LossKind end
struct MAE <: LossKind end
struct Logloss <: LossKind end
struct CrossEntropy <: LossKind end
struct MultiClass <: LossKind end
struct MultiLogloss <: LossKind end

end  # module Losses

using .Losses: LossKind

# Canonical string name for each `Losses.*` tag, consumed by the string-keyed
# `make_loss` dispatcher.
_loss_name(::Type{Losses.RMSE}) = "RMSE"
_loss_name(::Type{Losses.MAE}) = "MAE"
_loss_name(::Type{Losses.Logloss}) = "Logloss"
_loss_name(::Type{Losses.CrossEntropy}) = "CrossEntropy"
_loss_name(::Type{Losses.MultiClass}) = "MultiClass"
_loss_name(::Type{Losses.MultiLogloss}) = "MultiLogloss"
