# Custom Loss Functions

```@meta
CurrentModule = MichiBoost
```

MichiBoost.jl ships with the standard losses (`Losses.RMSE`, `Losses.MAE`,
`Losses.Logloss`, `Losses.MultiClass`), but you can plug in any loss by
subtyping [`LossFunction`](@ref). Custom losses are supported for all three
tasks: regression, binary classification, and multi-class classification.

## The contract

Subtype [`LossFunction`](@ref) and implement these methods. Two of them are
optional traits with sensible defaults; three are required.

### Required

| Method | Purpose |
|---|---|
| `gradient_hessian!(g, h, lf, y, pred, scratch)` | Fused, in-place gradient + hessian. Writes both `g` and `h`. |
| `initial_prediction(lf, y)` | Starting boosting prediction. Scalar for regression / binary, `Vector{Float64}` of length `n_classes` for multi-class. |
| `loss(lf, y, pred)` | Scalar training loss; used for the verbose log and as the default early-stopping signal. |

### Optional traits

| Trait | Default | When to override |
|---|---|---|
| `task_type(lf)` | `:regression` | Return `:binary` or `:multiclass` to declare the task. Drives label encoding and pred-tensor shape. |
| `link_inverse(lf, raw)` | identity | Applied at prediction time to turn raw boosted scores into probabilities. Sigmoid for binary, row-wise softmax for multi-class. Used by `predict_proba` and `predict_classes`. For `:binary`, must return values in `[0, 1]` — `predict_classes` thresholds at 0.5. |

The wrapper validates that `task_type` matches the wrapper kind:
`:regression` requires [`MichiBoostRegressor`](@ref);
`:binary` and `:multiclass` require [`MichiBoostClassifier`](@ref).

### Buffer shapes

The engine pre-allocates `g`, `h`, and `scratch` and reuses them across
iterations — your `gradient_hessian!` should be allocation-free in the hot
path.

| `task_type` | `g`, `h`, `pred`, `scratch` | `y` |
|---|---|---|
| `:regression` | `Vector{Float64}` of length `n_samples` | `Vector{Float64}` |
| `:binary` | `Vector{Float64}` of length `n_samples` | `Vector{Float64}` (`{0.0, 1.0}`-encoded) |
| `:multiclass` | `Matrix{Float64}` of shape `(n_samples, n_classes)` | one-hot `Matrix{Float64}` |

## Example: Huber loss (regression)

```julia
using MichiBoost
using MichiBoost: LossFunction
using Statistics
import MichiBoost: gradient_hessian!, initial_prediction, loss

struct HuberLoss <: LossFunction
    delta::Float64
end

function gradient_hessian!(g, h, lf::HuberLoss, y, pred, _scratch)
    δ = lf.delta
    @inbounds for i in eachindex(y)
        r = y[i] - pred[i]
        if abs(r) <= δ
            g[i] = r
            h[i] = 1.0
        else
            g[i] = δ * sign(r)
            h[i] = 0.0
        end
    end
    return nothing
end

initial_prediction(::HuberLoss, y) = median(y)

function loss(lf::HuberLoss, y, pred)
    δ = lf.delta
    s = 0.0
    @inbounds for i in eachindex(y)
        r = y[i] - pred[i]
        s += abs(r) <= δ ? 0.5 * r^2 : δ * (abs(r) - 0.5 * δ)
    end
    return s / length(y)
end

model = MichiBoostRegressor(; iterations=100, depth=4, loss_function=HuberLoss(1.0))
fit!(model, X_train, y_train)
ŷ = predict(model, X_test)
```

## Example: custom binary loss

```julia
using MichiBoost
using MichiBoost: LossFunction
using Statistics
import MichiBoost: gradient_hessian!, initial_prediction, loss
import MichiBoost: task_type, link_inverse

struct MyLogloss <: LossFunction end
task_type(::MyLogloss) = :binary

@inline _sig(x) = 1.0 / (1.0 + exp(-x))
link_inverse(::MyLogloss, raw::AbstractVector) = _sig.(raw)

function gradient_hessian!(g, h, ::MyLogloss, y, pred, scratch)
    scratch .= _sig.(pred)
    g .= y .- scratch
    h .= scratch .* (1.0 .- scratch)
    return nothing
end

function initial_prediction(::MyLogloss, y)
    p = clamp(mean(y), 1e-7, 1.0 - 1e-7)
    return log(p / (1.0 - p))
end

function loss(::MyLogloss, y, pred)
    p = clamp.(_sig.(pred), 1e-15, 1.0 - 1e-15)
    return -mean(y .* log.(p) .+ (1.0 .- y) .* log.(1.0 .- p))
end

clf = MichiBoostClassifier(; iterations=100, depth=4, loss_function=MyLogloss())
fit!(clf, X_train, y_train)
predict_proba(clf, X_test)
```

## Example: custom multi-class loss

For multi-class, `g`, `h`, `pred`, and `scratch` are matrices and `y` arrives
as a one-hot `Matrix{Float64}`.

```julia
struct MyMultiClass <: LossFunction end
task_type(::MyMultiClass) = :multiclass

function _row_softmax!(out, raw)
    for i in axes(raw, 1)
        m = maximum(view(raw, i, :))
        s = 0.0
        for j in axes(raw, 2)
            e = exp(raw[i, j] - m); out[i, j] = e; s += e
        end
        for j in axes(raw, 2); out[i, j] /= s; end
    end
    return out
end

link_inverse(::MyMultiClass, raw::AbstractMatrix) = _row_softmax!(similar(raw), raw)

function gradient_hessian!(g, h, ::MyMultiClass, y_onehot, pred, scratch)
    _row_softmax!(scratch, pred)
    g .= y_onehot .- scratch
    h .= scratch .* (1.0 .- scratch)
    return nothing
end

function initial_prediction(::MyMultiClass, y_onehot)
    class_probs = clamp.(vec(mean(y_onehot; dims=1)), 1e-7, 1.0 - 1e-7)
    return log.(class_probs)
end

function loss(::MyMultiClass, y_onehot, pred)
    probs = clamp.(_row_softmax!(similar(pred), pred), 1e-15, 1.0)
    return -mean(sum(y_onehot .* log.(probs); dims=2))
end

clf = MichiBoostClassifier(; iterations=100, depth=4, loss_function=MyMultiClass())
fit!(clf, X_train, y_train)
predict_proba(clf, X_test)
```

## Save / load

Models trained with a custom loss round-trip through [`save_model`](@ref) and
[`load_model`](@ref). The loss's type definition must be in scope when
calling [`load_model`](@ref) — load the package or `include` the file that
defines your loss before deserializing, otherwise JLD2 cannot reconstruct
the instance.
