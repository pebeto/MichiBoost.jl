@inline _sigmoid(x::Real) = one(x) / (one(x) + exp(-x))

# Custom losses subtype `LossFunction` and add methods to the functions below.
# `gradient_hessian!`, `initial_prediction`, and `loss` are required;
# `task_type` and `link_inverse` carry defaults. See docs/src/guide/custom_loss.md.

"""
    gradient_hessian!(g, h, lf::LossFunction, y, pred, scratch) -> nothing

Write the per-sample gradient into `g` and the Hessian into `h`, in place. The
engine calls this once per boosting iteration with buffers it reuses across
iterations, so keep the body allocation-free.

For `:regression` and `:binary` losses, `g`, `h`, `pred`, `scratch`, and `y`
are length-`n_samples` vectors, with `y` encoded as `{0.0, 1.0}` for `:binary`.
For `:multiclass`, all five are `(n_samples, n_classes)` matrices and `y` is
one-hot. Use `scratch` for temporaries.

Implement this for every custom [`LossFunction`](@ref).
"""
function gradient_hessian! end

"""
    initial_prediction(lf::LossFunction, y) -> Float64 or Vector{Float64}

Return the constant score the ensemble starts from before the first tree.
Regression and binary losses return a scalar; multi-class returns a
`Vector{Float64}` of length `n_classes`. `y` arrives as the label vector for
regression and binary, and as the one-hot matrix for multi-class.

Implement this for every custom [`LossFunction`](@ref).
"""
function initial_prediction end

"""
    loss(lf::LossFunction, y, pred) -> Float64

Return the scalar training loss. The engine prints it in the verbose log and
uses it as the default early-stopping signal when `eval_metric` is unset.
`pred` holds raw boosted scores rather than probabilities, so apply your own
link inside `loss` when you need them.

Implement this for every custom [`LossFunction`](@ref).
"""
function loss end

"""
    task_type(lf::LossFunction) -> Symbol

Declare which task the loss drives: `:regression` (the default), `:binary`, or
`:multiclass`. The value selects label encoding and the shape of the gradient,
Hessian, and prediction buffers passed to [`gradient_hessian!`](@ref).
[`fit!`](@ref) checks it against the wrapper: `:regression` needs a
[`MichiBoostRegressor`](@ref); `:binary` and `:multiclass` need a
[`MichiBoostClassifier`](@ref).

Override only when the loss is not regression.
"""
task_type(::LossFunction) = :regression

"""
    link_inverse(lf::LossFunction, raw) -> probabilities

Map raw boosted scores to probabilities at prediction time. The default returns
`raw` unchanged. [`predict_proba`](@ref) and [`predict_classes`](@ref) route
through it, so a `:binary` loss must return values in `[0, 1]`
([`predict_classes`](@ref) thresholds at 0.5) and a `:multiclass` loss must
return one distribution per row.

Override for `:binary` (sigmoid) and `:multiclass` (row-wise softmax) losses.
"""
link_inverse(::LossFunction, raw) = raw

# Leaf-value refinement. A loss whose Hessian is zero (the quantile family)
# cannot use the Newton leaf step `g_sum / (h_sum + λ)`. Such a loss opts in by
# returning `true` here and implementing `leaf_refine_targets!` (fills the
# per-sample value and weight a leaf reduces over) and `leaf_refine_value` (the
# per-leaf reducer). The engine then overrides each leaf's Newton value.
uses_leaf_refinement(::LossFunction) = false

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

# MAE refines each leaf to the weighted median of its in-leaf residuals, since
# the ±1 surrogate gradient drives split-finding but cannot set leaf values.
uses_leaf_refinement(::MAELoss) = true
function leaf_refine_targets!(vals, ws, ::MAELoss, y, pred, weights)
    vals .= y .- pred
    ws .= weights
    return nothing
end
leaf_refine_value(::MAELoss, vals, ws) = weighted_median(vals, ws)

# Weighted q-quantile over a sequence of values and non-negative weights.
# Returns the smallest `vals[i]` whose cumulative weight (in ascending value
# order) reaches `q` of the total. `q = 0.5` recovers the weighted median.
function weighted_quantile(
    vals::AbstractVector{Float64}, w::AbstractVector{Float64}, q::Float64
)
    n = length(vals)
    n == 0 && return 0.0
    n == 1 && return vals[1]
    perm = sortperm(vals)
    total = 0.0
    @inbounds for i in 1:n
        total += w[i]
    end
    thresh = q * total
    cum = 0.0
    @inbounds for j in 1:n
        cum += w[perm[j]]
        cum >= thresh && return vals[perm[j]]
    end
    @inbounds return vals[perm[end]]
end

function weighted_median(vals::AbstractVector{Float64}, w::AbstractVector{Float64})
    return weighted_quantile(vals, w, 0.5)
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

"""
    Huber(delta)

Huber regression loss. Squared error for residuals within `delta`, linear
beyond it, which curbs the pull of outliers while staying smooth near zero.
`delta > 0` sets the crossover. Pass as `loss_function=Huber(1.0)` to a
[`MichiBoostRegressor`](@ref).
"""
struct Huber <: LossFunction
    delta::Float64
    function Huber(delta::Real)
        delta > 0 || error("Huber delta must be > 0; got $delta.")
        return new(Float64(delta))
    end
end

function gradient_hessian!(g, h, lf::Huber, y, pred, _scratch)
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

initial_prediction(::Huber, y) = median(y)

function loss(lf::Huber, y, pred)
    δ = lf.delta
    s = 0.0
    @inbounds for i in eachindex(y)
        r = abs(y[i] - pred[i])
        s += r <= δ ? 0.5 * r * r : δ * (r - 0.5 * δ)
    end
    return s / length(y)
end

"""
    Quantile(alpha)

Quantile (pinball) regression loss for the `alpha`-quantile, `0 < alpha < 1`.
`alpha = 0.5` is the median (equivalent to MAE). Leaf values are refined to the
weighted `alpha`-quantile of the in-leaf residuals. Pass as
`loss_function=Quantile(0.9)` to a [`MichiBoostRegressor`](@ref).
"""
struct Quantile <: LossFunction
    alpha::Float64
    function Quantile(alpha::Real)
        0 < alpha < 1 || error("Quantile alpha must be in (0, 1); got $alpha.")
        return new(Float64(alpha))
    end
end

function gradient_hessian!(g, h, lf::Quantile, y, pred, _scratch)
    α = lf.alpha
    @inbounds for i in eachindex(y)
        g[i] = y[i] > pred[i] ? α : α - 1.0
        h[i] = 1.0
    end
    return nothing
end

initial_prediction(lf::Quantile, y) = quantile(y, lf.alpha)

loss(lf::Quantile, y, pred) = _pinball(y, pred, lf.alpha)

uses_leaf_refinement(::Quantile) = true
function leaf_refine_targets!(vals, ws, ::Quantile, y, pred, weights)
    vals .= y .- pred
    ws .= weights
    return nothing
end
leaf_refine_value(lf::Quantile, vals, ws) = weighted_quantile(vals, ws, lf.alpha)

# Mean pinball loss at level α.
function _pinball(y, pred, α)
    s = 0.0
    @inbounds for i in eachindex(y)
        u = y[i] - pred[i]
        s += u >= 0 ? α * u : (α - 1.0) * u
    end
    return s / length(y)
end

"""
    Expectile(alpha)

Expectile regression loss for the `alpha`-expectile, `0 < alpha < 1`. The
asymmetric squared-error analogue of [`Quantile`](@ref): residuals above the
fit are weighted by `alpha`, those below by `1 - alpha`. `alpha = 0.5` recovers
RMSE. Pass as `loss_function=Expectile(0.9)` to a [`MichiBoostRegressor`](@ref).
"""
struct Expectile <: LossFunction
    alpha::Float64
    function Expectile(alpha::Real)
        0 < alpha < 1 || error("Expectile alpha must be in (0, 1); got $alpha.")
        return new(Float64(alpha))
    end
end

function gradient_hessian!(g, h, lf::Expectile, y, pred, _scratch)
    α = lf.alpha
    @inbounds for i in eachindex(y)
        r = y[i] - pred[i]
        w = r > 0 ? α : 1.0 - α
        g[i] = 2.0 * w * r
        h[i] = 2.0 * w
    end
    return nothing
end

initial_prediction(::Expectile, y) = mean(y)

function loss(lf::Expectile, y, pred)
    α = lf.alpha
    s = 0.0
    @inbounds for i in eachindex(y)
        r = y[i] - pred[i]
        w = r > 0 ? α : 1.0 - α
        s += w * r * r
    end
    return s / length(y)
end

"""
    MAPE()

Mean absolute percentage error: `mean(abs((y - pred) / y))`. Each residual is
scaled by `1 / abs(y)`, so the fit chases relative rather than absolute error.
Leaf values are refined to the weighted median of in-leaf residuals with
weights `1 / abs(y)`. Targets must be non-zero. Pass as
`loss_function=MAPE()` to a [`MichiBoostRegressor`](@ref).
"""
struct MAPE <: LossFunction end

@inline _mape_w(yi) = 1.0 / max(abs(yi), 1e-8)

function gradient_hessian!(g, h, ::MAPE, y, pred, _scratch)
    @inbounds for i in eachindex(y)
        g[i] = sign(y[i] - pred[i]) * _mape_w(y[i])
        h[i] = 1.0
    end
    return nothing
end

initial_prediction(::MAPE, y) = median(y)

loss(::MAPE, y, pred) = mean(abs.(y .- pred) .* _mape_w.(y))

uses_leaf_refinement(::MAPE) = true
function leaf_refine_targets!(vals, ws, ::MAPE, y, pred, weights)
    vals .= y .- pred
    ws .= weights .* _mape_w.(y)
    return nothing
end
leaf_refine_value(::MAPE, vals, ws) = weighted_median(vals, ws)

"""
    Tweedie(p)

Tweedie regression loss with log link and variance power `1 < p < 2`, for
non-negative, right-skewed targets with a mass at zero (insurance claims, sales
counts). The model predicts `log(mean)`; [`predict`](@ref) exponentiates it.
Pass as `loss_function=Tweedie(1.5)` to a [`MichiBoostRegressor`](@ref).
"""
struct Tweedie <: LossFunction
    p::Float64
    function Tweedie(p::Real)
        1 < p < 2 || error("Tweedie p must be in (1, 2); got $p.")
        return new(Float64(p))
    end
end

function gradient_hessian!(g, h, lf::Tweedie, y, pred, _scratch)
    p = lf.p
    @inbounds for i in eachindex(y)
        a = exp((1.0 - p) * pred[i])
        b = exp((2.0 - p) * pred[i])
        g[i] = y[i] * a - b
        h[i] = max(-y[i] * (1.0 - p) * a + (2.0 - p) * b, 1e-6)
    end
    return nothing
end

initial_prediction(::Tweedie, y) = log(max(mean(y), 1e-8))

link_inverse(::Tweedie, raw) = exp.(raw)

function loss(lf::Tweedie, y, pred)
    p = lf.p
    s = 0.0
    @inbounds for i in eachindex(y)
        s +=
            -y[i] * exp((1.0 - p) * pred[i]) / (1.0 - p) +
            exp((2.0 - p) * pred[i]) / (2.0 - p)
    end
    return s / length(y)
end

"""
    LogLinQuantile(alpha)

Quantile regression with a log link, `0 < alpha < 1`. The pinball loss is taken
on `y - exp(pred)`, so the model fits the `alpha`-quantile of positive,
wide-ranging targets and [`predict`](@ref) exponentiates the raw score. Leaf
values come from the log of a weighted quantile. Targets must be positive. Pass
as `loss_function=LogLinQuantile(0.9)` to a [`MichiBoostRegressor`](@ref).
"""
struct LogLinQuantile <: LossFunction
    alpha::Float64
    function LogLinQuantile(alpha::Real)
        0 < alpha < 1 || error("LogLinQuantile alpha must be in (0, 1); got $alpha.")
        return new(Float64(alpha))
    end
end

function gradient_hessian!(g, h, lf::LogLinQuantile, y, pred, _scratch)
    α = lf.alpha
    @inbounds for i in eachindex(y)
        ea = exp(pred[i])
        g[i] = ea * (y[i] > ea ? α : α - 1.0)
        h[i] = 1.0
    end
    return nothing
end

initial_prediction(lf::LogLinQuantile, y) = log(max(quantile(y, lf.alpha), 1e-8))

link_inverse(::LogLinQuantile, raw) = exp.(raw)

function loss(lf::LogLinQuantile, y, pred)
    α = lf.alpha
    s = 0.0
    @inbounds for i in eachindex(y)
        u = y[i] - exp(pred[i])
        s += u >= 0 ? α * u : (α - 1.0) * u
    end
    return s / length(y)
end

uses_leaf_refinement(::LogLinQuantile) = true
function leaf_refine_targets!(vals, ws, ::LogLinQuantile, y, pred, weights)
    @inbounds for i in eachindex(y)
        ea = exp(pred[i])
        vals[i] = y[i] / ea
        ws[i] = weights[i] * ea
    end
    return nothing
end
function leaf_refine_value(lf::LogLinQuantile, vals, ws)
    return log(max(weighted_quantile(vals, ws, lf.alpha), 1e-8))
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
