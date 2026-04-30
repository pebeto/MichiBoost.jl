"""
    MichiBoost.Metrics

Singleton tag types for the `eval_metric` keyword argument.

Pass the bare type — for example `Metrics.AUC` — rather than instantiating it.
Each method that consumes a metric dispatches on `::Type{<:Metric}` so typos
become `UndefVarError` at parse time instead of runtime errors.

```julia
clf = MichiBoostClassifier(;
    iterations=1000, early_stopping_rounds=20, eval_metric=Metrics.AUC,
)
```

CatBoost-style string names (e.g., `"AUC"`) still resolve to the corresponding
type via `_resolve_metric`.
"""
module Metrics

abstract type Metric end

struct RMSE <: Metric end
struct MAE <: Metric end
struct R2 <: Metric end
struct Logloss <: Metric end
struct MultiLogloss <: Metric end
struct Accuracy <: Metric end
struct F1 <: Metric end
struct AUC <: Metric end

end  # module Metrics

using .Metrics: Metric

# Resolve a CatBoost-style string into a `Metric` singleton type. Used by the
# wrapper API so users coming from CatBoost can keep writing
# `eval_metric="AUC"`; idiomatic Julia callers pass `Metrics.AUC` directly.
function _resolve_metric(name::AbstractString)
    upper = uppercase(String(name))
    upper == "RMSE" && return Metrics.RMSE
    upper == "MAE" && return Metrics.MAE
    upper in ("R2", "RSQ", "RSQUARED") && return Metrics.R2
    upper in ("LOGLOSS", "CROSSENTROPY") && return Metrics.Logloss
    upper in ("MULTICLASS", "MULTILOGLOSS") && return Metrics.MultiLogloss
    upper == "ACCURACY" && return Metrics.Accuracy
    upper in ("F1", "F1SCORE") && return Metrics.F1
    upper == "AUC" && return Metrics.AUC
    return error("Unknown eval_metric name: `$name`")
end

# Build the (orientation, evaluator) pair for an early-stopping eval metric.
# Dispatched on the singleton `Metric` type. Each method validates that the
# metric is compatible with the inferred task (regression / binary / multiclass)
# and returns:
#   orientation ∈ (:minimize, :maximize)
#   evaluator(y_eval, raw_pred) -> Float64
#
# `y_eval` shape matches `loss(...)`: a Vector for binary/regression, a one-hot
# Matrix for multiclass. `raw_pred` is the running raw prediction (logits for
# classification, values for regression).
function _eval_metric(name::AbstractString, is_multiclass::Bool, n_classes::Int)
    return _eval_metric(_resolve_metric(name), is_multiclass, n_classes)
end

function _eval_metric(::Type{Metrics.RMSE}, is_multiclass::Bool, n_classes::Int)
    (is_multiclass || n_classes == 2) &&
        error("`Metrics.RMSE` is only valid for regression")
    return :minimize, (y, ŷ) -> Float64(StatisticalMeasures.rmse(ŷ, y))
end

function _eval_metric(::Type{Metrics.MAE}, is_multiclass::Bool, n_classes::Int)
    (is_multiclass || n_classes == 2) && error("`Metrics.MAE` is only valid for regression")
    return :minimize, (y, ŷ) -> Float64(StatisticalMeasures.mae(ŷ, y))
end

function _eval_metric(::Type{Metrics.R2}, is_multiclass::Bool, n_classes::Int)
    (is_multiclass || n_classes == 2) && error("`Metrics.R2` is only valid for regression")
    return :maximize, (y, ŷ) -> Float64(StatisticalMeasures.rsq(ŷ, y))
end

function _eval_metric(::Type{Metrics.Logloss}, is_multiclass::Bool, n_classes::Int)
    (is_multiclass || n_classes != 2) &&
        error("`Metrics.Logloss` is only valid for binary classification")
    return :minimize, (y, logits) -> loss(LoglossLoss(), y, logits)
end

function _eval_metric(::Type{Metrics.MultiLogloss}, is_multiclass::Bool, n_classes::Int)
    is_multiclass ||
        error("`Metrics.MultiLogloss` is only valid for multi-class classification")
    return :minimize,
    (y_onehot, logits) -> loss(MultiClassLoss(n_classes), y_onehot, logits)
end

function _eval_metric(::Type{Metrics.Accuracy}, is_multiclass::Bool, n_classes::Int)
    if is_multiclass
        return :maximize, (y_onehot, logits) -> _mc_accuracy(y_onehot, logits)
    elseif n_classes == 2
        return :maximize, (y, logits) -> _binary_accuracy(y, logits)
    end
    return error("`Metrics.Accuracy` is only valid for classification")
end

function _eval_metric(::Type{Metrics.F1}, is_multiclass::Bool, n_classes::Int)
    (is_multiclass || n_classes != 2) &&
        error("`Metrics.F1` is only valid for binary classification")
    return :maximize, (y, logits) -> _binary_f1(y, logits)
end

function _eval_metric(::Type{Metrics.AUC}, is_multiclass::Bool, n_classes::Int)
    (is_multiclass || n_classes != 2) &&
        error("`Metrics.AUC` is only valid for binary classification")
    return :maximize, (y, logits) -> _binary_auc(y, logits)
end

@inline function _binary_accuracy(y::AbstractVector, logits::AbstractVector)
    n = length(y)
    correct = 0
    @inbounds for i in 1:n
        ŷi = logits[i] >= 0.0 ? 1.0 : 0.0
        correct += ŷi == y[i]
    end
    return correct / n
end

function _binary_f1(y::AbstractVector, logits::AbstractVector)
    tp, fp, fn = 0, 0, 0
    @inbounds for i in eachindex(y)
        ŷ = logits[i] >= 0.0
        if y[i] == 1.0
            ŷ ? (tp += 1) : (fn += 1)
        else
            ŷ && (fp += 1)
        end
    end
    tp == 0 && return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)
end

function _binary_auc(y::AbstractVector, scores::AbstractVector)
    n = length(y)
    n_pos = 0
    @inbounds for v in y
        v == 1.0 && (n_pos += 1)
    end
    n_neg = n - n_pos
    (n_pos == 0 || n_neg == 0) && return 0.5
    # Mann-Whitney U: average rank of positives gives AUC directly.
    order = sortperm(scores)
    sum_ranks_pos = 0.0
    @inbounds for (rank, i) in enumerate(order)
        if y[i] == 1.0
            sum_ranks_pos += rank
        end
    end
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
end

@inline function _mc_accuracy(y_onehot::AbstractMatrix, logits::AbstractMatrix)
    n, k = size(logits)
    correct = 0
    @inbounds for i in 1:n
        pred = 1
        best = logits[i, 1]
        for c in 2:k
            if logits[i, c] > best
                best = logits[i, c]
                pred = c
            end
        end
        true_class = 1
        for c in 1:k
            if y_onehot[i, c] > 0.5
                true_class = c
                break
            end
        end
        correct += pred == true_class
    end
    return correct / n
end
