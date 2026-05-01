# Advanced Features

```@meta
CurrentModule = MichiBoost
```

This guide covers features that are supported by both [`MichiBoostRegressor`](@ref) and
[`MichiBoostClassifier`](@ref).

## Sample Weights

Per-row importance weights are passed through [`Pool`](@ref):

```julia
using MichiBoost

X = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
y = [0.0, 1.0, 0.0, 1.0]
w = [1.0, 2.0, 0.5, 1.0]

pool = Pool(X; label=y, weight=w)

model = MichiBoostClassifier(; iterations=100)
fit!(model, pool)
```

Weights scale the per-sample gradients and hessians during training. A weight of
`0.0` effectively drops a row; a weight of `2.0` makes a row count twice as
much.

## Class Weights

For classification, per-class weights can be supplied directly on
[`MichiBoostClassifier`](@ref) without constructing a per-row vector. They are
multiplied into the per-sample weights at fit time:

```julia
clf = MichiBoostClassifier(;
    iterations=200,
    class_weights=Dict(0.0 => 1.0, 1.0 => 5.0),  # upweight the positive class
)
fit!(clf, X, y)
```

The dict's keys must cover every distinct label; numeric keys match `Float64`
labels by value (so `Dict(0 => 1.0)` works against `y = [0.0, 1.0, ...]`),
non-numeric keys match the original labels passed to [`Pool`](@ref) /
[`fit!`](@ref). If a [`Pool`](@ref) already has per-row `weight` set, class
weights are multiplied on top of it.

For automatic weighting from label frequency, pass `auto_class_weights` instead:

```julia
clf = MichiBoostClassifier(; iterations=200, auto_class_weights=AutoClassWeights.Balanced)
# or, CatBoost-style:
clf = MichiBoostClassifier(; iterations=200, auto_class_weights="Balanced")
```

- `AutoClassWeights.Balanced` — `weight[c] = n / (n_classes * count[c])`.
- `AutoClassWeights.SqrtBalanced` — `weight[c] = sqrt(n / count[c])`. Less
  aggressive than `Balanced`.

`class_weights` and `auto_class_weights` are mutually exclusive.

## Cross-Validation

[`cv`](@ref) runs k-fold cross-validation on a [`Pool`](@ref) and returns per-fold and mean
losses:

```julia
using MichiBoost

pool = Pool(X; label=y)
result = cv(
    pool;
    fold_count=5,
    params=Dict("iterations" => 100, "depth" => 4, "learning_rate" => 0.05),
    random_seed=42,
    verbose=true,
)

println("Mean train loss: ", result.mean_train_loss)
println("Mean test loss:  ", result.mean_test_loss)
```

For classification on imbalanced data, pass `stratified=true` so each fold
preserves the global class proportions:

```julia
result = cv(
    pool;
    fold_count=5,
    stratified=true,
    params=Dict("iterations" => 100, "depth" => 4,
                "loss_function" => "Logloss"),
)
```

Stratified CV requires every class to have at least `fold_count` samples.

## Early-Stopping Inspection

After training with `early_stopping_rounds` and `eval_set`, two accessors expose
the best round and its eval-metric value (matching CatBoost's
`get_best_iteration` / `get_best_score`):

```julia
clf = MichiBoostClassifier(;
    iterations=1000,
    early_stopping_rounds=20,
    eval_metric=Metrics.AUC,
)
fit!(clf, train_pool; eval_set=val_pool)

println("Best iteration: ", get_best_iteration(clf))
println("Best AUC:       ", get_best_score(clf))
```

When early stopping was not active (no `eval_set` / `early_stopping_rounds`),
`get_best_iteration` returns the total number of trees actually built and
`get_best_score` returns `nothing`.

The returned `NamedTuple` exposes `train_loss`, `test_loss`, `mean_train_loss`,
and `mean_test_loss`. Keys in `params` may be strings or symbols. The loss
function used is taken from `params[:loss_function]` (default `"RMSE"`).

## SHAP Values

SHAP values explain individual predictions by attributing the deviation from
the mean prediction to each feature:

```julia
using MichiBoost

model = MichiBoostRegressor(; iterations=100)
fit!(model, X_train, y_train)

shap = shap_values(model, X_test)
# Regression / binary classification: Matrix{Float64} of shape (n_samples, n_features)
# Multi-class:                        Array{Float64,3} of shape (n_samples, n_features, n_classes)
```

For each row `i`, `sum(shap[i, :])` is approximately equal to
`prediction[i] - mean_prediction`.

## Feature Importance

[`feature_importance`](@ref) returns a `Vector{Pair{Symbol,Float64}}` mapping feature
names to percentages (summing to 100). The result is already sorted: features
that split the most come first, followed by features that never appear in any
split (score `0.0`). The score is split-count based, not gain-based.

```julia
importance = feature_importance(model)
for (feature, score) in importance
    println(rpad(string(feature), 20), round(score; digits=2), "%")
end
```

## Model Persistence

Trained models can be serialized to disk and reloaded later. Serialization uses
Julia's `Serialization` stdlib, so loaded files are only guaranteed to work
with the same MichiBoost version that wrote them.

```julia
using MichiBoost

model = MichiBoostRegressor(; iterations=100)
fit!(model, X_train, y_train)

save_model(model, "model.jls")

loaded = load_model("model.jls")
predictions = predict(loaded, Pool(X_test))
```

[`save_model`](@ref) accepts either a wrapper ([`MichiBoostRegressor`](@ref) /
[`MichiBoostClassifier`](@ref)) or a raw [`MichiBoostModel`](@ref). [`load_model`](@ref)
always returns a [`MichiBoostModel`](@ref), which takes a [`Pool`](@ref) as prediction
input.
