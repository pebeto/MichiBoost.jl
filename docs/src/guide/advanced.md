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
```

- `AutoClassWeights.Balanced`: `weight[c] = n / (n_classes * count[c])`.
- `AutoClassWeights.SqrtBalanced`: `weight[c] = sqrt(n / count[c])`. Less
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

The returned `NamedTuple` also exposes per-fold `train_loss` and `test_loss`
vectors alongside the means shown above. `params` keys may be strings or
symbols; the loss used to score each fold is taken from
`params[:loss_function]` (default `"RMSE"`).

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

## SHAP Values

SHAP values explain individual predictions by attributing the deviation from
the mean prediction to each feature:

```julia
using MichiBoost

model = MichiBoostRegressor(; iterations=100)
fit!(model, X_train, y_train)

shap = shap_values(model, X_test)
# Regression and binary classification: Matrix{Float64} of shape (n_samples, n_features)
# Multi-class: Array{Float64,3} of shape (n_samples, n_features, n_classes)
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

## Monotonicity Constraints

Force the prediction to move only one way in a feature. Pass
`monotone_constraints` as a `Dict` keyed by feature name or 1-based column
index, with `+1` for non-decreasing, `-1` for non-increasing, and `0` for
unconstrained.

```julia
model = MichiBoostRegressor(;
    iterations=500,
    monotone_constraints=Dict(:price => +1, :age => -1),
)
fit!(model, X_train, y_train)
```

A length-`n_features` vector of signs in column order works too:

```julia
# +1 on column 1, -1 on column 2, the rest free.
MichiBoostRegressor(; monotone_constraints=[1, -1, 0, 0])
```

The constraint is a hard guarantee: for any two inputs that differ only in a
`+1` feature, the one with the larger feature value gets a prediction that is
greater than or equal to the other's. MichiBoost enforces it by bounding leaf
values during tree construction, so it holds for every tree and therefore the
whole ensemble. Guaranteeing monotonicity separates the affected subtrees, which
costs a little accuracy compared with an unconstrained fit.

Constraints apply to numerical features in regression and binary classification.
Passing them with a categorical feature, a multi-class or multi-target loss, or
a refinement loss (MAE, the quantile family) raises an error.

## Model Persistence

Trained models can be serialized to disk and reloaded later. Serialization uses
[JLD2.jl](https://github.com/JuliaIO/JLD2.jl) (HDF5-flavoured), which is stable
across Julia versions; struct-layout changes in MichiBoost itself may still
require migration.

```julia
using MichiBoost

model = MichiBoostRegressor(; iterations=100)
fit!(model, X_train, y_train)

save_model(model, "model.jld2")

loaded = load_model("model.jld2")
predictions = predict(loaded, Pool(X_test))
```

[`save_model`](@ref) accepts either a wrapper ([`MichiBoostRegressor`](@ref) /
[`MichiBoostClassifier`](@ref)) or a raw [`MichiBoostModel`](@ref). [`load_model`](@ref)
always returns a [`MichiBoostModel`](@ref), which takes a [`Pool`](@ref) as prediction
input.

## Continuing Training

Pass a previously fitted model as `init_model` to resume training from it. The
new fit inherits every tree of `init_model` and adds `iterations` more on top.

```julia
base = MichiBoostRegressor(; iterations=100, depth=6, learning_rate=0.03)
fit!(base, X_train, y_train)

continued = MichiBoostRegressor(; iterations=50, depth=6, learning_rate=0.03)
fit!(continued, X_train, y_train; init_model=base)

length(continued.model.trees) == 150
```

MichiBoost reuses `init_model`'s feature borders and (if present) categorical
encoder. The `border_count` kwarg is ignored when `init_model` is set, since
changing borders would invalidate every split in the inherited trees.

Accepted forms for `init_model`:

- a fitted [`MichiBoostRegressor`](@ref) or [`MichiBoostClassifier`](@ref) wrapper
- a raw [`MichiBoostModel`](@ref)
- `nothing` (the default; equivalent to a fresh fit)

The task must match. A regressor only continues from a regressor, a classifier
only from a classifier with the same `n_classes` and `is_multiclass`. Feature
counts must also match. [`fit!`](@ref) errors at the start of training when
these don't line up.

`learning_rate` applies to every tree in the final model, inherited and new
alike. Set it to the value `base` used. Picking a different `learning_rate`
rescales the inherited trees' contribution, so `predict(continued, X)` will not
match `predict(base, X)` even before any new trees fire.

`early_stopping_rounds` works with `init_model`: the eval metric runs over the
full ensemble after each new tree, and `best_iteration` counts from tree #1 of
the inherited model, not from the first newly added tree.

## Snapshots

Set `snapshot_path` to write the partial model to disk during training.
MichiBoost serialises the current model every `snapshot_interval` iterations
and once more after the training loop exits, using JLD2. Each write overwrites
the previous file, so only the latest snapshot survives.

```julia
model = MichiBoostRegressor(;
    iterations=10_000,
    snapshot_path="checkpoint.jld2",
    snapshot_interval=500,
)
fit!(model, X_train, y_train)
```

If the process crashes or you cancel it, load the file and resume with
`init_model`:

```julia
snap = load_model("checkpoint.jld2")
resumed = MichiBoostRegressor(;
    iterations=10_000 - length(snap.trees),
    learning_rate=0.03,  # match the original fit
)
fit!(resumed, X_train, y_train; init_model=snap)
```

The final on-disk snapshot matches the model [`fit!`](@ref) returns. That
includes early-stopping truncation: when `early_stopping_rounds` triggers a
break, the post-loop save writes the truncated `trees[1:best_iteration]`, not
the trees built between the last interval and the break.

`snapshot_interval` must be `>= 1`. [`fit!`](@ref) errors before training
starts otherwise.
