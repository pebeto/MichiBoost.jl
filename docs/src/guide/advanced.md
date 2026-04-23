# Advanced Features

This guide covers features that are supported by both `MichiBoostRegressor` and
`MichiBoostClassifier`.

## Sample Weights

Per-row importance weights are passed through `Pool`:

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

## Cross-Validation

`cv` runs k-fold cross-validation on a `Pool` and returns per-fold and mean
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

`feature_importance` returns a `Vector{Pair{Symbol,Float64}}` mapping feature
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

`save_model` accepts either a wrapper (`MichiBoostRegressor` /
`MichiBoostClassifier`) or a raw `MichiBoostModel`. `load_model` always returns
a `MichiBoostModel`, which takes a `Pool` as prediction input.
