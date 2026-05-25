# Hyperparameters

This guide covers all hyperparameters available in MichiBoost.jl.

## Core Parameters

### `iterations`

**Type:** `Int`
**Default:** `1000`

Number of boosting rounds (trees to build).

```julia
model = MichiBoostRegressor(; iterations=200)
```

### `learning_rate`

**Type:** `Float64`
**Default:** `0.03`

Step-size shrinkage applied to each tree's contribution.

```julia
model = MichiBoostRegressor(; learning_rate=0.05)
```

### `depth`

**Type:** `Int`
**Default:** `6`

Depth of each symmetric (oblivious) tree. A tree of depth `d` has `2^d` leaves.

```julia
model = MichiBoostRegressor(; depth=4)
```

### `l2_leaf_reg`

**Type:** `Float64`
**Default:** `3.0`

L2 regularization coefficient for leaf values.

```julia
model = MichiBoostRegressor(; l2_leaf_reg=5.0)
```

## Loss Functions

Pass a `Losses.*` tag type. Tag types catch typos at parse time (e.g.,
`Losses.RMES` is an `UndefVarError`). CatBoost-style strings (`"RMSE"`) and
Symbols (`:RMSE`) resolve to the same loss.

### Regression

```julia
# Root Mean Squared Error (default)
model = MichiBoostRegressor(; loss_function=Losses.RMSE)

# Mean Absolute Error
model = MichiBoostRegressor(; loss_function=Losses.MAE)
```

### Classification

```julia
# Binary classification (default)
model = MichiBoostClassifier(; loss_function=Losses.Logloss)

# Multi-class (auto-detected if target has >2 classes)
model = MichiBoostClassifier(; loss_function=Losses.MultiClass)
```

`Losses.CrossEntropy` is an alias for `Losses.Logloss`, and `Losses.MultiLogloss`
for `Losses.MultiClass`. The same aliases resolve through the string and Symbol
forms.

## Feature Processing

### `border_count`

**Type:** `Int`
**Default:** `254`

Maximum number of quantization borders per numerical feature.

```julia
model = MichiBoostRegressor(; border_count=128)
```

### `min_data_in_leaf`

**Type:** `Int`
**Default:** `1`

Minimum number of samples required in a leaf node.

```julia
model = MichiBoostRegressor(; min_data_in_leaf=5)
```

## Categorical Features

### `boosting_type`

**Type:** `Union{Type{<:BoostingTypes.BoostingType}, String, Symbol}`
**Default:** `BoostingTypes.Ordered`

Controls how target statistics are computed for categorical features.

- `BoostingTypes.Ordered`: uses a random permutation so each sample's
  encoding is derived from preceding samples only, reducing target leakage.
- `BoostingTypes.Plain`: encodes each category using statistics from the full
  training set.

Gradient computation uses standard (plain) gradient boosting in both modes.

```julia
model = MichiBoostRegressor(; boosting_type=BoostingTypes.Ordered)
model = MichiBoostRegressor(; boosting_type=BoostingTypes.Plain)
```

## Early Stopping

### `early_stopping_rounds`

**Type:** `Union{Int, Nothing}`
**Default:** `nothing`

Stop training if validation loss doesn't improve for this many rounds.
Requires providing `eval_set` to [`fit!`](@ref).

```julia
train_pool = Pool(X_train; label=y_train)
val_pool = Pool(X_val; label=y_val)

model = MichiBoostRegressor(; iterations=1000, early_stopping_rounds=50)
fit!(model, train_pool; eval_set=val_pool)
```

### `eval_metric`

**Type:** `Union{Type{<:Metric}, String, Symbol, Nothing}`
**Default:** `nothing`

Metric used to drive the early-stopping comparison. `nothing` falls back to the
training loss. Pass a tag from the `Metrics` submodule for compile-time-checked
names (typos become `UndefVarError` at parse time); the matching string or
Symbol resolves to the same metric. Comparison direction is inferred per metric.

| Task        | Available metrics                                                  |
| ----------- | ------------------------------------------------------------------ |
| Regression  | `Metrics.RMSE`, `Metrics.MAE`, `Metrics.R2`                        |
| Binary      | `Metrics.Logloss`, `Metrics.Accuracy`, `Metrics.F1`, `Metrics.AUC` |
| Multi-class | `Metrics.MultiLogloss`, `Metrics.Accuracy`                         |

```julia
using MichiBoost

clf = MichiBoostClassifier(;
    iterations=1000,
    early_stopping_rounds=20,
    eval_metric=Metrics.AUC,        # bare type, no `()`
)
fit!(clf, train_pool; eval_set=val_pool)
```

Regression metrics are computed via
[StatisticalMeasures.jl](https://github.com/JuliaAI/StatisticalMeasures.jl);
classification metrics are inlined to avoid the per-iteration cost of building
`UnivariateFinite` distributions.

## Other Parameters

### `random_seed`

**Type:** `Union{Int, Nothing}`
**Default:** `nothing`

Random seed for reproducibility.

```julia
model = MichiBoostRegressor(; random_seed=42)
```

### `verbose`

**Type:** `Bool`
**Default:** `false`

Print training progress.

```julia
model = MichiBoostRegressor(; verbose=true)
```

### `rsm`

**Type:** `Float64`
**Default:** `1.0`

Random subspace method: the fraction of features sampled per split search.
`1.0` uses all features; `0.5` samples half. The subset is re-sampled
independently at each tree level.

```julia
model = MichiBoostRegressor(; rsm=0.8)
```

## Sample Weights

Per-row importance is passed through [`Pool`](@ref), not through the model constructor.
See the [Advanced Features](advanced.md) guide.
