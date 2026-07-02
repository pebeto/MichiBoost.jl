# Regression

[`MichiBoostRegressor`](@ref) is the entry point for regression tasks.
RMSE is the default loss; pass `loss_function=Losses.MAE` for MAE. The full
list of supported losses and every other hyperparameter lives in the
[Hyperparameters](hyperparameters.md) guide.

## Minimal Example

```julia
using MichiBoost

model = MichiBoostRegressor(;
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function=Losses.RMSE,
)
```

## Complete Example

```julia
using MichiBoost, Random, Statistics

Random.seed!(42)
n = 1000
X = randn(n, 5)
y = 2.0 .* X[:, 1] .- 1.5 .* X[:, 2] .+ 0.5 .* X[:, 3] .+ randn(n) .* 0.1

train_idx = 1:800
test_idx = 801:1000

X_train, y_train = X[train_idx, :], y[train_idx]
X_test, y_test = X[test_idx, :], y[test_idx]

model = MichiBoostRegressor(;
    iterations=200,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=3.0,
    verbose=true,
)

fit!(model, X_train, y_train)

predictions = predict(model, X_test)
rmse = sqrt(mean((predictions .- y_test) .^ 2))
println("Test RMSE: $rmse")
```

## Choosing a Loss

RMSE fits the conditional mean and suits clean, symmetric targets. The
parametrized losses below cover the cases where RMSE falls short. Pass an
instance to `loss_function`.

| Loss | Use it when |
|---|---|
| `Huber(δ)` | Outliers distort an RMSE fit. Squared error within `δ`, linear past it. |
| `Quantile(α)` | You need a prediction interval or an asymmetric over/under penalty. Fits the `α`-quantile. |
| `Expectile(α)` | Same asymmetry as `Quantile` but with squared error, so the fit stays smooth. |
| `MAPE()` | Relative error matters more than absolute. Targets must be non-zero. |
| `Tweedie(p)` | Non-negative, right-skewed targets with a spike at zero (claims, demand). Predicts on a log scale. |
| `LogLinQuantile(α)` | Quantiles of positive targets that span orders of magnitude. |
| `RMSEWithUncertainty()` | You want a per-prediction uncertainty estimate, not just a point. Returns mean and standard deviation. |

```julia
# 90th-percentile fit for an upper prediction bound.
upper = MichiBoostRegressor(; iterations=500, loss_function=Quantile(0.9))
fit!(upper, X_train, y_train)

# Outlier-robust fit.
robust = MichiBoostRegressor(; iterations=500, loss_function=Huber(1.0))
fit!(robust, X_train, y_train)
```

`Tweedie` and `LogLinQuantile` predict on a log scale; [`predict`](@ref)
exponentiates the result, so the values it returns are on the original target
scale. Each loss is documented in the [Models](@ref) API reference. To write
your own, see [Custom Loss Functions](@ref).

### Predicting with Uncertainty

`RMSEWithUncertainty` fits a mean and a standard deviation per sample by
Gaussian likelihood. [`predict`](@ref) returns an `(n_samples, 2)` matrix whose
columns are the mean and the standard deviation.

```julia
model = MichiBoostRegressor(; iterations=500, loss_function=RMSEWithUncertainty())
fit!(model, X_train, y_train)

out = predict(model, X_test)
means = out[:, 1]
stds = out[:, 2]
```

`staged_predict`, `score`, and a custom `eval_metric` do not apply to this loss;
early stopping falls back to the loss itself.

## Early Stopping

Pass an `eval_set` and an `early_stopping_rounds` count to halt training once
the evaluation loss stops improving. See
[Early Stopping](hyperparameters.md#early-stopping) for the available knobs.

```julia
using MichiBoost, Random

Random.seed!(42)
n = 1000
X = randn(n, 5)
y = 2.0 .* X[:, 1] .- 1.5 .* X[:, 2] .+ randn(n) .* 0.1

X_train, y_train = X[1:800, :], y[1:800]
X_val, y_val = X[801:end, :], y[801:end]

train_pool = Pool(X_train; label=y_train)
val_pool = Pool(X_val; label=y_val)

model = MichiBoostRegressor(; iterations=1000, early_stopping_rounds=50)
fit!(model, train_pool; eval_set=val_pool)
```

See the [Advanced Features](advanced.md) guide for feature importance, SHAP
values, sample weights, cross-validation, and model persistence.
