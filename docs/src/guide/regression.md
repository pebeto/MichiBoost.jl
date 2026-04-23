# Regression

MichiBoost.jl supports regression tasks with multiple loss functions.

## Creating a Regression Model

```julia
using MichiBoost

model = MichiBoostRegressor(;
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE"
)
```

## Supported Loss Functions

### RMSE (Root Mean Squared Error)

The default loss function for regression. Minimizes the squared differences between predictions and targets.

```julia
model = MichiBoostRegressor(; loss_function="RMSE")
```

### MAE (Mean Absolute Error)

Minimizes the absolute differences between predictions and targets.

```julia
model = MichiBoostRegressor(; loss_function="MAE")
```

## Complete Example

```julia
using MichiBoost, Random, Statistics

# Generate synthetic data
Random.seed!(42)
n = 1000
X = randn(n, 5)
y = 2.0 .* X[:, 1] .- 1.5 .* X[:, 2] .+ 0.5 .* X[:, 3] .+ randn(n) .* 0.1

# Split into train/test
train_idx = 1:800
test_idx = 801:1000

X_train, y_train = X[train_idx, :], y[train_idx]
X_test, y_test = X[test_idx, :], y[test_idx]

# Train model
model = MichiBoostRegressor(;
    iterations=200,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=3.0,
    verbose=true
)

fit!(model, X_train, y_train)

# Evaluate
predictions = predict(model, X_test)
rmse = sqrt(mean((predictions .- y_test).^2))
println("Test RMSE: $rmse")
```

## Early Stopping

Use held-out validation data to stop training when the evaluation loss stops
improving:

```julia
using MichiBoost, Random

Random.seed!(42)
n = 1000
X = randn(n, 5)
y = 2.0 .* X[:, 1] .- 1.5 .* X[:, 2] .+ randn(n) .* 0.1

X_train, y_train = X[1:800, :], y[1:800]
X_val,   y_val   = X[801:end, :], y[801:end]

train_pool = Pool(X_train; label=y_train)
val_pool   = Pool(X_val;   label=y_val)

model = MichiBoostRegressor(; iterations=1000, early_stopping_rounds=50)
fit!(model, train_pool; eval_set=val_pool)
```

See the [Advanced Features](advanced.md) guide for feature importance, SHAP
values, sample weights, cross-validation, and model persistence.
