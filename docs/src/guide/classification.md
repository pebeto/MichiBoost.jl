# Classification

[`MichiBoostClassifier`](@ref) handles both binary (Logloss) and multi-class
(Softmax) tasks. The loss is set automatically from the number of unique
target values, so you rarely need to pass `loss_function` yourself. The full
list of hyperparameters lives in the [Hyperparameters](hyperparameters.md)
guide.

## Binary Classification

```julia
using MichiBoost

model = MichiBoostClassifier(;
    iterations=200,
    learning_rate=0.03,
    depth=6,
)

fit!(model, X_train, y_train)
```

### Prediction Types

```julia
# Probability of the positive class
probabilities = predict_proba(model, X_test)

# Predicted class labels (default for classifiers)
classes = predict(model, X_test)

# Raw logits (before sigmoid)
logits = predict(model, X_test; prediction_type=PredictionTypes.RawFormulaVal)
```

See [`predict`](@ref) for the full list of `prediction_type` values and their
string / Symbol aliases.

## Multi-Class Classification

A target with more than two unique values trips the multi-class path
automatically:

```julia
using MichiBoost

y = [0, 1, 2, 0, 1, 2, 0, 1, 2]

model = MichiBoostClassifier(; iterations=200)
fit!(model, X, y)
```

### Multi-Class Predictions

```julia
# Get probability matrix (n_samples × n_classes)
probabilities = predict_proba(model, X_test)

# Get predicted class labels
classes = predict(model, X_test)

# probabilities[i, :] sums to 1.0 for each sample i
```

### One-vs-All

The default multi-class loss is softmax. Pass `MultiClassOneVsAll()` to fit one
independent binary problem per class instead, which often does better on
heavily unbalanced classes. `predict_proba` still returns rows that sum to 1.

```julia
model = MichiBoostClassifier(; iterations=200, loss_function=MultiClassOneVsAll())
fit!(model, X, y)
```

## Complete Binary Classification Example

```julia
using MichiBoost, Random, Statistics

Random.seed!(42)
n = 1000
X = randn(n, 5)
y = Float64.(X[:, 1] .+ X[:, 2] .> 0)

train_idx = 1:800
test_idx = 801:1000

X_train, y_train = X[train_idx, :], y[train_idx]
X_test, y_test = X[test_idx, :], y[test_idx]

model = MichiBoostClassifier(;
    iterations=200,
    learning_rate=0.05,
    depth=4,
    verbose=true,
)

fit!(model, X_train, y_train)

probs = predict_proba(model, X_test)
preds = predict(model, X_test)

accuracy = mean(preds .== y_test)
println("Test Accuracy: $(round(accuracy * 100, digits=2))%")
```

## Complete Multi-Class Example

```julia
using MichiBoost, Random, Statistics

Random.seed!(42)
n = 900
X = randn(n, 5)
y = mod.(1:n, 3)  # Classes 0, 1, 2

train_idx = 1:700
test_idx = 701:900

X_train, y_train = X[train_idx, :], y[train_idx]
X_test, y_test = X[test_idx, :], y[test_idx]

model = MichiBoostClassifier(;
    iterations=200,
    learning_rate=0.05,
    depth=4,
)

fit!(model, X_train, y_train)

probs = predict_proba(model, X_test)  # shape: (200, 3)
preds = predict(model, X_test)

accuracy = mean(preds .== y_test)
println("Test Accuracy: $(round(accuracy * 100, digits=2))%")
```

## Working with String Labels

String class labels round-trip without conversion:

```julia
using MichiBoost

y = ["cat", "dog", "cat", "bird", "dog", "bird"]
X = randn(6, 3)

model = MichiBoostClassifier(; iterations=50)
fit!(model, X, y)

# Predictions return the original string labels
predictions = predict(model, X)
# ["cat", "dog", "cat", "bird", "dog", "bird"]
```

See the [Advanced Features](advanced.md) guide for cross-validation, SHAP
values, feature importance, sample weights, and model persistence.
