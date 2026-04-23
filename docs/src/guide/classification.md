# Classification

MichiBoost.jl supports both binary and multi-class classification tasks.

## Binary Classification

For binary classification, use `MichiBoostClassifier` with the Logloss loss function:

```julia
using MichiBoost

model = MichiBoostClassifier(;
    iterations=200,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss"
)

fit!(model, X_train, y_train)
```

### Prediction Types

```julia
# Get probability of positive class
probabilities = predict_proba(model, X_test)

# Get predicted class labels (default for classifiers)
classes = predict(model, X_test)

# Get raw logits (before sigmoid)
logits = predict(model, X_test; prediction_type="RawFormulaVal")
```

`predict(clf, X)` returns class labels by default. Pass
`prediction_type="Probability"` to get probabilities, or
`prediction_type="RawFormulaVal"` for the pre-transform scores.

## Multi-Class Classification

Multi-class classification is automatically detected when your target has more than 2 unique values:

```julia
using MichiBoost

# Target with 3 classes
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

## Complete Binary Classification Example

```julia
using MichiBoost, Random, Statistics

# Generate synthetic binary classification data
Random.seed!(42)
n = 1000
X = randn(n, 5)
y = Float64.(X[:, 1] .+ X[:, 2] .> 0)

# Split data
train_idx = 1:800
test_idx = 801:1000

X_train, y_train = X[train_idx, :], y[train_idx]
X_test, y_test = X[test_idx, :], y[test_idx]

# Train model
model = MichiBoostClassifier(;
    iterations=200,
    learning_rate=0.05,
    depth=4,
    verbose=true
)

fit!(model, X_train, y_train)

# Evaluate
probs = predict_proba(model, X_test)
preds = predict(model, X_test)

accuracy = mean(preds .== y_test)
println("Test Accuracy: $(round(accuracy * 100, digits=2))%")
```

## Complete Multi-Class Example

```julia
using MichiBoost, Random, Statistics

# Generate synthetic 3-class data
Random.seed!(42)
n = 900
X = randn(n, 5)
y = mod.(1:n, 3)  # Classes 0, 1, 2

# Split data
train_idx = 1:700
test_idx = 701:900

X_train, y_train = X[train_idx, :], y[train_idx]
X_test, y_test = X[test_idx, :], y[test_idx]

# Train model
model = MichiBoostClassifier(;
    iterations=200,
    learning_rate=0.05,
    depth=4,
    loss_function="MultiClass"
)

fit!(model, X_train, y_train)

# Evaluate
probs = predict_proba(model, X_test)  # Shape: (200, 3)
preds = predict(model, X_test)

accuracy = mean(preds .== y_test)
println("Test Accuracy: $(round(accuracy * 100, digits=2))%")
```

## Working with String Labels

MichiBoost.jl automatically handles string class labels:

```julia
using MichiBoost

y = ["cat", "dog", "cat", "bird", "dog", "bird"]
X = randn(6, 3)

model = MichiBoostClassifier(; iterations=50)
fit!(model, X, y)

# Predictions return original string labels
predictions = predict(model, X)
# ["cat", "dog", "cat", "bird", "dog", "bird"]
```

See the [Advanced Features](advanced.md) guide for cross-validation, SHAP
values, feature importance, sample weights, and model persistence.
