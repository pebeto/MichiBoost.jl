# Getting Started

This guide walks you through your first MichiBoost.jl model.

## Installation

Install MichiBoost.jl from the Julia package registry:

```julia
using Pkg
Pkg.add("MichiBoost")
```

Or in the Julia REPL package mode (press `]`):

```julia
add MichiBoost
```

## Basic Workflow

A typical MichiBoost.jl session has three steps:

1. **Create a model**: choose [`MichiBoostRegressor`](@ref) or
   [`MichiBoostClassifier`](@ref).
2. **Fit the model**: train on your data with [`fit!`](@ref).
3. **Make predictions**: call [`predict`](@ref) for values or class labels,
   or [`predict_proba`](@ref) for probabilities (classifiers only).

## Your First Model

### Regression Example

```julia
using MichiBoost

X = [1.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0; 30.0 40.0 50.0 60.0]
y = [10.0, 20.0, 30.0]

model = MichiBoostRegressor(; iterations=100, learning_rate=0.1, depth=4)
fit!(model, X, y)

predictions = predict(model, X)
println(predictions)
```

### Classification Example

```julia
using MichiBoost

X = [0.0 3.0; 4.0 1.0; 8.0 1.0; 9.0 1.0]
y = [0.0, 0.0, 1.0, 1.0]

model = MichiBoostClassifier(; iterations=100, learning_rate=0.1, depth=4)
fit!(model, X, y)

probabilities = predict_proba(model, X)
classes = predict(model, X)
```

## Working with DataFrames

MichiBoost.jl accepts `DataFrame`s and other Tables.jl-compatible sources:

```julia
using MichiBoost, DataFrames

df = DataFrame(
    feature1=[1.0, 2.0, 3.0, 4.0],
    feature2=[10.0, 20.0, 30.0, 40.0],
    target=[0.0, 0.0, 1.0, 1.0],
)

X = select(df, Not(:target))
y = df.target

model = MichiBoostClassifier(; iterations=50)
fit!(model, X, y)
predict(model, X)
```

## Threading

MichiBoost uses `Threads.nthreads()` during training and inference. For best
performance, start Julia with threads enabled:

```bash
julia -t 4   # any thread count works
```

## Next Steps

- [Regression](guide/regression.md) and [Classification](guide/classification.md):
  task-focused tutorials.
- [Hyperparameters](guide/hyperparameters.md): every knob the constructor takes.
- [Categorical Features](guide/categorical_features.md): how target encoding
  works and when to override it.
- [Custom Loss Functions](guide/custom_loss.md): plug in your own loss.
- [Advanced Features](guide/advanced.md): sample weights, cross-validation,
  SHAP, model persistence.
