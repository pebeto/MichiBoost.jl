# Getting Started

This guide will help you get up and running with MichiBoost.jl quickly.

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

## Threading

MichiBoost uses `Threads.nthreads()` during training and inference. Start Julia
with threads enabled for best performance:

```bash
julia -t 4   # or any thread count
```

## Basic Workflow

The typical workflow with MichiBoost.jl involves:

1. **Create a model** — Choose `MichiBoostRegressor` or `MichiBoostClassifier`
2. **Fit the model** — Train on your data using `fit!`
3. **Make predictions** — Use `predict` (returns values or class labels) or
   `predict_proba` (probabilities, classifiers only)

## Your First Model

### Regression Example

```julia
using MichiBoost

# Prepare data
X = [1.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0; 30.0 40.0 50.0 60.0]
y = [10.0, 20.0, 30.0]

# Create and train model
model = MichiBoostRegressor(; iterations=100, learning_rate=0.1, depth=4)
fit!(model, X, y)

# Make predictions
predictions = predict(model, X)
println(predictions)
```

### Classification Example

```julia
using MichiBoost

# Prepare data
X = [0.0 3.0; 4.0 1.0; 8.0 1.0; 9.0 1.0]
y = [0.0, 0.0, 1.0, 1.0]

# Create and train model
model = MichiBoostClassifier(; iterations=100, learning_rate=0.1, depth=4)
fit!(model, X, y)

# Get probabilities
probabilities = predict_proba(model, X)

# Get class predictions
classes = predict(model, X)
```

## Working with DataFrames

MichiBoost.jl works seamlessly with DataFrames and other Tables.jl-compatible data structures:

```julia
using MichiBoost, DataFrames

df = DataFrame(
    feature1=[1.0, 2.0, 3.0, 4.0],
    feature2=[10.0, 20.0, 30.0, 40.0],
    target=[0.0, 0.0, 1.0, 1.0]
)

X = select(df, Not(:target))
y = df.target

model = MichiBoostClassifier(; iterations=50)
fit!(model, X, y)
predict(model, X)
```
