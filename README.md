# MichiBoost.jl

[![CI](https://github.com/pebeto/MichiBoost.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/pebeto/MichiBoost.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/pebeto/MichiBoost.jl/graph/badge.svg?token=3RIH95Q485)](https://codecov.io/github/pebeto/MichiBoost.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A pure Julia implementation of gradient boosting with symmetric (oblivious) decision trees and ordered target encoding for categorical features, inspired by the [CatBoost](https://catboost.ai/) algorithm.

_Michi (ミチ) means cat in Japanese._

## Features

- **Pure Julia** — no Python, no C++ bindings, no CondaPkg
- **Ordered target encoding** for native categorical feature handling without preprocessing
- **Symmetric (oblivious) trees** as the base learner
- **Histogram-based split finding** with quantile-based feature binning and pre-allocated buffers
- **Low-overhead inference** for single rows and small batches (see [benchmark results](benchmark/README.md))
- Regression (RMSE, MAE), binary classification (Logloss), and multi-class (Softmax)
- **SHAP values** for feature-level explanation of individual predictions
- **Sample weights** — pass per-row importance via `Pool(...; weight=...)`
- Cross-validation, early stopping, RSM feature subsampling, model serialization

## Quick Start

Run with threading enabled for best performance:

```bash
julia -t 4   # or any thread count
```

This sets `Threads.nthreads()` which MichiBoost uses during training.

### Regression

```julia
using MichiBoost

X = [1.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0; 30.0 40.0 50.0 60.0]
y = [10.0, 20.0, 30.0]

model = MichiBoostRegressor(; iterations=100, learning_rate=0.1, depth=4)
fit!(model, X, y)

preds = predict(model, X)
```

### Binary Classification

```julia
using MichiBoost

X = [0.0 3.0; 4.0 1.0; 8.0 1.0; 9.0 1.0]
y = [0.0, 0.0, 1.0, 1.0]

model = MichiBoostClassifier(; iterations=100, learning_rate=0.1, depth=4)
fit!(model, X, y)

probs = predict_proba(model, X)   # P(class=1)
classes = predict(model, X)        # predicted class labels
```

### Categorical Features

String columns are automatically detected as categorical:

```julia
using MichiBoost, DataFrames

df = DataFrame(color=["red", "blue", "red", "green"], size=[1.0, 2.0, 3.0, 4.0])
y = [0.0, 1.0, 0.0, 1.0]

model = MichiBoostClassifier(; iterations=50)
fit!(model, df, y)
predict(model, df)
```

### Cross-Validation

```julia
pool = Pool(X; label=y)
scores = cv(pool; fold_count=5, params=Dict("iterations" => 100, "depth" => 4))
println("Mean test loss: ", scores.mean_test_loss)
```

### SHAP Values

Explain individual predictions with SHAP values:

```julia
shap = shap_values(model, X)   # shape: (n_samples, n_features)
```

### Sample Weights

Weight individual training samples via `Pool`:

```julia
w = [1.0, 2.0, 0.5, 1.0]
pool = Pool(X; label=y, weight=w)
fit!(model, pool)
```

## Validation Against CatBoost

MichiBoost.jl is benchmarked against the reference CatBoost implementation (via [CatBoost.jl](https://github.com/beacon-biosignals/CatBoost.jl)) across four axes: correctness on held-out data, a training/inference speed sweep, threading and real-dataset scaling (UCI Covertype), and the cost of each advertised feature (CV, early stopping, SHAP, RSM, sample weights, save/load).

See [`benchmark/README.md`](benchmark/README.md) for the full methodology, per-script commands, latest results, and caveats.
