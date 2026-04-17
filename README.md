# MichiBoost.jl

[![CI](https://github.com/pebeto/MichiBoost.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/pebeto/MichiBoost.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/pebeto/MichiBoost.jl/graph/badge.svg?token=3RIH95Q485)](https://codecov.io/github/pebeto/MichiBoost.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A pure Julia implementation of gradient boosting with ordered target statistics and symmetric (oblivious) decision trees, inspired by the [CatBoost](https://catboost.ai/) algorithm.

_Michi (ミチ) means cat in Japanese._

## Features

- **Pure Julia** — no Python, no C++ bindings, no CondaPkg
- **Ordered target statistics** for native categorical feature handling without preprocessing
- **Symmetric (oblivious) trees** as the base learner — strong regularization and fast inference
- **Histogram-based split finding** with quantile-based feature binning
- Regression (RMSE, MAE), binary classification (Logloss), and multi-class (Softmax)
- Cross-validation, early stopping, model serialization

## Quick Start

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

## Validation Against CatBoost

MichiBoost.jl has been validated against the reference CatBoost implementation (Python wrapper). The benchmark compares both implementations on identical datasets with the same hyperparameters (100 iterations, depth=6, learning_rate=0.03):

### Correctness

| Task                      | Metric                  | Result   |
| ------------------------- | ----------------------- | -------- |
| **Regression**            | Prediction correlation  | r = 0.99 |
|                           | RMSE (CatBoost)         | 1.55     |
|                           | RMSE (MichiBoost)       | 1.17     |
| **Binary Classification** | Probability correlation | r = 0.97 |
|                           | Class agreement         | 96.5%    |
|                           | Accuracy (CatBoost)     | 94.5%    |
|                           | Accuracy (MichiBoost)   | 97.0%    |
| **Multi-class**           | Class agreement         | 87.0%    |
|                           | Accuracy (CatBoost)     | 86.7%    |
|                           | Accuracy (MichiBoost)   | 97.3%    |

### Performance

| Dataset          | Task                  | CatBoost Training | MichiBoost Training |
| ---------------- | --------------------- | ----------------- | ------------------- |
| Small (200×10)   | Regression            | 94.3 ms           | **64.2 ms**         |
| Medium (2000×20) | Regression            | **125.5 ms**      | 320.2 ms            |
| 1000×15          | Binary Classification | **107.6 ms**      | 178.5 ms            |

MichiBoost.jl shows excellent agreement with CatBoost, with high correlation and comparable or better accuracy. Performance is competitive on small datasets, while CatBoost's optimized C++ implementation is faster on larger datasets.

Run the validation yourself:

```bash
julia --project=test/benchmark_project test/benchmark_vs_catboost.jl
```
