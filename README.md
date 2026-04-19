# MichiBoost.jl

[![CI](https://github.com/pebeto/MichiBoost.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/pebeto/MichiBoost.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/pebeto/MichiBoost.jl/graph/badge.svg?token=3RIH95Q485)](https://codecov.io/github/pebeto/MichiBoost.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A pure Julia implementation of gradient boosting with symmetric (oblivious) decision trees and ordered target encoding for categorical features, inspired by the [CatBoost](https://catboost.ai/) algorithm.

_Michi (ミチ) means cat in Japanese._

## Features

- **Pure Julia** — no Python, no C++ bindings, no CondaPkg
- **Ordered target encoding** for native categorical feature handling without preprocessing
- **Symmetric (oblivious) trees** as the base learner — strong regularization and fast inference
- **Histogram-based split finding** with quantile-based feature binning and pre-allocated buffers
- **Fast inference** — beats CatBoost on small regression, multiclass, and categorical workloads
- Regression (RMSE, MAE), binary classification (Logloss), and multi-class (Softmax)
- Cross-validation, early stopping, RSM feature subsampling, model serialization

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

MichiBoost.jl has been validated against the reference CatBoost implementation (Python wrapper). The benchmark compares both implementations on identical datasets with the same hyperparameters (100 iterations, depth=6, learning_rate=0.03). All correctness metrics are **out-of-sample** (80/20 train/test split).

### Correctness

| Task                      | Metric                  | Result    |
| ------------------------- | ----------------------- | --------- |
| **Regression**            | Prediction correlation  | r = 0.98  |
|                           | RMSE (CatBoost)         | 2.07      |
|                           | RMSE (MichiBoost)       | 2.04      |
| **Binary Classification** | Probability correlation | r = 0.97  |
|                           | Class agreement         | 92.5%     |
|                           | Accuracy (CatBoost)     | 77.5%     |
|                           | Accuracy (MichiBoost)   | 77.5%     |
| **Multi-class**           | Class agreement         | 75.8%     |
|                           | Accuracy (CatBoost)     | 57.5%     |
|                           | Accuracy (MichiBoost)   | 66.7%     |
| **Categorical**           | Class agreement         | 63.5%     |
|                           | Accuracy (CatBoost)     | 51.5%     |
|                           | Accuracy (MichiBoost)   | 48.0%     |

### Performance

Training uses the train split; inference is measured on the held-out test split.

| Dataset                 | Task                  | CatBoost Training | MichiBoost Training | CatBoost Inference | MichiBoost Inference |
| ----------------------- | --------------------- | ----------------- | ------------------- | ------------------ | -------------------- |
| Small (200×10)          | Regression (RMSE)     | 73.8 ms           | **27.2 ms**         | 0.213 ms           | **0.010 ms**         |
| Small (200×10)          | Regression (MAE)      | 76.3 ms           | **26.9 ms**         | 0.210 ms           | **0.010 ms**         |
| Medium (2000×20)        | Regression            | **117.4 ms**      | 167.2 ms            | **0.286 ms**       | **0.165 ms**         |
| 1000×15                 | Binary Classification | **106.4 ms**      | **88.2 ms**         | 0.198 ms           | **0.040 ms**         |
| 500×10 (3 classes)      | Multiclass            | 129.9 ms          | **111.4 ms**        | 0.174 ms           | **0.036 ms**         |
| 1000×10 (5 cat + 5 num) | Categorical           | **87.4 ms**       | 113.5 ms            | **0.518 ms**       | **0.172 ms**         |

Both implementations use 4 threads. MichiBoost.jl shows strong performance on small-to-medium datasets: training is 2–3× faster on small regression and multiclass, and inference beats CatBoost across all synthetic workloads (CatBoost's Python call overhead is significant for small batches).

Run the validation yourself:

```bash
julia --project=test/benchmark_project -t 4 test/benchmark_vs_catboost.jl
```
