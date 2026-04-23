# MichiBoost.jl

A pure Julia implementation of gradient boosting with ordered target statistics and symmetric (oblivious) decision trees, inspired by the [CatBoost](https://catboost.ai/) algorithm.

_Michi (ミチ) means cat in Japanese._

## Features

- **Pure Julia** — no Python, no C++ bindings, no CondaPkg
- **Ordered target encoding** for native categorical feature handling without preprocessing
- **Symmetric (oblivious) trees** as the base learner
- **Histogram-based split finding** with quantile-based feature binning and pre-allocated buffers
- **Low-overhead inference** for single rows and small batches
- Regression (RMSE, MAE), binary classification (Logloss), and multi-class (Softmax)
- **SHAP values** for feature-level explanation of individual predictions
- **Sample weights** — pass per-row importance via `Pool(...; weight=...)`
- Cross-validation, early stopping, RSM feature subsampling, model serialization

## Installation

```julia
using Pkg
Pkg.add("MichiBoost")
```

Or from the Julia REPL:

```julia
] add MichiBoost
```

## Quick Example

```julia
using MichiBoost

# Regression
X = [1.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0; 30.0 40.0 50.0 60.0]
y = [10.0, 20.0, 30.0]

model = MichiBoostRegressor(; iterations=100, learning_rate=0.1, depth=4)
fit!(model, X, y)
preds = predict(model, X)

# Classification
X_cls = [0.0 3.0; 4.0 1.0; 8.0 1.0; 9.0 1.0]
y_cls = [0.0, 0.0, 1.0, 1.0]

clf = MichiBoostClassifier(; iterations=100, learning_rate=0.1, depth=4)
fit!(clf, X_cls, y_cls)
probs = predict_proba(clf, X_cls)
```
