# MichiBoost.jl

A pure Julia implementation of gradient boosting with ordered target statistics and symmetric (oblivious) decision trees, inspired by the [CatBoost](https://catboost.ai/) algorithm.

_Michi (ミチ) means cat in Japanese._

## Features

- **Pure Julia** gradient boosting with no Python or C++ dependencies.
- **Native categorical handling** via ordered target encoding (CatBoost-style),
  with no manual preprocessing.
- **Symmetric (oblivious) decision trees** as the base learner, with
  histogram-based split finding and threaded training.
- **Custom loss functions** via the [`LossFunction`](@ref) interface for
  regression, binary, and multi-class tasks.
- **Standard toolkit**: sample and class weights, cross-validation (with
  stratification), early stopping with configurable eval metrics, SHAP
  values, model serialisation. See the [User Guide](guide/regression.md)
  and [API Reference](api/models.md) for the full list.

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
