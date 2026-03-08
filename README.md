# MichiBoost.jl

A pure Julia implementation of gradient boosting with ordered target statistics and symmetric (oblivious) decision trees, inspired by the [CatBoost](https://catboost.ai/) algorithm.

*Michi (ミチ) means cat in Japanese.*

## Features

- **Pure Julia** — no Python, no C++ bindings, no CondaPkg
- **Ordered target statistics** for native categorical feature handling without preprocessing
- **Ordered boosting** to prevent target leakage during gradient computation
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