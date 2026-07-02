# Models

```@meta
CurrentModule = MichiBoost
```

## Regression

```@docs
MichiBoostRegressor
```

## Classification

```@docs
MichiBoostClassifier
```

## Fitted Model

```@docs
MichiBoostModel
```

## Built-in Regression Losses

Parametrized regression losses. Pass an instance as `loss_function`, e.g.
`MichiBoostRegressor(; loss_function=Quantile(0.9))`. See the
[Regression](../guide/regression.md) guide for when to reach for each.

```@docs
Huber
Quantile
Expectile
MAPE
Tweedie
LogLinQuantile
RMSEWithUncertainty
```

## Built-in Classification Losses

```@docs
MultiClassOneVsAll
```

## Custom Losses

Subtype [`LossFunction`](@ref) and add methods to the functions below to plug a
new loss into the engine. The [Custom Loss Functions](@ref) guide walks through
regression, binary, and multi-class examples.

```@docs
LossFunction
gradient_hessian!
initial_prediction
loss
task_type
link_inverse
```
