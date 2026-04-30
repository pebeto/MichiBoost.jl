# Prediction

```@meta
CurrentModule = MichiBoost
```

## Prediction Functions

```@docs
predict
predict_proba
predict_classes
```

The [`predict`](@ref) function accepts a `prediction_type` keyword argument.
Pass either a `PredictionTypes.*` tag or its CatBoost-style string:

- `PredictionTypes.Class` / `"Class"` (default) — regression values, or
  predicted class labels for classifiers.
- `PredictionTypes.Probability` / `"Probability"` — predicted probabilities
  (classification only).
- `PredictionTypes.RawFormulaVal` / `"RawFormulaVal"` — raw logits / scores
  before any transformation.

## Scoring

```@docs
score
```
