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

The `predict` function accepts a `prediction_type` keyword argument:

- `"Class"` (default) — regression values, or predicted class labels for classifiers.
- `"Probability"` — predicted probabilities (classification only).
- `"RawFormulaVal"` — raw logits / scores before any transformation.
