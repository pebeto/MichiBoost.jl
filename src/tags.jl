"""
    MichiBoost.BoostingTypes

Singleton tag types for the `boosting_type` keyword argument. `Ordered` uses a
random permutation when computing categorical target statistics (reduces
leakage); `Plain` encodes on the full training set. CatBoost-style strings
(`"Ordered"`, `"Plain"`) and the matching Symbols (`:Ordered`, `:Plain`) are
still accepted at the wrapper boundary.
"""
module BoostingTypes

abstract type BoostingType end

struct Ordered <: BoostingType end
struct Plain <: BoostingType end

end  # module BoostingTypes

using .BoostingTypes: BoostingType

_boosting_name(::Type{BoostingTypes.Ordered}) = "Ordered"
_boosting_name(::Type{BoostingTypes.Plain}) = "Plain"

"""
    MichiBoost.AutoClassWeights

Singleton tag types for the `auto_class_weights` keyword argument. `Balanced`
sets each class weight to `n / (n_classes * count[c])`; `SqrtBalanced` uses
`sqrt(n / count[c])`. CatBoost-style strings and the matching Symbols
(`:Balanced`, `:SqrtBalanced`) still work.
"""
module AutoClassWeights

abstract type AutoClassWeightMode end

struct Balanced <: AutoClassWeightMode end
struct SqrtBalanced <: AutoClassWeightMode end

end  # module AutoClassWeights

using .AutoClassWeights: AutoClassWeightMode

_auto_class_weight_name(::Type{AutoClassWeights.Balanced}) = "Balanced"
_auto_class_weight_name(::Type{AutoClassWeights.SqrtBalanced}) = "SqrtBalanced"

"""
    MichiBoost.PredictionTypes

Singleton tag types for the `prediction_type` keyword argument of `predict`.
`Class` returns regression values or predicted class labels; `Probability`
returns probabilities (classification only); `RawFormulaVal` returns raw logits
or scores before any transformation. CatBoost-style strings and the matching
Symbols (`:Class`, `:Probability`, `:RawFormulaVal`) still work.
"""
module PredictionTypes

abstract type PredictionType end

struct Class <: PredictionType end
struct Probability <: PredictionType end
struct RawFormulaVal <: PredictionType end

end  # module PredictionTypes

using .PredictionTypes: PredictionType

_prediction_name(::Type{PredictionTypes.Class}) = "Class"
_prediction_name(::Type{PredictionTypes.Probability}) = "Probability"
_prediction_name(::Type{PredictionTypes.RawFormulaVal}) = "RawFormulaVal"
